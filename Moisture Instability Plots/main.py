import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os
from NN import train_model,SimpleNN
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Load and prepare data
df = pd.read_csv("data.csv").dropna()
df['y0'] = df['x0']

x_columns = ['x0','x3','x6','x9','x12','x15','x18','x23']
y_columns = ['y0','y3','y6','y9','y12','y15','y18','y23']

X = df['Timestamp'].values.flatten()

THRESHOLD = 0.25
POWER = 0.4
# Storage
results = {}

def load_or_train_model(X, y, col_name):
    model_path = f"model_{col_name}.pt"
    model = SimpleNN()

    # Normalize X and y
    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()

    X_norm = (X - X_mean) / X_std

    if os.path.exists(model_path):
        print(f"Loading existing model for {col_name}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print(f"Training new model for {col_name}")
        train_model(X, y, col_name)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # Inference
    X_tensor = torch.tensor(X_norm, dtype=torch.float32, requires_grad=True)
    y_pred_norm = model(X_tensor)

    dy_dx = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=X_tensor,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True
    )[0]

    y_pred = y_pred_norm.detach().numpy() 
    dy_dx = dy_dx.detach().numpy() 

    return y_pred, dy_dx, X_mean, X_std, y_mean, y_std


def linear_regression(x, y):
    if x.size == 0:
        return 0.0, 0.0

    n = x.size
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.dot(x, y)
    sum_xx = np.dot(x, x)

    denominator = n * sum_xx - sum_x**2
    if denominator == 0:
        return 0.0, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    return slope, intercept

def calculate_cd(x, y, slope, intercept):
    if x.size == 0:
        return 0.0

    predicted_y = slope * x + intercept
    ss_res = np.sum((y - predicted_y)**2)
    ss_tot = np.sum((y - np.mean(y))**2)

    return 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot

def data_fitting_ransac(x_values, y_values, threshold=0.001, iterations=100):
    x_values = np.asarray(x_values, dtype=np.float32)
    y_values = np.asarray(y_values, dtype=np.float32)
    
    if x_values.size != y_values.size or x_values.size < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    n = x_values.size
    best_slope = 0.0
    best_intercept = 0.0
    best_cd = 0.0
    best_inlier_indices = None
    best_inlier_count = -1

    for _ in range(iterations):
        # 1. Randomly sample 2 points
        sample_indices = np.random.choice(n, 2, replace=False)
        x_sample = x_values[sample_indices]
        y_sample = y_values[sample_indices]

        # Avoid vertical lines
        if x_sample[0] == x_sample[1]:
            continue

        # 2. Fit a line to the sample
        slope = (y_sample[1] - y_sample[0]) / (x_sample[1] - x_sample[0])
        intercept = y_sample[0] - slope * x_sample[0]

        # 3. Find all inliers in the full dataset
        predicted = slope * x_values + intercept
        residuals = np.abs(y_values - predicted)
        inlier_indices = np.where(residuals < threshold)[0]
        inlier_count = len(inlier_indices)
        
        # 4. Check if this is the best model so far
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_indices = inlier_indices
    
    # 5. Refit the model using all inliers from the best model found
    if best_inlier_count > 1:
        x_inliers = x_values[best_inlier_indices]
        y_inliers = y_values[best_inlier_indices]

        # Use the robust linear regression on all found inliers
        refined_slope, refined_intercept = linear_regression(x_inliers, y_inliers)
        cd = calculate_cd(x_inliers, y_inliers, refined_slope, refined_intercept)
        
        # The start/end x values are now from the inliers
        start_x = np.min(x_inliers)
        end_x = np.max(x_inliers)
        
        return refined_slope, refined_intercept, cd, start_x, end_x
    else:
        # Failed to find a model
        return 0.0, 0.0, 0.0, 0.0, 0.0




X = df['Timestamp'].values.flatten()

def process_column(column_name: str):
    print(f"Processing model for column = {column_name}")
    y = df[column_name].values.flatten()
    try:
        y_pred, dy_dx, X_mean, X_std, y_mean, y_std = load_or_train_model(X, y, column_name)
        results[column_name] = {
            'pred': y_pred.flatten(),
            'dy_dx': dy_dx.flatten(),
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
    except Exception as e:
        print(f"Failed on {column_name}: {e}")

# --- Process all columns ---
for col in x_columns + y_columns:
    process_column(col)




# ---- Helper Function ----
def sliding_gradient_change(df, columns):
    markers = []

    for col in columns:
        prev_grad = None

        for i in range(0, len(df)):
            # Linear regression: gradient = slope
            y_std = results[col]['y_std']
            x_std = results[col]['X_std']
            grad = (results[col]['dy_dx'][i] * y_std) / (x_std)

            # Compare with previous window
            if prev_grad is not None:
                rel_change = abs((grad - prev_grad) / (prev_grad + 1e-8))
                if rel_change >= THRESHOLD:
                    timestamp = X[i]
                    y_mean = results[col]['y_mean']
                    temperature = (results[col]['pred'][i]*y_std)+y_mean
                    markers.append({
                        'Timestamp': timestamp,
                        'Temperature': temperature,
                        'Sensor': col,
                        'CurrentGrad': grad,
                        'PrevGrad': prev_grad
                    })

            prev_grad = grad

    return pd.DataFrame(markers)

def add_analysis_traces(fig, df, results, column_name, color, label, tr_table_data):
    """
    Calculates and adds analysis traces (markers, regression lines) for a single column to a figure.
    Populates tr_table_data with (Timestamp, Sensor, TR) values.
    """
    marker_df = sliding_gradient_change(df, [column_name])

    if marker_df.empty:
        return

    prevTimestep = 1
    is_first_marker = True
    for _, row in marker_df.iterrows():
        timestamp = row['Timestamp']
        sensor = row['Sensor']
        try:
            result = results[sensor]
            idx_list = df[df['Timestamp'] == timestamp].index
            if not idx_list.any():
                continue
            idx = idx_list[0]

            x0 = timestamp
            y0 = df[sensor].iloc[idx]

            x_range = df['Timestamp'][(df['Timestamp'] >= prevTimestep) & (df['Timestamp'] <= timestamp) & (df['Timestamp'] % 10 == 0)].values
            x_range = x_range[x_range > 0]
            if len(x_range) < 2:
                continue

            df_range = df[df['Timestamp'].isin(x_range)]
            y_values = df_range[sensor].values
            x_range_log = np.log(x_range)

            best_slope, best_intercept, _, _, _ = data_fitting_ransac(x_values=x_range_log, y_values=y_values)
            if best_slope == 0 and best_intercept == 0:
                continue

            t1 = best_slope * np.log(100) + best_intercept
            t2 = best_slope * np.log(1000) + best_intercept
            TR = 4 * np.pi * (t2 - t1) / (2.303 * POWER)

            # Append to TR table
            tr_table_data.append((f"{sensor}", f"{timestamp}", f"{TR:.3f}"))

            y_best_fit = x_range_log * best_slope + best_intercept

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_best_fit,
                mode='lines',
                line=dict(color=color, dash='dot'),
                name=f'Fit {sensor}',
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[x0], y=[y0],
                mode='markers',
                marker=dict(size=10, color=color, symbol='x'),
                name=label,
                customdata=[[sensor, row['CurrentGrad'], row['PrevGrad'], TR]],
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Temperature: %{y:.2f}°C<br>"
                    "Sensor: %{customdata[0]}<br>"
                    "CurrentGrad: %{customdata[1]:.3f}<br>"
                    "PrevGrad: %{customdata[2]:.3f}<br>"
                    "TR: %{customdata[3]:.3f}"
                ),
                showlegend=is_first_marker
            ))

            is_first_marker = False
            prevTimestep = timestamp
        except Exception as e:
            print(f"Could not compute regression for {sensor} at {timestamp}: {e}")

from plotly.subplots import make_subplots

for x_col, y_col in zip(x_columns, y_columns):
    fig = make_subplots(
    rows=1, cols=2,
    horizontal_spacing=0.03,
    specs=[[{"type": "scatter"},{"type": "table"}]],column_widths=[0.7, 0.3]
)
    tr_table_data = []

    # Add main data traces
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df[x_col], mode='lines',
        name=x_col, line=dict(color='royalblue')
    ),row=1,col=1)
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df[y_col], mode='lines',
        name=y_col, line=dict(color='firebrick')
    ),row=1,col=1)

    # Add analysis traces for the x-column
    add_analysis_traces(fig, df, results, x_col, color='darkblue', label=f'Shift ({x_col})', tr_table_data=tr_table_data)

    # Add analysis traces for the y-column
    if x_col != y_col:
        add_analysis_traces(fig, df, results, y_col, color='darkred', label=f'Shift ({y_col})', tr_table_data=tr_table_data)

    # Add TR Table as a side panel
    if tr_table_data:
        fig.add_trace(go.Table(
            header=dict(values=["Sensor", "Timestamp", "TR"],
                        fill_color='lightgrey', align='left'),
            cells=dict(values=list(zip(*tr_table_data)),
                       fill_color='white', align='left'),
            name="TR Info"
        ),row=1, col=2)

    # Update layout
    fig.update_layout(
        title=f'Analysis for {x_col} and {y_col} vs. Time',
        xaxis_title="Timestamp (log scale)",
        yaxis_title="Value",
        xaxis_type="log",
        yaxis_type="linear",
        yaxis=dict(range=[20, 70]),
        template="plotly_white",
        legend_title="Sensor",
    )

    fig.show()
