import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os
from NN import train_model,SimpleNN
import numpy as np

# Load and prepare data
df = pd.read_csv("data.csv").dropna()
df['y0'] = df['x0']

x_columns = ['x0','x3','x6','x9','x12','x15','x18','x23']
y_columns = ['y0','y3','y6','y9','y12','y15','y18','y23']

X = df['Timestamp'].values.flatten()

THRESHOLD = 0.05

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


# --- X input is timestamp ---
X = df['Timestamp'].values.flatten()

# --- Process x_columns ---
for x_col in x_columns:
    print(f"Processing model for x_col = {x_col}")
    y = df[x_col].values.flatten()
    try:
        y_pred, dy_dx, X_mean, X_std, y_mean, y_std = load_or_train_model(X, y, x_col)
        results[x_col] = {
            'pred': y_pred.flatten(),
            'dy_dx': dy_dx.flatten(),
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
    except Exception as e:
        print(f"Failed on x_col={x_col}: {e}")

# --- Process y_columns ---
for y_col in y_columns:
    print(f"Processing model for y_col = {y_col}")
    y = df[y_col].values.flatten()
    try:
        y_pred, dy_dx, X_mean, X_std, y_mean, y_std = load_or_train_model(X, y, y_col)
        results[y_col] = {
            'pred': y_pred.flatten(),
            'dy_dx': dy_dx.flatten(),
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
    except Exception as e:
        print(f"Failed on y_col={y_col}: {e}")




# ---- Helper Function ----
def sliding_gradient_change(df, columns):
    markers = []

    for col in columns:
        prev_grad = None

        for i in range(0, len(df)):
            # Linear regression: gradient = slope
            grad = results[col]['dy_dx'][i]

            # Compare with previous window
            if prev_grad is not None:
                rel_change = abs((grad - prev_grad) / (prev_grad + 1e-8))
                if rel_change >= THRESHOLD:
                    timestamp = X[i]
                    y_mean = results[col]['y_mean']
                    y_std = results[col]['y_std']
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

# ---- Build model prediction DataFrame for y_columns ----
pred_df_y = pd.DataFrame({'Timestamp': df['Timestamp']})

for col in y_columns:
    if col in results:
        y_mean = results[col]['y_mean']
        y_std = results[col]['y_std']
        pred_df_y[col] = (results[col]['pred']*y_std)+y_mean
    else:
        print(f"Prediction for {col} not found in results.")

# ---- Y Data ----
df_y_pred = pred_df_y.melt(id_vars='Timestamp', var_name='Sensor', value_name='Temperature')
marker_df_y = sliding_gradient_change(df, y_columns)

# ---- Plot predictions instead of raw data ----
fig_y = px.line(df_y_pred, x='Timestamp', y='Temperature', color='Sensor', title='Y Data Over Time')


if not marker_df_y.empty:
    for _, row in marker_df_y.iterrows():
        timestamp = row['Timestamp']
        sensor = row['Sensor']
        try:
            result = results[sensor]
            idx = df[df['Timestamp'] == timestamp].index[0]

            x0 = timestamp
            y0 = result['pred'][idx]
            slope = result['dy_dx'][idx]

            # Normalization params
            X_mean = result['X_mean']
            X_std = result['X_std']
            y_mean = result['y_mean']
            y_std = result['y_std']

            logrange = 0

                # Define small local window around x0
            # Define small local window around x0
            if timestamp<=10:
                logrange=1
            elif timestamp<=100:
                logrange=10
            elif timestamp<=1000:
                logrange=100
            elif timestamp<=10000:
                logrange=1000
            elif timestamp<=100000:
                logrange=10000
            x_range = df['Timestamp'][(df['Timestamp'] >= x0 - logrange) & (df['Timestamp'] <= x0 + logrange)].values


            # Normalize x_range
            x_range_norm = (x_range - X_mean) / X_std

            intercept = y0 - slope * ((timestamp - X_mean)/X_std)

            # Compute tangent line in normalized space, then unnormalize output
            y_tangent = slope * (x_range_norm) + intercept
            y_tangent = y_tangent * y_std + y_mean  # unnormalize prediction



            # Add marker at point of tangency
            fig_y.add_trace(go.Scatter(
                x=[x0],
                y=[(y0*y_std)+y_mean],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Gradient Shift ≥ 5% (Y)',
                customdata=[[sensor, row['CurrentGrad'], row['PrevGrad']]],
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Temperature: %{y:.2f}°C<br>"
                    "Sensor: %{customdata[0]}<br>"
                    "CurrentGrad: %{customdata[1]}<br>"
                    "PrevGrad: %{customdata[2]}<br>"
                ),
                showlegend=False
            ))
        except Exception as e:
            print(f"Could not compute tangent for {sensor} at {timestamp}: {e}")

fig_y.update_layout(
    yaxis_type="linear",  # or "linear"
    xaxis_type="log",  # if you want log-time
    template="plotly_white"
)

fig_y.show()

# ---- Build model prediction DataFrame for y_columns ----
pred_df_x = pd.DataFrame({'Timestamp': df['Timestamp']})

for col in x_columns:
    if col in results:
        y_mean = results[col]['y_mean']
        y_std = results[col]['y_std']
        pred_df_x[col] = (results[col]['pred']*y_std)+y_mean
    else:
        print(f"Prediction for {col} not found in results.")

# ---- Y Data ----
df_x_pred = pred_df_x.melt(id_vars='Timestamp', var_name='Sensor', value_name='Temperature')
marker_df_x = sliding_gradient_change(df, x_columns)

# ---- Plot predictions instead of raw data ----
fig_x = px.line(df_x_pred, x='Timestamp', y='Temperature', color='Sensor', title='X Data Over Time')

if not marker_df_x.empty:
    for _, row in marker_df_x.iterrows():
        timestamp = row['Timestamp']
        sensor = row['Sensor']
        try:
            result = results[sensor]
            idx = df[df['Timestamp'] == timestamp].index[0]

            x0 = timestamp
            y0 = result['pred'][idx]
            slope = result['dy_dx'][idx]

            # Normalization params
            X_mean = result['X_mean']
            X_std = result['X_std']
            y_mean = result['y_mean']
            y_std = result['y_std']

            intercept = y0 - slope * ((timestamp - X_mean)/X_std)
            # Compute tangent line in normalized space, then unnormalize output
            y_tangent = slope * (((X-X_mean)/X_std)) + intercept
            y_tangent = (y_tangent * y_std) + y_mean  # unnormalize prediction
            slope = (y_tangent[-1]-y_tangent[0])/(np.log(X[-1])-np.log(X[0]))
            t1 = (slope * np.log(1000)) + y_tangent[0]
            t2 = (slope * np.log(100)) + y_tangent[0]
            tr = ((4 * np.pi * (t1 - t2) )/ (2.303* 0.4))
            print(slope,intercept* y_std + y_mean,y_tangent[0])

            # Add marker at point of tangency
            fig_x.add_trace(go.Scatter(
                x=[x0],
                y=[(y0 * y_std) + y_mean],
                mode='markers',
                marker=dict(size=8, color='black', symbol='x'),
                name='Gradient Shift ≥ 5% (X)',
                customdata=[[sensor,tr]],
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Temperature: %{y:.2f}°C<br>"
                    "Sensor: %{customdata[0]}<br>"
                    "TR: %{customdata[1]}<br>"
                ),
                showlegend=False
            ))
        except Exception as e:
            print(f"Could not compute tangent for {sensor} at {timestamp}: {e}")

fig_x.update_layout(
    yaxis_type="linear",  # or "linear"
    xaxis_type="linear",  # if you want log-time
    template="plotly_white"
)

fig_x.show()

"""
# ---- Locate index of timestamp ----
timestamp_to_check = 500
index_array = df[df['Timestamp'] == timestamp_to_check].index
if len(index_array) == 0:
    raise ValueError("Timestamp not found.")
index = index_array[0]

# ---- Extract values ----
gradient_at_t_normalized = model_derivative.flatten()[index]
# Unnormalize dy/dx
gradient_at_t = gradient_at_t_normalized


y_at_timestep = y_pred.flatten()[index]

intercept = y_at_timestep - gradient_at_t * ((timestamp_to_check - X_mean)/X_std)
y_tangent = gradient_at_t * ((X - X_mean)/X_std) + intercept
"""