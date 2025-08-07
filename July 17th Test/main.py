import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import torch
import os
from NN import train_model,SimpleNN
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.signal import savgol_filter
from scipy.stats import linregress

# Load and prepare data
df = pd.read_csv("data.csv").dropna()
df['y0'] = df['x0']
df = df[df['Timestamp'] % 10 == 0]


x_columns = ['x0','x3','x6','x9','x12','x15','x18','x23']
y_columns = ['y0','y3','y6','y9','y12','y15','y18','y23']

X = df['Timestamp'].values.flatten()

THRESHOLD = 0.05
POWER = 0.4
# Storage
results = {}


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

    x_inliers = x_values
    y_inliers = y_values

    # Use the robust linear regression on all found inliers
    refined_slope, refined_intercept = linear_regression(x_inliers, y_inliers)
    cd = calculate_cd(x_inliers, y_inliers, refined_slope, refined_intercept)
    
    # The start/end x values are now from the inliers
    start_x = np.min(x_inliers)
    end_x = np.max(x_inliers)
    
    return refined_slope, refined_intercept, cd, start_x, end_x



X = df['Timestamp'].values.flatten()

def sliding_window_regression(column):
    y = df[column]

    for i in range(0,len(y)-6,6):
        temp_window = y[i:i+6]
        timestep_window = X[i:i+6]
        slope, intercept, r_value, p_value, std_err = linregress(timestep_window,temp_window)



# ---- Helper Function ----
def sliding_gradient_change(df, columns):
    markers = []

    for col in columns:
        prev_grad = None
        temps = df[col].values.flatten()

        i = 0
        while i < len(df) - 1:
            timestep = X[i]

            # Dynamically determine window size
            if timestep >= 10000:
                window = 10000
            elif timestep >= 1000:
                window = 1000
            elif timestep >= 100:
                window = 100
            elif timestep >= 10:
                window = 10
            else:
                window = 6  # default fallback

            # Ensure the window fits in the array
            end_idx = i + window
            if end_idx >= len(df):
                break

            timestep_window = X[i:end_idx]
            temp_window = temps[i:end_idx]

            # Skip if log(X) has non-positive values
            if np.any(np.array(timestep_window) <= 0):
                i += 1
                continue

            grad, intercept, r_value, p_value, std_err = linregress(np.log(timestep_window), temp_window)

            # Compare with previous window
            if prev_grad is not None:
                rel_change = abs((grad - prev_grad) / (prev_grad + 1e-8))
                if rel_change >= THRESHOLD:
                    markers.append({
                        'Timestamp': X[i],
                        'Temperature': temps[i],
                        'Sensor': col,
                        'CurrentGrad': grad,
                        'PrevGrad': prev_grad,
                        'Intercept': intercept,
                        'Start X': X[i],
                        'End X': X[end_idx],
                        'cd': r_value
                    })

            prev_grad = grad
            i += 1  # Move forward by 1 for overlap

    return pd.DataFrame(markers)


def add_analysis_traces(fig, df,  column_name, color, tr_table_data):
    """
    Calculates and adds analysis traces (markers, regression lines) for a single column to a figure.
    Populates tr_table_data with (Timestamp, Sensor, TR) values.
    """
    marker_df = sliding_gradient_change(df, [column_name])

    if marker_df.empty:
        return

    is_first_marker = True
    for _, row in marker_df.iterrows():
        timestamp = row['Timestamp']
        sensor = row['Sensor']
        best_slope = row['CurrentGrad']
        best_intercept = row['Intercept']
        start_x = row['Start X']
        end_x = row['End X']
        try:
            x_range = df['Timestamp'][(df['Timestamp'] >= start_x) & (df['Timestamp'] <= end_x)].values
            x_range = x_range[x_range > 0]
            if len(x_range) < 2:
                continue

            x_range_log = np.log(x_range)

            if best_slope == 0 and best_intercept == 0:
                continue

            t1 = best_slope * np.log(100) + best_intercept
            t2 = best_slope * np.log(1000) + best_intercept
            TR = 4 * np.pi * (t2 - t1) / (2.303 * POWER)

            # Append to TR table
            tr_table_data.append((f"{sensor}", f"{timestamp}", f"{TR:.3f}"))

            y_best_fit = x_range_log * best_slope + best_intercept

            custom_data = [[sensor, row['CurrentGrad'], row['PrevGrad'], TR,row['cd']]] * len(x_range)

            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_best_fit,
                mode='lines',
                line=dict(color=color, dash='dot'),
                name=f'Fit {sensor}',
                customdata=custom_data,
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Temperature: %{y:.2f}°C<br>"
                    "Sensor: %{customdata[0]}<br>"
                    "CurrentGrad: %{customdata[1]:.3f}<br>"
                    "PrevGrad: %{customdata[2]:.3f}<br>"
                    "TR: %{customdata[3]:.3f}<br>"
                    "CD: %{customdata[4]:.3f}"
                ),
                showlegend=is_first_marker
            ))

            is_first_marker = False
        except Exception as e:
            print(f"Could not compute regression for {sensor} at {timestamp}: {e}")

from plotly.subplots import make_subplots

for x_col, y_col in zip(x_columns, y_columns):
    df[x_col] = savgol_filter(df[x_col],51,3)
    df[y_col] = savgol_filter(df[y_col],51,3)
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
    add_analysis_traces(fig, df,  x_col, color='darkblue',  tr_table_data=tr_table_data)

    # Add analysis traces for the y-column
    if x_col != y_col:
        add_analysis_traces(fig, df, y_col, color='darkred', tr_table_data=tr_table_data)

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
