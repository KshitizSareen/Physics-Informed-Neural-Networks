import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

# -------------------------
# Config / constants
# -------------------------
DATA_PATH = "data.csv"
THRESHOLD = 0.05
POWER = 0.4
SMOOTH_WINDOW = 51   # must be odd and <= series length
SMOOTH_POLY = 3

# -------------------------
# Load + prepare data
# -------------------------
df = pd.read_csv(DATA_PATH).dropna()
df["y0"] = df["x0"]
df = df[df["Timestamp"] % 10 == 0].copy()

X = df["Timestamp"].values.flatten()

# discover available sensors (any column like xN / yN)
sensor_cols = [c for c in df.columns if (c.startswith("x") or c.startswith("y")) and c[1:].isdigit()]
sensor_cols.sort(key=lambda c: (c[0], int(c[1:])))


# -------------------------
# Helpers
# -------------------------
def sliding_gradient_change(df_in: pd.DataFrame, columns):
    """Find gradient-change markers per column using dynamic windows on log(t)."""
    markers = []
    for col in columns:
        temps = df_in[col].values.flatten()
        i = 0
        prev_grad = 0

        while i < len(df_in) - 1:
            timestep = X[i]
            if timestep >= 10000:
                window = 10000
            elif timestep >= 1000:
                window = 1000
            elif timestep >= 100:
                window = 100
            elif timestep >= 10:
                window = 10
            else:
                window = 6

            end_idx = i + window
            if end_idx >= len(df_in):
                break

            t_win = X[i:end_idx]
            y_win = temps[i:end_idx]

            if np.any(np.array(t_win) <= 0):
                i += 1
                continue

            grad, intercept, r_value, p_value, std_err = linregress(np.log(t_win), y_win)

            if prev_grad is not None:
                rel_change = abs((grad - prev_grad) / (prev_grad + 1e-8))
                if rel_change >= THRESHOLD and r_value >= 0.9:
                    markers.append({
                        "Timestamp": X[i],
                        "Temperature": temps[i],
                        "Sensor": col,
                        "CurrentGrad": grad,
                        "PrevGrad": prev_grad,
                        "Intercept": intercept,
                        "Start X": X[i],
                        "End X": X[end_idx],
                        "cd": r_value
                    })
                    prev_grad = grad
            i += 1

    return pd.DataFrame(markers)


def analysis_for_sensor(sensor: str):
    """
    Returns:
      - list of plotly traces (regression dotted lines),
      - TR table rows (list of dicts),
      - marker trace for detected change points (optional).
    """
    traces = []
    tr_rows = []
    markers_trace = None

    marker_df = sliding_gradient_change(df, [sensor])
    if marker_df.empty:
        return traces, tr_rows, markers_trace

    for first, (_, row) in enumerate(marker_df.iterrows()):
        start_x, end_x = row["Start X"], row["End X"]
        x_range = df["Timestamp"][(df["Timestamp"] >= start_x) & (df["Timestamp"] <= end_x)].values
        x_range = x_range[x_range > 0]
        if len(x_range) < 2:
            continue

        slope = row["CurrentGrad"]
        intercept = row["Intercept"]
        if slope == 0 and intercept == 0:
            continue

        x_log = np.log(x_range)
        y_fit = x_log * slope + intercept

        # TR from 100s to 1000s
        t1 = slope * np.log(100) + intercept
        t2 = slope * np.log(1000) + intercept
        TR = 4 * np.pi * (t2 - t1) / (2.303 * POWER)

        tr_rows.append({"Sensor": sensor, "Timestamp": f"{row['Timestamp']}", "TR": f"{TR:.3f}"})

        traces.append(
            go.Scatter(
                x=x_range,
                y=y_fit,
                mode="lines",
                line=dict(dash="dot"),
                name=f"Fit {sensor}",
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Temperature: %{y:.2f}°C<br>"
                    f"Sensor: {sensor}<br>"
                    f"TR: {TR:.3f}<br>"
                    f"CD: {row['cd']:.3f}"
                ),
                showlegend=(first == 0),
            )
        )

    # marker positions at change points
    join = pd.merge(marker_df[["Timestamp"]], df[["Timestamp", sensor]], on="Timestamp", how="left")
    markers_trace = go.Scatter(
        x=join["Timestamp"],
        y=join[sensor],
        mode="markers",
        marker=dict(size=8, symbol="circle-open"),
        name=f"Change points ({sensor})",
        hovertemplate="Timestamp: %{x}<br>Value: %{y:.2f}",
        showlegend=True
    )

    return traces, tr_rows, markers_trace


# -------------------------
# Dash app
# -------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
    children=[
        html.H2("Plot multiple sensors on one graph (with analysis)"),
        dcc.Dropdown(
            id="sensors-select",
            options=[{"label": s, "value": s} for s in sensor_cols],
            value=sensor_cols[:2] if len(sensor_cols) >= 2 else sensor_cols,  # default few
            multi=True,
            placeholder="Select one or more sensors…",
            style={"marginBottom": "12px", "maxWidth": "600px"},
        ),
        dcc.Graph(id="multi-sensor-graph"),
        html.H3("TR Info"),
        dash_table.DataTable(
            id="tr-table",
            columns=[
                {"name": "Sensor", "id": "Sensor"},
                {"name": "Timestamp", "id": "Timestamp"},
                {"name": "TR", "id": "TR"},
            ],
            data=[],
            sort_action="native",
            page_size=12,
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontFamily": "monospace"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"},
        ),
    ]
)

@app.callback(
    Output("multi-sensor-graph", "figure"),
    Output("tr-table", "data"),
    Input("sensors-select", "value"),
)
def update_plot(selected_sensors):
    fig = go.Figure()
    tr_rows = []

    if not selected_sensors:
        fig.update_layout(title="Select one or more sensors", template="plotly_white")
        return fig, tr_rows

    # Plot each selected sensor (smoothed if possible)
    for sensor in selected_sensors:
        series = df[sensor].copy()
        try:
            if len(series) >= SMOOTH_WINDOW:
                series = pd.Series(
                    savgol_filter(series, SMOOTH_WINDOW, SMOOTH_POLY), index=series.index
                )
        except ValueError:
            pass

        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=series,
                mode="lines",
                name=sensor
            )
        )

        # analysis traces + markers
        traces, tr_rows_sensor, markers_trace = analysis_for_sensor(sensor)
        tr_rows.extend(tr_rows_sensor)
        for t in traces:
            fig.add_trace(t)
        if markers_trace is not None:
            fig.add_trace(markers_trace)

    fig.update_layout(
        title="Selected sensors vs Time",
        xaxis_title="Timestamp (log scale)",
        yaxis_title="Value",
        template="plotly_white",
        legend_title="Legend",
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[20, 70])

    return fig, tr_rows


if __name__ == "__main__":
    app.run(debug=True)
