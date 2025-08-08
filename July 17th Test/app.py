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
THRESHOLD = 0.05  # (kept, but no longer used for picking ranges)
POWER = 0.4
SMOOTH_WINDOW = 51   # must be odd and <= series length
SMOOTH_POLY = 3

# Log-cycle options
CYCLES = {
    "(1–10)": (1, 10),
    "(10–100)": (10, 100),
    "(100–1000)": (100, 1000),
    "(1000–10000)": (1000, 10000),
    "(10000–100000)": (10000, 100000),
}

# -------------------------
# Load + prepare data
# -------------------------
df = pd.read_csv(DATA_PATH).dropna()
# Preserve your behavior
df["y0"] = df["x0"]
df = df[df["Timestamp"] % 10 == 0].copy()

X = df["Timestamp"].values.astype(float)

# discover available sensors (any column like xN / yN)
sensor_cols = [c for c in df.columns if (c.startswith("x") or c.startswith("y")) and c[1:].isdigit()]
sensor_cols.sort(key=lambda c: (c[0], int(c[1:])))

# -------------------------
# Helpers
# -------------------------
def fit_in_cycle(sensor: str, start_t: float, end_t: float):
    """
    Linear regression of sensor vs ln(t) ONLY within [start_t, end_t].
    Returns a plotly trace (dotted fit) or None if not enough points.
    The TR is baked into the legend label.
    """
    mask = (df["Timestamp"] >= start_t) & (df["Timestamp"] <= end_t) & (df["Timestamp"] > 0)
    x_range = df.loc[mask, "Timestamp"].values
    if x_range.size < 2:
        return None

    y_range = df.loc[mask, sensor].values
    x_log = np.log(x_range)

    slope, intercept, r, p, se = linregress(x_log, y_range)

    # Best-fit line plotted over available x_range
    y_fit = slope * x_log + intercept

    # TR from selected cycle
    t1 = slope * np.log(start_t) + intercept
    t2 = slope * np.log(end_t) + intercept
    TR = 4 * np.pi * (t2 - t1) / (2.303 * POWER)

    legend_label = f"Fit {sensor} [{int(start_t)}–{int(end_t)}] • TR={TR:.3f} "

    return go.Scatter(
        x=x_range,
        y=y_fit,
        mode="lines",
        line=dict(dash="dot"),
        name=legend_label,
        hovertemplate=(
            "Timestamp: %{x}<br>"
            "Temperature (fit): %{y:.2f}°C<br>"
            f"Sensor: {sensor}<br>"
            f"Cycle: [{int(start_t)}–{int(end_t)}]<br>"
            f"TR: {TR:.3f}<br>"
            f"R: {r:.3f}"
        ),
        showlegend=True,
    )

# -------------------------
# Dash app
# -------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={
        "height": "100vh",
        "width": "100vw",
        "display": "flex",
        "flexDirection": "column",
        "margin": "0",
        "padding": "0",
    },
    children=[
        # Controls bar
        html.Div(
            style={
                "padding": "8px 12px",
                "display": "flex",
                "gap": "12px",
                "alignItems": "center",
                "flexWrap": "wrap",
                "borderBottom": "1px solid #eee",
            },
            children=[
                html.Div("Select sensors:", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="sensors-select",
                    options=[{"label": s, "value": s} for s in sensor_cols],
                    value=sensor_cols[:2] if len(sensor_cols) >= 2 else sensor_cols,
                    multi=True,
                    placeholder="Select one or more sensors…",
                    style={"minWidth": "420px"},
                ),
                html.Div("Log cycle:", style={"fontWeight": 600}),
                dcc.Dropdown(
                    id="cycle-select",
                    options=[{"label": k, "value": k} for k in CYCLES.keys()],
                    value="(100–1000)",
                    clearable=False,
                    style={"width": "220px"},
                ),
            ],
        ),

        # Full-screen graph area
        html.Div(
            style={
                "flex": "1 1 auto",
                "minHeight": 0,  # important for flexbox to allow graph to size correctly
            },
            children=[
                dcc.Graph(
                    id="multi-sensor-graph",
                    style={"height": "100%", "width": "100%"},
                    config={"responsive": True},
                )
            ],
        ),
    ],
)

@app.callback(
    Output("multi-sensor-graph", "figure"),
    Input("sensors-select", "value"),
    Input("cycle-select", "value"),
)
def update_plot(selected_sensors, cycle_key):
    fig = go.Figure()

    if not selected_sensors:
        fig.update_layout(
            title="Select one or more sensors",
            template="plotly_white",
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(x=1.02, xanchor="left", y=1, orientation="v"),
        )
        return fig

    start_t, end_t = CYCLES[cycle_key]

    # Plot each selected sensor (smoothed main line)
    for sensor in selected_sensors:
        series = df[sensor].copy()
        try:
            if len(series) >= SMOOTH_WINDOW:
                series = pd.Series(
                    savgol_filter(series, SMOOTH_WINDOW, SMOOTH_POLY), index=series.index
                )
        except ValueError:
            pass

        # Main line
        fig.add_trace(
            go.Scatter(
                x=df["Timestamp"],
                y=series,
                mode="lines",
                name=sensor,
                hovertemplate="Timestamp: %{x}<br>Value: %{y:.2f}°C",
            )
        )

        # Dotted fit for the chosen cycle; legend includes TR
        fit_trace = fit_in_cycle(sensor, start_t, end_t)
        if fit_trace is not None:
            fig.add_trace(fit_trace)

    # Layout: log-x, full-screen, legend on the right
    fig.update_layout(
        title=f"Sensors vs Time (analysis over {cycle_key})",
        xaxis_title="Timestamp (log scale)",
        yaxis_title="Value",
        template="plotly_white",
        legend_title="Legend",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(x=1.02, xanchor="left", y=1, orientation="v"),  # right-side legend
    )
    fig.update_xaxes(type="log")
    fig.update_yaxes(range=[20, 70])

    return fig


if __name__ == "__main__":
    # Make Plotly responsive; Dash handles the viewport
    app.run(debug=True)
