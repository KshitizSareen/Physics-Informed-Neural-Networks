import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
import mplcursors
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig = None
ax = None
bprev = None
bnext = None
callback = None

# Define constant R (not currently used, but kept for completeness)
R = 0.81  # Replace with your specific value

# Load data from CSV file
df = pd.read_csv("temperature readings 2025-04-07_11-51-21 - Dataset.csv", header=None)

# Sensor positions (used for naming columns)
sensorPositionsY = [0, 1, 2.5, 5, 15]
sensorPositionsX = [0, 1, 2.5, 5, 10, 15]

# Define column names
sensorColumns = [f"Sensor placed Horizontally at {x} and Vertically at {y}"
                 for y in sensorPositionsY for x in sensorPositionsX]
column_names = ["Timestamp", "Power"] + sensorColumns
df.columns = column_names

# Convert Timestamp to a 1D NumPy array (for easier handling with np.log, etc.)
Timestamp = df["Timestamp"].values  # shape: (N,)

def plot(axis):
    """
    Plots the specified column vs. Timestamp on the given axis.
    """
    # Clear the axis before plotting new data
    axis.clear()

    # Read start/end from the Tkinter input boxes
    start_val = float(entry_start.get())
    end_val = float(entry_end.get())

    row_num_val = entry_vertical.get()
    col_num_val = entry_horizontal.get()

    column = f"Sensor placed Horizontally at {col_num_val} and Vertically at {row_num_val}"

    # Calculate average I for X between start_val and end_val
    mask = (df['Timestamp'] >= start_val) & (df['Timestamp'] <= end_val)
    I = df.loc[mask, 'Power'].mean()

    if column in df.columns:

        # Filter X and y based on provided range
        X_filtered = df.loc[mask, "Timestamp"].values  # shape: (N,)
        y_filtered = df.loc[mask, column].values       # shape: (N,)

        # Safeguard: If there's insufficient data in the chosen range, skip
        if len(X_filtered) < 2:
            axis.text(0.5, 0.5, "Not enough data in the chosen range",
                    ha='center', va='center', transform=axis.transAxes)
            return

        # Reshape log(X_filtered) to 2D for sklearn
        logX_filtered = np.log(X_filtered).reshape(-1, 1)

        # Linear Regression fitting
        reg = LinearRegression()
        reg.fit(logX_filtered, y_filtered)

        # Predict across all Timestamps (full range), using log(Timestamp)
        # Reshape again for sklearn's predict
        y_pred_all = reg.predict(np.log(Timestamp).reshape(-1, 1))  # shape: (N,)

        # Calculate slope and intercept “manually” from the endpoints of the fitted line
        #   slope = (y2 - y1) / (ln(x2) - ln(x1))
        #   intercept = y1
        # Because we want to parametrize the line as: y = slope*ln(x) + intercept
        y1, y2 = y_pred_all[0], y_pred_all[-1]
        lnX1, lnX2 = np.log(Timestamp[0]), np.log(Timestamp[-1])
        slope = (y2 - y1) / (lnX2 - lnX1)
        y_intercept = y1  # So that at lnX1, we get y1

        # Recompute the fitted line based on slope/intercept
        y_pred_all = slope * np.log(Timestamp) + y_intercept

        # Calculate TR
        #  t1 = slope*ln(start_val) + intercept
        #  t2 = slope*ln(end_val)   + intercept
        #  TR = [4 * pi * (t2 - t1)] / [2.303 * I]
        t1 = slope * np.log(start_val) + y_intercept
        t2 = slope * np.log(end_val) + y_intercept
        TR = 4 * np.pi * (t2 - t1) / (2.303 * I)  # This should be a float

        # Plot the data
        axis.scatter(Timestamp, df[column].values, label='Data')
        lineOfBestFit = axis.semilogx(Timestamp, y_pred_all, 'g--', label='Fit')

        axis.set_ylim([0, 70])

        # Labeling
        axis.set_xlabel(f'Timestamp | TR: {TR:.4f}')
        axis.set_ylabel(column)
        axis.set_title(column)
        axis.legend()
        axis.grid(True)

        # Add hover functionality for the fitted line
        cursor = mplcursors.cursor(lineOfBestFit, hover=True)
        @cursor.connect("add")
        def _(sel):
            # sel.target is [x, y]; format them as desired
            sel.annotation.set_text(f"X: {sel.target[0]:.2f}\nY: {sel.target[1]:.2f}")

def replot():

    fig, ax = plt.subplots()
    row_num_val = entry_vertical.get()
    col_num_val = entry_horizontal.get()
    plot(ax)
    plt.draw()
    # First tab + figure
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=f"V: {row_num_val}, H: {col_num_val}")
    canvas = FigureCanvasTkAgg(fig, master=tab)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# ---------------- TKINTER UI ------------------
ui = tk.Tk()
ui.title("Configure Plots")

# Create a Notebook widget
notebook = ttk.Notebook(ui)
notebook.pack(expand=True, fill="both")

# 1) Start Time
label_start = tk.Label(ui, text="Start Time:")
label_start.pack()
entry_start = tk.Entry(ui)
entry_start.insert(0, "100")  # default value
entry_start.pack()

# 2) End Time
label_end = tk.Label(ui, text="End Time:")
label_end.pack()
entry_end = tk.Entry(ui)
entry_end.insert(0, "1000")  # default value
entry_end.pack()

# 3) Vertical Position
label_vertical = tk.Label(ui, text="Vertical Position:")
label_vertical.pack()
entry_vertical = tk.Entry(ui)
entry_vertical.insert(0,"0")
entry_vertical.pack()

# 4) Horizontal Position
label_horizontal = tk.Label(ui, text="Horizontal Position:")
label_horizontal.pack()
entry_horizontal = tk.Entry(ui)
entry_horizontal.insert(0,"0")
entry_horizontal.pack()

# Replot Button
button = tk.Button(ui, text="Replot", command=replot)
button.pack()





ui.mainloop()
