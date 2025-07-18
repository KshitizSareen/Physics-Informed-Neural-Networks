from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch

# Import the necessary functions (load_model, test_model, etc.)
from model_train import load_model,X_POSITIONS,Y_POSITIONS,TEMP_Y,TEMP_X


blue_red_cmap = colors.LinearSegmentedColormap.from_list('BlueRed', ['blue', 'red'])
norm = colors.Normalize(vmin=0, vmax=100)

model = load_model()  # Load the already-trained model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def extract_min_max_values(filepath="data.csv"):
    column_names = ["Timestamp", "Power"] + [f"{x},{y}" for y in TEMP_Y for x in TEMP_X]
    df = pd.read_csv(filepath, sep=",", names=column_names)
    minTemp = float("inf")
    maxTemp = float("-inf")
    max_time = int(df["Timestamp"].max())
    min_time = int(df["Timestamp"].min())
    for x in TEMP_X:
        for y in TEMP_Y:
            
            temp = df[f"{x},{y}"]
            if temp.min()<minTemp:
                minTemp = temp.min()
            if temp.max()>maxTemp:
                maxTemp = temp.max()
    return minTemp,maxTemp,-15,15,0,15,min_time,max_time

minTemp,maxTemp,minX,maxX,minY,maxY,minTimestep,maxTimestep = extract_min_max_values()

def fetchData(t,minTemp,maxTemp,minX,maxX,minY,maxY,minTimestep,maxTimestep):
    """
    Return 2D array of predictions (shape matches Y vs X) for the given timestamp t.
    XData has 11 entries, YData has 5 entries (in your code snippet).
    """
    TempData = []
    for y in Y_POSITIONS:
        row_values = []
        for x in X_POSITIONS:
            input_tensor = torch.tensor([[(x-minX)/(maxX-minX), (y-minY)/(maxY-minY), (t-minTimestep)/(maxTimestep-minTimestep)]], dtype=torch.float32).to(device)
            with torch.no_grad():
                value = model(input_tensor).item()
                print(value)
                row_values.append(int(value*(maxTemp-minTemp)+minTemp))
        TempData.append(row_values)
    Input = np.array(TempData)  # shape will be (len(YData), len(XData))
    return Input

def update(val):
    ax.cla()
    timestamp_val = time_slider.val
    Z = fetchData(timestamp_val,minTemp,maxTemp,minX,maxX,minY,maxY,minTimestep,maxTimestep)

    new_contour_f = ax.contourf(X, Y, Z, cmap=blue_red_cmap, norm=norm)
    new_contour = ax.contour(X, Y, Z, colors='black', linewidths=0.8)
    ax.clabel(new_contour, inline=True, fontsize=8)
    ax.set_title(f"Timestamp = {timestamp_val}")

    # Update colorbar
    cax.cla()
    cb = fig.colorbar(new_contour_f, cax=cax)
    cb.set_label('Temperature')
    cb.ax.yaxis.set_major_formatter(ScalarFormatter())
    cb.ax.yaxis.get_major_formatter().set_useOffset(False)
    cb.ax.yaxis.get_major_formatter().set_scientific(False)

    plt.draw()

def test_model_for_coord(x, y):
    """
    (Optional) If you want direct testing for a given (x, y) at the current slider value.
    """
    t_val = time_slider.val
    input_tensor = torch.tensor([[(x-minX)/(maxX-minX), (y-minY)/(maxY-minY), (t_val-minTimestep)/(maxTimestep-minTimestep)]], dtype=torch.float32).to(device)
    with torch.no_grad():
        value = model(input_tensor).item()
        prediction = int(value*(maxTemp-minTemp)+minTemp)
    return prediction

def format_coord(x, y):
    # Convert to nearest integer or keep float, up to you
    z_est = test_model_for_coord(x, y)
    return f"X={x:.2f}, Y={y:.2f}, Z={z_est:.2f}"

if __name__ == "__main__":

    blue_red_cmap = colors.LinearSegmentedColormap.from_list('BlueRed', ['blue', 'red'])
    norm = colors.Normalize(vmin=0, vmax=100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(bottom=0.25)


    # Create the meshgrid
    X, Y = np.meshgrid(X_POSITIONS,Y_POSITIONS)  # shape: (len(YData), len(XData))

    # Initial data (timestamp=1)
    initial_Z = fetchData(1,minTemp,maxTemp,minX,maxX,minY,maxY,minTimestep,maxTimestep)
    print(initial_Z)
    CSF = ax.contourf(X, Y, initial_Z, cmap=blue_red_cmap, norm=norm)
    CS = ax.contour(X, Y, initial_Z, colors='black', linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=8)

    # Slider for time
    time = np.arange(0, maxTimestep, 10)
    axTime = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axTime,
        label='Timestamp',
        valmin=1,
        valmax=maxTimestep,
        valinit=1,
        valstep=time
    )

    # Colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(CSF, cax=cax)
    cbar.set_label('Temperature')

    cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    cbar.ax.yaxis.get_major_formatter().set_useOffset(False)
    cbar.ax.yaxis.get_major_formatter().set_scientific(False)

    time_slider.on_changed(update)

    ax.format_coord = format_coord

    plt.show()
