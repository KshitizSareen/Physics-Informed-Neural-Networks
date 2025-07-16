import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch

# Import the necessary functions (load_model, test_model, etc.)
from model_train import load_model, data

XData = range(-15,16,1)
YData = range(0,16,1)
model = load_model()  # Load the already-trained model


def fetchData(t,YData,XData):
    """
    Return 2D array of predictions (shape matches Y vs X) for the given timestamp t.
    XData has 11 entries, YData has 5 entries (in your code snippet).
    """
    TempData = []
    for y in YData:
        row_values = []
        for x in XData:
            new_input = torch.tensor(np.array([x, y, t]), dtype=torch.float32)
            with torch.no_grad():
                prediction = model(new_input).item()
                value = prediction
            row_values.append(value)
        TempData.append(row_values)
    return np.array(TempData)  # shape will be (len(YData), len(XData))


def update(val):
    ax.cla()
    timestamp_val = time_slider.val
    Z = fetchData(timestamp_val,YData,XData)

    new_contour_f = ax.contourf(X, Y, Z, cmap=blue_red_cmap, norm=norm)
    new_contour = ax.contour(X, Y, Z, colors='black', linewidths=0.8)
    ax.clabel(new_contour, inline=True, fontsize=8)
    ax.set_title(f"Timestamp = {timestamp_val}")

    # Update colorbar
    cax.cla()
    cb = fig.colorbar(new_contour_f, cax=cax)
    cb.set_label('Temperature')

    plt.draw()

def test_model_for_coord(x, y):
    """
    (Optional) If you want direct testing for a given (x, y) at the current slider value.
    """
    t_val = time_slider.val
    new_input = torch.tensor(np.array([x, y, t_val]), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(new_input).item()
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

    maxTimestamp = np.max(data["Timestamp"])

    # Create the meshgrid
    X, Y = np.meshgrid(XData, YData)  # shape: (len(YData), len(XData))

    # Initial data (timestamp=1)
    initial_Z = fetchData(1,YData,XData)
    CSF = ax.contourf(X, Y, initial_Z, cmap=blue_red_cmap, norm=norm)
    CS = ax.contour(X, Y, initial_Z, colors='black', linewidths=0.8)
    ax.clabel(CS, inline=True, fontsize=8)

    # Slider for time
    time = np.arange(0, maxTimestamp, 10)
    axTime = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axTime,
        label='Timestamp',
        valmin=1,
        valmax=maxTimestamp,
        valinit=1,
        valstep=time
    )

    # Colorbar
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
    cbar = plt.colorbar(CSF, cax=cax)
    cbar.set_label('Temperature')

    time_slider.on_changed(update)

    ax.format_coord = format_coord

    plt.show()
