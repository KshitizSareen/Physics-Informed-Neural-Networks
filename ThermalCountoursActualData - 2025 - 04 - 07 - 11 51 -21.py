import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
from matplotlib import colors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# ------------------------------------------------------------------
# Your data loading and definitions (unmodified)
TempXColumns = ["0", "1","2.5","5","10","15"]
TempYColumns = ["0", "1","2.5","5","15"]

data = pd.read_csv(
    "temperature readings 2025-04-07_11-51-21 - Dataset.csv", sep=',',
    names=["Timestamp", "Power"] + [f"{x},{y}" for y in TempYColumns for x in TempXColumns]
)


for x in TempXColumns:
    negativeX = "-"+x
    for y in TempYColumns:
        data[f"{negativeX},{y}"] = data[f"{x},{y}"]


XData = np.array([-15,-10,-5,-2.5,-1,0,1,2.5,5,10,15])  # shape = (11,)
YData = np.array([0,1,2.5,5,15])                   # shape = (6,)
TData = data["Timestamp"]

def prepare_training_data():
    training_data = []
    maxTimestamp = np.max(data["Timestamp"])
    for i in range(1,maxTimestamp+1):
        if i==1 or i%10==0:
            idx = data.index[data['Timestamp']==i][0]
            for y in YData:
                for x in XData:
                    training_data.append([x,y,data[f"{x},{y}"][idx],i])
    
    return np.array(training_data)



blue_red_cmap = colors.LinearSegmentedColormap.from_list('BlueRed', ['blue', 'red'])
norm = colors.Normalize(vmin=20, vmax=50)

def fetchData(t):
    """Return 11x6 array of (TempX + TempY)/2 for the given timestamp t."""
    # (Note: XData has 11 entries, YData has 6)
    idx = data.index[data['Timestamp']==t][0]
    TempData = []
    for y in YData:
        tempValues = []
        for x in XData:
            new_input = torch.tensor(np.array([x,y,t]), dtype=torch.float32)
            with torch.no_grad():
                prediction = model(new_input)
                tempValues.append(prediction)
        TempData.append(tempValues)
    return TempData  # shape will be (6, 11) if stacked row-wise

# ------------------------------------------------------------------
# Main plotting code
fig = plt.figure()
ax = fig.add_subplot(111)
plt.subplots_adjust(bottom=0.25)

maxTimestamp = np.max(data["Timestamp"])

# Create the meshgrid of X, Y (note: default np.meshgrid -> X.shape=(6,11), Y.shape=(6,11))
X, Y = np.meshgrid(XData, YData)  # X and Y will match the shape we produce from fetchData

# Get initial data and plot both contourf and contour for labeling
initial_Z = fetchData(1)  # shape (6, 11)
CSF = ax.contourf(X, Y, initial_Z, cmap=blue_red_cmap, norm=norm)
CS = ax.contour(X, Y, initial_Z, colors='black', linewidths=0.8)
ax.clabel(CS, inline=True, fontsize=8)

# Create a global (or module-level) array to store the current Z data
# so that the mplcursors callback can reference it.
current_Z = initial_Z.copy()

# Prepare slider steps (0 to maxTimestamp in steps of 10)
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

# Create axes for the colorbar
cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
cbar = plt.colorbar(CSF, cax=cax)
cbar.set_label('Temperature')

def update(val):
    global current_Z
    # Clear the colorbar axis (so a new colorbar can be drawn)
    cax.clear()
    timesliderVal = time_slider.val

    # Clear main axes and draw new contour
    ax.cla()
    Z = fetchData(timesliderVal)
    current_Z = Z  # Store new Z data for the mplcursors callback
    
    # Redraw the filled contour
    new_contour_f = ax.contourf(X, Y, Z, cmap=blue_red_cmap, norm=norm)
    
    # Draw the line contour on top for labels
    new_contour = ax.contour(X, Y, Z, colors='black', linewidths=0.8)
    ax.clabel(new_contour, inline=True, fontsize=8)
    
    # Update colorbar
    cb = fig.colorbar(new_contour_f, cax=cax)
    cb.set_label('Temperature')
    
    plt.draw()

time_slider.on_changed(update)

training_data = prepare_training_data()

def train_model(n_epochs=50, batch_size=32, lr=0.001):
    # Separate features and target
    X = training_data[:, 0:3]
    Y = training_data[:, 3]

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    # Use GPU if available for efficiency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    # Define a more efficient model with fewer layers
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),  # linear output for regression
    ).to(device)

    # Define loss function (MSE) and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    model.train()

    # Train the model
    for epoch in range(10000):
        # Shuffle data indices for each epoch (optional but often beneficial)
        permutation = torch.randperm(X.size()[0])

        for i in range(0, len(X), batch_size):
            indices = permutation[i:i+batch_size]
            Xbatch = X[indices]
            ybatch = y[indices]

            # Forward pass
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        y_pred_full = model(X)
        # Calculate final MSE
        final_mse = loss_fn(y_pred_full, y).item()
        print(f"MSE on training data: {final_mse:.6f}")
        if final_mse<=0.05:
            break

    model.eval()
    with torch.no_grad():
        y_pred_full = model(X)

    # Calculate final MSE
    final_mse = loss_fn(y_pred_full, y).item()
    print(f"Final MSE on training data: {final_mse:.6f}")

    torch.save(model.state_dict(), 'model.pth')


    return model

train_model()

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define a more efficient model with fewer layers
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),  # linear output for regression
    ).to(device)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

model = load_model()

def test_model(x,y,t):
    new_input = torch.tensor(np.array([x,y,t]), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(new_input)
    return prediction

# Override format_coord
def format_coord(x, y):
    z = test_model(x,y,time_slider.val)[0]
    return f"X={x:.2f}, Y={y:.2f}, Z={z:.2f}"

ax.format_coord = format_coord
# ----------------------------------------------------------------------------

plt.show()



