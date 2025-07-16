import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# ------------------------------------------------------------------
# Your data loading and definitions for training
TempXColumns = ["0", "1","2.5","5","10","15"]
TempYColumns = ["0", "1","2.5","5","15"]

data = pd.read_csv(
    "data.csv", sep=',',
    names=["Timestamp", "Power"] + [f"{x},{y}" for y in TempYColumns for x in TempXColumns]
)

# Duplicate negative x columns
for x in TempXColumns:
    negativeX = "-"+x
    for y in TempYColumns:
        data[f"{negativeX},{y}"] = data[f"{x},{y}"]

XData = [-15,-10,-5,-2.5,-1,0,1,2.5,5,10,15]  # shape = (11,)
YData = [0,1,2.5,5,15]                     # shape = (5, not 6 as the code suggests, 
                                                      # just from your snippet. Adjust if needed.)
TData = data["Timestamp"]

def prepare_training_data():
    training_data = []
    maxTimestamp = np.max(data["Timestamp"])
    for i in range(1, maxTimestamp+1):
        # Adjust how often you collect training data (example: every 10 timestamps)
        if i == 1 or i % 10 == 0:
            idx = data.index[data['Timestamp'] == i][0]
            for y in YData:
                for x in XData:
                    training_data.append([x, y, i,data[f"{x},{y}"][idx]])
    return np.array(training_data)

def train_model(n_epochs=50, batch_size=32, lr=0.001):
    """
    Train the PyTorch model and save to 'model.pth'.
    Returns the trained model if needed in the same script.
    """
    training_data = prepare_training_data()


    # Separate features and target
    # Here, the first 3 columns: x, y, (?), but in your original code you had X[:,0:3] as inputs
    # and X[:,3] as Y. Adjust accordingly if you need a different indexing
    X = training_data[:, 0:3]  # x, y, and *some column*
    Y = training_data[:, 3]    # the 4th column as the target

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    # Define the model
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)

    # Loss, optimizer, and scheduler
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

    model.train()
    for epoch in range(n_epochs):
        permutation = torch.randperm(X.size()[0])
        for i in range(0, len(X), batch_size):
            indices = permutation[i:i+batch_size]
            Xbatch = X[indices]
            ybatch = y[indices]

            # Forward + Backward
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Scheduler step
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: SGD lr {before_lr:.4f} -> {after_lr:.4f}")

        # Check overall MSE
        y_pred_full = model(X)
        final_mse = loss_fn(y_pred_full, y).item()
        print(f"MSE on training data: {final_mse:.6f}")
        if final_mse <= 0.05:
            break

    model.eval()
    with torch.no_grad():
        y_pred_full = model(X)

    final_mse = loss_fn(y_pred_full, y).item()
    print(f"Final MSE on training data: {final_mse:.6f}")

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    return model

def load_model():
    """
    Load the trained model from 'model.pth'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()
    return model

def test_model(x, y, t):
    """
    Pass a single input (x, y, t) to the loaded model and get the prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_input = torch.tensor(np.array([x, y, t]), dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = load_model()(new_input)
    return prediction

if __name__ == "__main__":
    # Train model when you run this file directly
    trained_model = train_model()
