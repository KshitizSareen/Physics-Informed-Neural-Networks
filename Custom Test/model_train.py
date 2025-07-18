import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tempfile
import pkgutil
import time
import sys, os

output_path = "model.pth"

# === Constants ===
POWER_APPLIED = 0.15
DIAMETER = 0.3
RADIUS = DIAMETER / 2
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
TIME_STEP = 0.1
E_GEN = (POWER_APPLIED / (np.pi * RADIUS**2))*TIME_STEP

TEMP_X = ["0", "1", "2.5", "5", "10", "15"]
TEMP_Y = ["0", "1", "2.5", "5", "15"]
X_POSITIONS = [-15, -10, -5, -2.5, -1, 0, 1, 2.5, 5, 10, 15]
Y_POSITIONS = [0, 1, 2.5, 5, 15]

# === Data Loading ===
def load_data(filepath="temperature_output.csv"):
    df = pd.read_csv(filepath)
    all_columns = df.columns.tolist()
    minTemp = float("inf")
    maxTemp = float("-inf")
    for i in range(2,len(all_columns)):
        column_name = all_columns[i]
        temp_column = df[column_name]
        if temp_column.min()<minTemp:
            minTemp = temp_column.min()
        if temp_column.max()>maxTemp:
            maxTemp = temp_column.max()
            
    return df,minTemp,maxTemp

# === Prepare Training Data ===
def prepare_training_data(df,minTemp,maxTemp):
    training_data = []
    max_time = int(df["Timestamp"].max())
    min_time = int(df["Timestamp"].min())
    minX = 0
    maxX = 1
    minY = 0
    maxY = 1
    all_columns = df.columns.tolist()
    t=1
    for idx in range(1,len(df)):
        for i in range(2,len(all_columns)):
            column = all_columns[i]
            x, y = map(float, column.strip("()").split(","))
            if x<=maxX and y<=maxY: 
                t = df["Timestamp"][idx]
                temp = df[column][idx]
                training_data.append([(x-minX)/(maxX-minX), (y-minY)/(maxY-minY), (t-min_time)/(max_time-min_time), (temp-minTemp)/(maxTemp-minTemp)])
    return np.array(training_data),minX,maxX,minY,maxY,min_time,max_time

# === Neural Network Model ===
def get_model():
    return nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )


def get_k(k):
    return torch.nn.functional.softplus(k)

def train_model(n_epochs=50000, lr=0.001, val_split=0.2, patience=5000,batch_size = 32,lambda_phys=1e-5):

# Ensure directory exists
    os.makedirs("plots", exist_ok=True)
    torch.manual_seed(42)
    df,minTemp,maxTemp = load_data()
    print(df,len(df.columns.tolist()))
    data,minX,maxX,minY,maxY,min_time,max_time = prepare_training_data(df,minTemp,maxTemp)
    print(data,len(data))

    X_data = data[:, 0:3]
    y_data = data[:, 3]

    # Train-validation split
    X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_data, y_data, test_size=val_split, random_state=42)

    # Convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).view(-1, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_val, y_val = X_train.to(device), y_train.to(device), X_val.to(device), y_val.to(device)

    N_phys = 20 # Using a smaller number of points for grid to manage memory
    x_p = torch.linspace(0, 1, N_phys, device=device)
    y_p = torch.linspace(0, 1, N_phys, device=device)
    t_p = torch.linspace(0, 1, N_phys, device=device)
    
    # torch.meshgrid creates all combinations of the input points
    X_p, Y_p, T_p = torch.meshgrid(x_p, y_p, t_p, indexing='ij')

    # Flatten and combine into a single tensor of [x, y, t] coordinates
    x_phys = X_p.flatten().view(-1, 1).requires_grad_(True)
    y_phys = Y_p.flatten().view(-1, 1).requires_grad_(True)
    t_phys = T_p.flatten().view(-1, 1).requires_grad_(True)

    input_phys = torch.cat([x_phys, y_phys, t_phys], dim=1)
    

    model = get_model().to(device)
    k = torch.nn.Parameter(torch.tensor([0.015], dtype=torch.float32, requires_grad=True).to(device))
    optimizer = optim.Adam(list(model.parameters()) + [k], lr=lr)


    # Early stopping trackers
    best_val_loss = float("inf")
    best_model_state = True
    best_k_val = None

    loss_fn = nn.MSELoss()
    best_phys_loss = float("inf")

    epoch = 0

    trace_ks=[]

    for epoch in range(n_epochs):
        avg_loss_physics = 0
        count=0
        permutation = torch.randperm(X_train.size()[0])
        model.train()
        for i in range(0, len(X_train), batch_size):
            T_phys = model(input_phys)

            dtdx = torch.autograd.grad(T_phys, x_phys, torch.ones_like(T_phys), create_graph=True)[0]
            d2tdx2 = torch.autograd.grad(dtdx, x_phys, torch.ones_like(dtdx), create_graph=True)[0]
            dtdy = torch.autograd.grad(T_phys, y_phys, torch.ones_like(T_phys), create_graph=True)[0]
            d2tdy2 = torch.autograd.grad(dtdy, y_phys, torch.ones_like(dtdy), create_graph=True)[0]
            dtdt = torch.autograd.grad(T_phys, t_phys, torch.ones_like(T_phys), create_graph=True)[0]

            residual = d2tdx2 + d2tdy2 - ((DENSITY * SPECIFIC_HEAT_CAPACITY) / k) * dtdt + (E_GEN / k)
            loss_physics = torch.mean(residual**2)
            avg_loss_physics+=loss_physics.item()
            count+=1
            indices = permutation[i:i+batch_size]
            Xbatch = X_train[indices]
            ybatch = y_train[indices]

            # Forward + Backward
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch) + (lambda_phys* loss_physics)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trace_ks.append(k.item())

        print(epoch)

        if(epoch%100==0 and epoch>0):
            tempArrayPredicted = []
            tempArrayActual = []
            tpoints=[]
            t=1
            for idx in range(1,len(df)):
                t = df["Timestamp"][idx]
                input_tensor = torch.tensor([[(0-minX)/(maxX-minX), (0-minY)/(maxY-minY), (t-min_time)/(max_time-min_time)]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    prediction = model(input_tensor).item()
                    prediction = prediction*(maxTemp-minTemp)+minTemp
                    tempArrayPredicted.append(prediction)
                    temp = df["(0.0,0.0)"][idx]
                    tempArrayActual.append(temp)
                tpoints.append(t)

            # Save predicted vs. actual plot
            plt.figure(figsize=(6, 2.5))
            plt.scatter(tpoints, tempArrayActual, label="observations", alpha=0.6)
            plt.plot(tpoints, tempArrayPredicted, label="predicted", color="tab:green")
            plt.title(f"Training step {epoch}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/predicted_vs_actual_step_{epoch}.png")
            plt.close()

            # Save μ convergence plot
            plt.figure(figsize=(8, 4))
            plt.plot(trace_ks, label='Learned μ over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('μ')
            plt.title('Convergence of Learned μ')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/mu_convergence_{epoch}.png")
            plt.close()


    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        k = best_k_val

    model.eval()
    with torch.no_grad():
        final_val = torch.mean((model(X_val) - y_val) ** 2).item()
        print(f"✅ Final Validation MSE: {final_val:.6f}, Final Physics Loss: {best_phys_loss}, Final val_loss {best_val_loss}")

    torch.save(model.state_dict(), 'model.pth')
    return model, k


# === Load Model ===
def load_model():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)

    if hasattr(sys, '_MEIPASS'):
        # Extract model.pth from package using PyInstaller-safe method
        data = pkgutil.get_data(__name__, "model.pth")
        temp_path = os.path.join(tempfile.gettempdir(), "model.pth")

        with open(temp_path, "wb") as f:
            f.write(data)
    else:
        # Running from source (not PyInstaller)
        temp_path = "model.pth"

    model.load_state_dict(torch.load(temp_path, map_location=device))
    model.eval()
    return model

# === Test Model ===
def test_model(x, y, t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    input_tensor = torch.tensor([[x, y, t]], dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(input_tensor).item()

# === Main ===
if __name__ == "__main__":
    model, k = train_model()
    print(f"Estimated Thermal Conductivity (k): {k.item():.6f}")
