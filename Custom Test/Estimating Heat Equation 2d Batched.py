import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch import nn
from timeit import default_timer
from torch.utils.data import TensorDataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

# Set default types and seeds
seeds_num = 666
torch.manual_seed(seeds_num)
np.random.seed(seeds_num)

# === Constants ===
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
Q = 2.192
INITIAL_TEMP = 21.23
THERMAL_CONDUCTIVITY = 10
NUM_DIV = 100
DURATION = 1
LENGTH = 1
TIMESTEP = 0.01
steps = 100000

def load_data(filepath="temperature_output.csv"):
    return pd.read_csv(filepath)

# === Prepare Training Data ===
def prepare_training_data(df):
    xValues = set()
    yValues = set()
    tValues = set()
    all_columns = df.columns.tolist()

    all_coords = []
    all_temps = []
    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(all_columns)):
            column = all_columns[i]
            x, y = map(float, column.strip("()").split(","))
            temp = df.iloc[idx, i]

            
            xValues.add(x)
            yValues.add(y)
            tValues.add(t_raw)
            all_coords.append([x, y, t_raw])
            all_temps.append([temp])
    return (
        np.array(all_coords),np.array(all_temps)
    )






k=np.random.rand()

class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, hidden_dim=50, num_hidden=3, activation='sin'):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.epoch = 0
        self.k = nn.Parameter(torch.tensor([k], requires_grad=True).float())

        if activation == 'sin':
            self.activation = torch.sin
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'cos':
            self.activation = torch.cos
        elif activation == 'silu':
            self.activation = torch.nn.functional.silu
            
    def forward(self, x,y, t):
        out = torch.cat([x,y,t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, y, t):
        x = x.requires_grad_()
        y = y.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x,y, t)

        # Scale factors
        dx_factor = 1.0 / xScale
        dy_factor = 1.0 / yScale
        dt_factor = 1.0 / tScale

        # Derivatives w.r.t. normalized coordinates
        du_dx_norm = torch.autograd.grad(u_pred_norm, x, grad_outputs=torch.ones_like(u_pred_norm), retain_graph=True, create_graph=True)[0]
        d2u_dx2_norm = torch.autograd.grad(du_dx_norm, x, grad_outputs=torch.ones_like(du_dx_norm), create_graph=True)[0]
        du_dy_norm = torch.autograd.grad(u_pred_norm, y, grad_outputs=torch.ones_like(u_pred_norm), retain_graph=True, create_graph=True)[0]
        d2u_dy2_norm = torch.autograd.grad(du_dy_norm, y, grad_outputs=torch.ones_like(du_dy_norm), create_graph=True)[0]
        du_dt_norm = torch.autograd.grad(u_pred_norm, t, grad_outputs=torch.ones_like(u_pred_norm), create_graph=True)[0]

        # Convert to real derivatives
        u_t = du_dt_norm * dt_factor
        u_xx = d2u_dx2_norm * (dx_factor ** 2)
        u_yy = d2u_dy2_norm * (dy_factor ** 2)

        # Scale temperature derivative back to physical units if temp is normalized
        u_t = u_t * (tempScale / 2)
        u_xx = u_xx * (tempScale / 2)
        u_yy = u_yy * (tempScale / 2)

        # PDE residual
        residual = (self.k * u_t) - (THERMAL_CONDUCTIVITY * u_xx) - (THERMAL_CONDUCTIVITY * u_yy) - Q
        loss_r = torch.mean(residual ** 2)
        return loss_r

    
    def loss_initial(self, x,y, t):    
        x = x.requires_grad_()
        y = y.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x,y, t)

        # Convert prediction from normalized to physical temperature
        u_pred_physical = 0.5 * (u_pred_norm + 1) * tempScale + minTemp

        loss_i = torch.mean((u_pred_physical - INITIAL_TEMP) ** 2)
        return loss_i

        
    def loss_bounds(self, x, y, t):    
        x = x.requires_grad_()
        y = y.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x,y, t)

        # Compute du/dx_norm
        dudx_norm = torch.autograd.grad(
            outputs=u_pred_norm,
            inputs=x,
            grad_outputs=torch.ones_like(u_pred_norm),
            create_graph=True,
            retain_graph=True
        )[0]

        # Convert to physical space: dT/dx = (2 / x_scale) * dT/dx_norm
        dudx_physical = dudx_norm * (1.0 / xScale)

        # Enforce Neumann BC: dT/dx = 0
        loss_b_x = torch.mean(dudx_physical ** 2)

        dudy_norm = torch.autograd.grad(
            outputs=u_pred_norm,
            inputs=y,
            grad_outputs=torch.ones_like(u_pred_norm),
            create_graph=True,
            retain_graph=True
        )[0]

        # Convert to physical space: dT/dx = (2 / x_scale) * dT/dx_norm
        dudy_physical = dudy_norm * (1.0 / yScale)

        # Enforce Neumann BC: dT/dx = 0
        loss_b_y = torch.mean(dudy_physical ** 2)
        return loss_b_x+loss_b_y

    
    def loss_data(self, x,y, t, u):    
        u_pred = self.forward(x,y, t) 
        loss_d = torch.mean((u_pred - u) ** 2)
        return loss_d
    
    def losses(self):
        loss_r = self.loss_PDE(x_train_Nu, y_train_Nu, t_train_Nu)
        loss_i = self.loss_initial(x_train_initial, y_train_initial, t_train_initial)
        loss_b = self.loss_bounds(x_train_boundary, y_train_boundary, t_train_boundary)
        loss_d = self.loss_data(x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu) 
        return loss_r, loss_i, loss_b, loss_d
    
# Load raw data
df = load_data() 
coords_array, tempValues = prepare_training_data(df)

# Compute min/max for normalization
minX, maxX = coords_array[:, 0].min(), coords_array[:, 0].max()
minY, maxY = coords_array[:, 1].min(), coords_array[:, 1].max()
minT, maxT = coords_array[:, 2].min(), coords_array[:, 2].max()
minTemp, maxTemp = tempValues.min(), tempValues.max()

# Avoid division by zero
xScale = maxX - minX if maxX != minX else 1.0
yScale = maxY - minY if maxY != minY else 1.0
tScale = maxT - minT if maxT != minT else 1.0
tempScale = maxTemp - minTemp if maxTemp != minTemp else 1.0

# --- Normalize ---
coords_array[:, 0] = (coords_array[:, 0] - minX) / xScale   # x ∈ [0, 1]
coords_array[:, 1] = (coords_array[:, 1] - minY) / yScale   # x ∈ [0, 1]
coords_array[:, 2] = (coords_array[:, 2] - minT) / tScale   # t ∈ [0, 1]
tempValues = 2 * (tempValues - minTemp) / tempScale - 1            # temp ∈ [-1, 1]




# --- All training data ---
# --- Training data (for model training) ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
y_train_Nu = X_train_Nu_tensor[:, 1:2]
t_train_Nu = X_train_Nu_tensor[:, 2:3]



# --- Boundary points from training set: x = 0 or x = 1 ---
boundary_mask = np.isclose(coords_array[:, 0], 0) | np.isclose(coords_array[:, 0], 1.0) | np.isclose(coords_array[:, 1], 0) | np.isclose(coords_array[:, 1], 1)
X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
y_train_boundary = X_train_boundary_tensor[:, 1:2]
t_train_boundary = X_train_boundary_tensor[:, 2:3]

# --- Initial points from training set: t = 0 ---
initial_mask = np.isclose(coords_array[:, 2], 0)
X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
y_train_initial = X_train_initial_tensor[:, 1:2]
t_train_initial = X_train_initial_tensor[:, 2:3]
    
def closure():
    global lambda1, lambda2, lambda3, lambda4

    optimizer.zero_grad()
    loss_r, loss_i, loss_b, loss_d = model.losses()
    loss = lambda1 * loss_r + lambda2 * loss_i + lambda3 * loss_b + lambda4 * loss_d
    loss.backward()

    # Logging
    epoch_loss_r.append(loss_r.item())
    epoch_loss_b.append(loss_b.item())
    epoch_loss_i.append(loss_i.item())
    epoch_loss_d.append(loss_d.item())
    epoch_beta.append(model.k.item())
    epoch_lambda1.append(lambda1)
    epoch_lambda2.append(lambda2)
    epoch_lambda3.append(lambda3)
    epoch_lambda4.append(lambda4)


    print(f'Epoch {model.epoch}, loss = {loss.item():.4e},  k = {model.k.item():.6f}')

    model.epoch += 1
    return loss

train_dataset = TensorDataset(x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu)
batch_size = 1250  # You can adjust this
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def train_dg_pinn(model, optimizer, iters, train_loader,
                  early_stop_patience=500,
                  min_delta=0,
                  epoch_loss_d=[]):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(iters):
        model.train()
        t1 = default_timer()
        epoch_loss = 0.0

        for batch in train_loader:
            x_batch, y_batch, t_batch, u_batch = [b.to(device) for b in batch]

            optimizer.zero_grad()
            loss = model.loss_data(x_batch, y_batch, t_batch, u_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        epoch_loss_d.append(epoch_loss)

        t2 = default_timer()
        print(f"Epoch {epoch}, time = {t2 - t1:.2e}s, loss = {epoch_loss:.4e}, k = {model.k.item():.6f}")

        # Early stopping
        if epoch_loss < best_val_loss - min_delta:
            best_val_loss = epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"[Adam] Early stopping triggered at epoch {epoch}")
            break

    return epoch_loss_d, best_model_state



# === Training ===
torch.manual_seed(seeds_num)
epoch_loss_r = []
epoch_loss_i = []
epoch_loss_b = []
epoch_loss_d = []
epoch_beta = []
epoch_lambda1 = []
epoch_lambda2 = []
epoch_lambda3 = []
epoch_lambda4 = []

model = PINN(
    input_dim=3,
    output_dim=1,
    hidden_dim=100,
    num_hidden=3, 
    activation='tanh'
).to(device)
print(model)

iter_1s = 100000
iter_2 = 100000  # Maximun number  of iterations for L-BFGS optimizer
t11 = default_timer()
# Adam optimizer to decrease loss in Phase 1
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
epoch_loss_d, best_model_state = train_dg_pinn(
    model, optimizer, iters=iter_1s,
    train_loader=train_loader,
    epoch_loss_d=[]
)

import copy

# Define lambda weights if not already done
lambda1, lambda2, lambda3, lambda4 = 1, 1, 1, 1

# Reset optimizer to L-BFGS (after Adam)
optimizer = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=1, history_size=100, line_search_fn="strong_wolfe"
)

# Setup early stopping
best_loss_total = float('inf')
patience_counter = 0
early_stop_patience = 200
min_delta = 0  # Small improvement threshold

# L-BFGS loop
for epoch in range(iter_2):
    optimizer.zero_grad()
    loss = optimizer.step(closure)


    if loss < best_loss_total - min_delta:
        best_loss_total = loss
        patience_counter = 0
        best_model_state = model.state_dict()  # Save best model
    else:
        patience_counter += 1

    if patience_counter >= early_stop_patience:
        print(f"[L-BFGS] Early stopping triggered at epoch {epoch}")
        break




t22 = default_timer()
print('Time elapsed: %.2f min' % ((t22 - t11) / 60))

print(model.k.item())
