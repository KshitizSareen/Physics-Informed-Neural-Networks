import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from torch import nn
from timeit import default_timer
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
Q = 1.94
INITIAL_TEMP = 21.23
THERMAL_CONDUCTIVITY = 10
NUM_DIV = 100
DURATION = 1
LENGTH = 1
TIMESTEP = 0.01
steps = 100000

def load_data(filepath="temperature_output_1d.csv"):
    return pd.read_csv(filepath)

# === Prepare Training Data ===
def prepare_training_data(df):
    xValues = set()
    tValues = set()
    tempValues = []
    all_coords = []
    columns = df.columns.tolist()

    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(columns)):
            x = float(columns[i])
            temp = df.iloc[idx, i]
            xValues.add(x)
            tValues.add(t_raw)
            tempValues.append([temp])
            all_coords.append([x, t_raw])
    
    return sorted(xValues), sorted(tValues), np.array(all_coords), np.array(tempValues)

# Load raw data
df = load_data()
xValues, tValues, coords_array, tempValues = prepare_training_data(df)

# Compute min/max for normalization
minX, maxX = coords_array[:, 0].min(), coords_array[:, 0].max()
minT, maxT = coords_array[:, 1].min(), coords_array[:, 1].max()
minTemp, maxTemp = tempValues.min(), tempValues.max()

# Avoid division by zero
xScale = maxX - minX if maxX != minX else 1.0
tScale = maxT - minT if maxT != minT else 1.0
tempScale = maxTemp - minTemp if maxTemp != minTemp else 1.0

# --- Normalize ---
coords_array[:, 0] = (coords_array[:, 0] - minX) / xScale   # x ∈ [-1, 1]
coords_array[:, 1] = (coords_array[:, 1] - minT) / tScale   # t ∈ [-1, 1]
tempValues = 2 * (tempValues - minTemp) / tempScale - 1            # temp ∈ [-1, 1]




# --- All training data ---
# --- Training data (for model training) ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
t_train_Nu = X_train_Nu_tensor[:, 1:2]



# --- Boundary points from training set: x = -1 or x = 1 ---
boundary_mask = np.isclose(coords_array[:, 0], 0) | np.isclose(coords_array[:, 0], 1.0)
X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
t_train_boundary = X_train_boundary_tensor[:, 1:2]

# --- Initial points from training set: t = -1 ---
initial_mask = np.isclose(coords_array[:, 1], 0)
X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
t_train_initial = X_train_initial_tensor[:, 1:2]




k=12.0

class PINN(nn.Module):
    def __init__(self, input_dim=3, output_dim=2, hidden_dim=50, num_hidden=3, activation='sin'):
        super(PINN, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.epoch = 0
        self.theta = nn.Parameter(torch.tensor([12.0]).float())

        if activation == 'sin':
            self.activation = torch.sin
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'cos':
            self.activation = torch.cos
        elif activation == 'silu':
            self.activation = torch.nn.functional.silu

    @property
    def k(self):
        return torch.exp(self.theta)       
            
    def forward(self, x, t):
        out = torch.cat([x, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, t):
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x, t)

        # Scale factors
        dx_factor = 1.0 / xScale
        dt_factor = 1.0 / tScale

        # Derivatives w.r.t. normalized coordinates
        du_dx_norm = torch.autograd.grad(u_pred_norm, x, grad_outputs=torch.ones_like(u_pred_norm), retain_graph=True, create_graph=True)[0]
        d2u_dx2_norm = torch.autograd.grad(du_dx_norm, x, grad_outputs=torch.ones_like(du_dx_norm), create_graph=True)[0]
        du_dt_norm = torch.autograd.grad(u_pred_norm, t, grad_outputs=torch.ones_like(u_pred_norm), create_graph=True)[0]

        # Convert to real derivatives
        u_t = du_dt_norm * dt_factor
        u_xx = d2u_dx2_norm * (dx_factor ** 2)

        # Scale temperature derivative back to physical units if temp is normalized
        u_t = u_t * (tempScale / 2)
        u_xx = u_xx * (tempScale / 2)

        # PDE residual
        residual = (DENSITY * SPECIFIC_HEAT_CAPACITY * u_t) - ((self.theta) * u_xx) - (Q)
        loss_r = torch.mean(residual ** 2)
        return loss_r

    
    def loss_initial(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x, t)

        # Convert prediction from normalized to physical temperature
        u_pred_physical = 0.5 * (u_pred_norm + 1) * tempScale + minTemp

        loss_i = torch.mean((u_pred_physical - INITIAL_TEMP) ** 2)
        return loss_i

        
    def loss_bounds(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred_norm = self.forward(x, t)

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
        loss_b = torch.mean(dudx_physical ** 2)
        return loss_b

    
    def loss_data(self, x, t, u):    
        u_pred = self.forward(x, t) 
        loss_d = torch.mean((u_pred - u) ** 2)
        return loss_d
    
    def losses(self):
        loss_r = self.loss_PDE(x_train_Nu, t_train_Nu)
        loss_i = self.loss_initial(x_train_initial, t_train_initial)
        loss_b = self.loss_bounds(x_train_boundary, t_train_boundary)
        loss_d = self.loss_data(x_train_Nu, t_train_Nu, U_train_Nu) 
        return loss_r, loss_i, loss_b, loss_d
    
def closure():

    optimizer_2.zero_grad()
    loss_r, loss_i, loss_b, loss_d = model.losses()
    loss = loss_r + loss_i + loss_b
    loss.backward()

    epoch_loss_r.append(loss_r.item())
    epoch_loss_b.append(loss_b.item())
    epoch_loss_i.append(loss_i.item())
    epoch_loss_d.append(loss_d.item())
    epoch_beta.append(model.k.item())

    print(f'Epoch {model.epoch}, loss = {loss.item():.4e},  k = {model.theta.item():.6f}')
    print(f"Gradient of theta: {model.theta.grad}")
    model.epoch += 1
    return loss



def train_dg_pinn(model, optimizer, iters,
                  x_train, t_train, u_train,
                  early_stop_patience=500,
                  min_delta=0,
                  epoch_loss_d=[]):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(iters):
        t1 = default_timer()

        optimizer.zero_grad()
        train_loss = model.loss_data(x_train, t_train, u_train)
        train_loss.backward()
        optimizer.step()

        epoch_loss_d.append(train_loss.item())

        t2 = default_timer()
        print('Epoch %d, time = %e, train_loss = %e, k = %e' %
              (epoch, float(t2 - t1), float(train_loss),  model.k.item()))

        # Early stopping
        if train_loss.item() < best_val_loss - min_delta:
            best_val_loss = train_loss.item()
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
    input_dim=2,
    output_dim=1,
    hidden_dim=100,
    num_hidden=3, 
    activation='tanh'
).to(device)
print(model)

iter_1s = 100000
iter_2 = 100000  # Maximun number  of iterations for L-BFGS optimizer
t11 = default_timer()
# === Phase 1: Pretraining on data ===
optimizer_1 = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_loss_d, best_model_state = train_dg_pinn(
    model, optimizer_1, iters=iter_1s,
    x_train=x_train_Nu, t_train=t_train_Nu, u_train=U_train_Nu,
    epoch_loss_d=[]
)

# Restore best weights from data-fitting phase
model.load_state_dict(best_model_state)

# === Phase 2: Train full PINN loss ===
optimizer_2 = torch.optim.Adam(model.parameters(), lr=0.01)  # Use smaller LR for stability

# Reset tracking variables if needed
best_loss_total = float('inf')
patience_counter = 0

for epoch in range(iter_2):
    optimizer_2.zero_grad()
    loss = closure()  # Includes backward()
    optimizer_2.step()

    if loss.item() < best_loss_total:
        best_loss_total = loss.item()
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1

    if patience_counter >= 1000:
        print(f"[Adam-2] Early stopping triggered at epoch {epoch}")
        break

# Save the final best model
model.load_state_dict(best_model_state)




t22 = default_timer()
print('Time elapsed: %.2f min' % ((t22 - t11) / 60))

print(model.k.item())
