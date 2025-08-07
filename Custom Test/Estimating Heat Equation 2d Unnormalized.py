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


# Load raw data
df = load_data() 
coords_array, tempValues = prepare_training_data(df)

# --- No normalization ---

# --- All training data ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
y_train_Nu = X_train_Nu_tensor[:, 1:2]
t_train_Nu = X_train_Nu_tensor[:, 2:3]

# --- Boundary points: x = 0 or x = max, y = 0 or y = max ---
boundary_mask = np.isclose(coords_array[:, 0], 0) | np.isclose(coords_array[:, 0], coords_array[:, 0].max()) | \
                np.isclose(coords_array[:, 1], 0) | np.isclose(coords_array[:, 1], coords_array[:, 1].max())
X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
y_train_boundary = X_train_boundary_tensor[:, 1:2]
t_train_boundary = X_train_boundary_tensor[:, 2:3]

# --- Initial points: t = 0 ---
initial_mask = np.isclose(coords_array[:, 2], 0)
X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
y_train_initial = X_train_initial_tensor[:, 1:2]
t_train_initial = X_train_initial_tensor[:, 2:3]

k = np.random.rand()

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
            
    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, y, t):
        x = x.requires_grad_()
        y = y.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, y, t)

        du_dx = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]

        du_dy = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

        du_dt = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]

        # PDE residual: k * u_t - k_xx - k_yy - Q = 0
        residual = (self.k * du_dt) - (THERMAL_CONDUCTIVITY * d2u_dx2) - (THERMAL_CONDUCTIVITY * d2u_dy2) - Q
        loss_r = torch.mean(residual ** 2)
        return loss_r

    def loss_initial(self, x, y, t):
        u_pred = self.forward(x, y, t)
        loss_i = torch.mean((u_pred - INITIAL_TEMP) ** 2)
        return loss_i

    def loss_bounds(self, x, y, t):
        x = x.requires_grad_()
        y = y.requires_grad_()
        t = t.requires_grad_()

        u_pred = self.forward(x, y, t)

        dudx = torch.autograd.grad(
            outputs=u_pred,
            inputs=x,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        dudy = torch.autograd.grad(
            outputs=u_pred,
            inputs=y,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True
        )[0]

        # Neumann BC: zero flux at boundaries
        loss_b_x = torch.mean(dudx ** 2)
        loss_b_y = torch.mean(dudy ** 2)
        return loss_b_x + loss_b_y

    def loss_data(self, x, y, t, u):
        u_pred = self.forward(x, y, t)
        loss_d = torch.mean((u_pred - u) ** 2)
        return loss_d

    def losses(self):
        loss_r = self.loss_PDE(x_train_Nu, y_train_Nu, t_train_Nu)
        loss_i = self.loss_initial(x_train_initial, y_train_initial, t_train_initial)
        loss_b = self.loss_bounds(x_train_boundary, y_train_boundary, t_train_boundary)
        loss_d = self.loss_data(x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu)
        return loss_r, loss_i, loss_b, loss_d

    
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


def train_dg_pinn(model, optimizer, iters,
                  x_train, y_train, t_train, u_train,
                  early_stop_patience=500,
                  min_delta=0,
                  epoch_loss_d=[]):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(iters):
        t1 = default_timer()

        optimizer.zero_grad()
        train_loss = model.loss_data(x_train, y_train, t_train, u_train)
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
    x_train=x_train_Nu, y_train=y_train_Nu, t_train=t_train_Nu, u_train=U_train_Nu,
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
