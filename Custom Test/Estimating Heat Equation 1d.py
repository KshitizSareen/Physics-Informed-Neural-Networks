import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
from torch import nn

# Set default types and seeds
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Constants ===
POWER_APPLIED = 0.15
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
Q = 2.192
INITIAL_TEMP = 21.23
THERMAL_CONDUCTIVITY = 10
NUM_DIV = 100
DURATION = 10
LENGTH = 10
TIMESTEP = 0.1
steps = 100000

# === Data Loading ===
def load_data(filepath="temperature_output_from_pinn-1d.csv"):
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
            tempValues.append(temp)
            all_coords.append([x, t_raw])
    
    return sorted(xValues), sorted(tValues), np.array(all_coords), np.array(tempValues)

# Load and normalize data
df = load_data()
xValues, tValues, coords_array, tempValues = prepare_training_data(df)

minX, maxX = coords_array[:, 0].min(), coords_array[:, 0].max()
minT, maxT = coords_array[:, 1].min(), coords_array[:, 1].max()
minTemp, maxTemp = tempValues.min(), tempValues.max()

xScale = maxX - minX
tScale = maxT - minT
tempScale = maxTemp - minTemp

coords_array[:, 0] = (2 * (coords_array[:, 0] - minX) / xScale) - 1
coords_array[:, 1] = ((coords_array[:, 1] - minT) / tScale)
tempValues = (2 * (tempValues - minTemp) / tempScale) - 1

X_train_Nu = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().unsqueeze(1).to(device)
f_hat = torch.zeros(X_train_Nu.shape[0], 1).to(device)

# Normalized bounds (since input is already scaled to [-1, 1])
lb = np.array([-1.0, 0])
ub = np.array([1.0, 1.0])

# === Neural Network ===
class DNN(nn.Module):
    def __init__(self, layers, lb, ub):
        super().__init__()
        self.activation = nn.Tanh()
        self.lb = torch.from_numpy(lb).float().to(device)
        self.ub = torch.from_numpy(ub).float().to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x = (x - self.lb) / (self.ub - self.lb)
        a = x
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

# === Physics-Informed Neural Network ===
class FCN():
    def __init__(self, layers):
        self.iter = 0
        self.dnn = DNN(layers, lb, ub).to(device)
        self.thermal_conductivity_param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True).float().to(device))
        self.dnn.register_parameter('thermal_conductivity', self.thermal_conductivity_param)
        self.loss_function = nn.MSELoss(reduction='mean')

    def loss_data(self, x, y):
        return self.loss_function(self.dnn(x), y)

    def loss_PDE(self, x):
        k = self.thermal_conductivity_param
        g = x.clone()
        g.requires_grad = True
        u = self.dnn(g)
        u_x_t = torch.autograd.grad(u, g, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx_tt = torch.autograd.grad(u_x_t, g, torch.ones_like(u_x_t), create_graph=True)[0]

        u_xx = u_xx_tt[:, [0]]
        u_t = u_x_t[:, [1]]

        d2Tdx2 = ((tempScale / 2) * u_xx * (4 / xScale**2))
        dTdt = ((tempScale / 2) * u_t * (1/tScale))

        residual = d2Tdx2 - ((DENSITY * SPECIFIC_HEAT_CAPACITY) / k) * dTdt + (Q / k)
        return self.loss_function(residual, f_hat)

    def loss(self):
        loss_u = self.loss_data(X_train_Nu, U_train_Nu)
        loss_f = self.loss_PDE(X_train_Nu)
        total_loss = loss_u + loss_f
        print(f"Iter {self.iter}: Data: {loss_u:.6f}, Physics: {loss_f:.6f}, "
              f"k: {self.thermal_conductivity_param.item():.6f}, Total Loss: {total_loss:.6f}")
        self.iter += 1
        return total_loss

    def test(self, x):
        return self.dnn(x)

# === Training ===
layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
PINN = FCN(layers)
params = list(PINN.dnn.parameters()) + [PINN.thermal_conductivity_param]

optimizer = torch.optim.LBFGS(params, lr=0.1,
    max_iter=steps,
    max_eval=None,
    tolerance_grad=1e-11,
    tolerance_change=1e-11,
    history_size=100,
    line_search_fn='strong_wolfe'
)

start_time = time.time()

def closure():
    optimizer.zero_grad()
    loss = PINN.loss()
    loss.backward()
    return loss

optimizer.step(closure)

elapsed = time.time() - start_time
print(f"Training time: {elapsed:.2f} seconds")
