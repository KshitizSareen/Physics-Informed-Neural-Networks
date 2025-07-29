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
def load_data(filepath="temperature_output_1d.csv"):
    return pd.read_csv(filepath)

# === Prepare Training Data ===
def prepare_training_data(df):
    xValues = set()
    tValues = set()
    tempValues = []
    all_coords = []
    columns = df.columns.tolist()
    x_data = np.linspace(0, 1, 100)
    t_data = np.linspace(0, 1, 100)
    beta_1_true = 1/(20)
    for x in x_data:
        for t in t_data:
            temp = np.exp(-(10*np.pi*beta_1_true)**2*t)*np.sin(10*np.pi*x)
            tempValues.append([temp])
            all_coords.append([x, t])
    
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



# --- All training data ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
t_train_Nu = X_train_Nu_tensor[:, 1:2]

# --- Boundary points: x = 0 or x = 1 ---
boundary_mask = np.isclose(coords_array[:, 0], 0.0) | np.isclose(coords_array[:, 0], 10.0)
X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
t_train_boundary = X_train_boundary_tensor[:, 1:2]

# --- Initial points: t = 0 ---
initial_mask = np.isclose(coords_array[:, 1], 0.0)
X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
t_train_initial = X_train_initial_tensor[:, 1:2]


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
            
    def forward(self, x, t):
        out = torch.cat([x, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, t):
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
        
        u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]


        residual =   u_t - self.k**2*u_xx
        loss_r = torch.mean(residual ** 2)
        return loss_r
    
    def loss_initial(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
        u_0 = torch.sin(10*np.pi*x)

        loss_i = torch.mean((u_pred - u_0) ** 2)
        return loss_i
    
    def loss_bounds(self, x, t):    
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)

        loss_b = torch.mean(u_pred ** 2)
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
    global lambda1, lambda2, lambda3, lambda4
    
    optimizer.zero_grad()

    # Calculate losses
    loss_r, loss_i, loss_b, loss_d = model.losses()

    # Calculate total loss with updated beta1
    loss = lambda1*loss_r + lambda2*loss_i + lambda3*loss_b + lambda4*loss_d

    # Backpropagation
    loss.backward()

    # Append losses for monitoring
    epoch_loss_r.append(loss_r.item())
    epoch_loss_b.append(loss_b.item())
    epoch_loss_i.append(loss_i.item())
    epoch_loss_d.append(loss_d.item())

    epoch_beta.append(model.k.item())
    
    epoch_lambda1.append(lambda1)
    epoch_lambda2.append(lambda2)
    epoch_lambda3.append(lambda3)
    epoch_lambda4.append(lambda4)
    
        # lambda1, lambda2, lambda3 = Adap_weights(model, X_train)
    print('Epoch %d,  loss = %e, loss_r = %e, loss_i = %e, loss_b = %e, loss_d = %e, beta = %f, lambda1= %e , lambda2= %e, lambda3= %e, lambda4=%e' %
            (epoch, float(loss), float(loss_r), float(loss_i), float(loss_b), float(loss_d),
            model.k.item(), float(lambda1), float(lambda2), float(lambda3), float(lambda4)))
        
    model.epoch += 1
    return loss

def train_dg_pinn(model, optimizer, iters=50001,
           epoch_loss_d=[]):
    for epoch in range(iters):
        t1 = default_timer()
        optimizer.zero_grad()
        loss_d = model.loss_data(x_train_Nu, t_train_Nu, U_train_Nu)
        loss = loss_d
        loss.backward()
        optimizer.step()
        epoch_loss_d.append(loss_d.item())
        t2 = default_timer()
        print('Epoch %d, time = %e, loss = %e,  lambda1 = %e' %
                (epoch, float(t2-t1), float(loss),  model.k.item()))
        
    return epoch_loss_d

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

iter_1s = 2000
iter_2 = 10000  # Maximun number  of iterations for L-BFGS optimizer
t11 = default_timer()
# Adam optimizer to decrease loss in Phase 1
optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-3)
epoch_loss_d = train_dg_pinn(model, optimizer,  iters=iter_1s, epoch_loss_d=[])
lambda1, lambda2, lambda3, lambda4 = 1,1,1,1

# L-BFGS optimizer for fine-tuning in Phase 2
optimizer = torch.optim.LBFGS(list(model.parameters()), lr=1e-1, max_iter=1,
                            history_size=100)
for epoch in range(iter_2):
    optimizer.zero_grad()
    optimizer.step(closure)

t22 = default_timer()
print('Time elapsed: %.2f min' % ((t22 - t11) / 60))

print(model.k.item())
