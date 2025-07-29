import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import time

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Training Hyperparameters
steps = 20000
learning_rate = 1e-3

# === Constants (k is unknown) ===
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
Q = 2.192
TRUE_K = 10  # The ground truth value for comparison
INITIAL_TEMP = 21.23
DURATION = 10
LENGTH = 10

# === Data Loading ===
def load_data(filepath="temperature_output_1d.csv"):
    """Loads the temperature data from the CSV file."""
    df = pd.read_csv(filepath)
    return df

# === Prepare Training Data (REWRITTEN FOR YOUR DATA FORMAT) ===
def prepare_training_data(df):
    """
    Transforms the wide-format dataframe into a long-format tensor for training.
    This is now robust for the provided data structure.
    """
    # Drop the 'Q' column as it's not a coordinate or temperature value
    if 'Q' in df.columns:
        df_temp = df.drop(columns=['Q'])
    else:
        df_temp = df.copy()

    # Use pandas.melt to convert from wide to long format
    df_long = df_temp.melt(id_vars=['Timestamp'],
                           var_name='x',
                           value_name='temperature')

    # Convert column 'x' from string (e.g., "0.1") to float
    df_long['x'] = pd.to_numeric(df_long['x'])

    # Extract unique coordinates for normalization and ICs
    x_unique = sorted(df_long['x'].unique())
    t_unique = sorted(df_long['Timestamp'].unique())

    # Get all data points for the data-driven loss term
    x_data = df_long['x'].values
    t_data = df_long['Timestamp'].values
    temp_data = df_long['temperature'].values

    # Stack into coordinate pairs [x, t]
    data_coords = np.stack([x_data, t_data], axis=1)
    
    # Points for initial condition loss (t=0)
    ic_coords = np.stack([np.array(x_unique), np.zeros_like(x_unique)], axis=1)

    return (
        x_unique, t_unique,
        data_coords, temp_data,
        ic_coords,
        df_long # Return for plotting convenience
    )

# --- Load and Process Data ---
df = load_data()
x_values, t_values, data_coords, temp_data, ic_coords, df_long = prepare_training_data(df)

# Normalization scalers
min_x, max_x = x_values[0], x_values[-1]
min_t, max_t = t_values[0], t_values[-1]
x_scale = max_x - min_x
t_scale = max_t - min_t

# Normalize coordinates
data_coords_norm = data_coords.copy()
data_coords_norm[:, 0] = (data_coords[:, 0] - min_x) / x_scale
data_coords_norm[:, 1] = (data_coords[:, 1] - min_t) / t_scale

ic_coords_norm = ic_coords.copy()
ic_coords_norm[:, 0] = (ic_coords[:, 0] - min_x) / x_scale
ic_coords_norm[:, 1] = (ic_coords[:, 1] - min_t) / t_scale


# --- Create Tensors ---
# These points will be used for both the PDE loss and the Data loss
X_data_train = torch.from_numpy(data_coords_norm).float().to(device)
T_data_train = torch.from_numpy(temp_data).float().view(-1, 1).to(device)

# Points for the Initial Condition (IC) loss
X_ic_train = torch.from_numpy(ic_coords_norm).float().to(device)

# For the PDE loss, we need a target of zeros
f_hat = torch.zeros(X_data_train.shape[0], 1).to(device)

#  Deep Neural Network (DNN)
class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh() # Tanh is often a good choice for PINNs
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

# Physics-Informed Neural Network (PINN)
class InversePINN():
    def __init__(self, layers):
        self.iter = 0
        self.dnn = DNN(layers).to(device)
        
        # Make thermal conductivity a trainable parameter with an initial guess
        self.k = torch.nn.Parameter(torch.tensor([1.0], device=device))

        self.loss_function = nn.MSELoss(reduction='mean')

    def loss_pde(self, x_pde):
        g = x_pde.clone()
        g.requires_grad = True
        u = self.dnn(g)
        
        # Calculate first derivatives
        u_grad = torch.autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_x_norm = u_grad[:, [0]]
        u_t_norm = u_grad[:, [1]]
        
        # Calculate second derivative
        u_xx_norm = torch.autograd.grad(u_x_norm, g, torch.ones_like(u_x_norm), create_graph=True)[0][:, [0]]

        # Un-normalize derivatives
        d2Tdx2 = u_xx_norm / (x_scale**2)
        dTdt = u_t_norm / t_scale
        
        # PDE Residual: k * d²T/dx² + Q - ρ*c*dT/dt = 0
        residual = self.k * d2Tdx2 - (DENSITY * SPECIFIC_HEAT_CAPACITY) * dTdt + Q
        
        loss_f = self.loss_function(residual, f_hat)
        return loss_f

    def loss_ic(self, x_ic):
        u_ic_pred = self.dnn(x_ic)
        loss_t = self.loss_function(u_ic_pred, torch.full_like(u_ic_pred, INITIAL_TEMP))
        return loss_t
    
    def loss_data(self, x_data, T_data):
        T_pred = self.dnn(x_data)
        loss = self.loss_function(T_pred, T_data)
        return loss

    def loss(self, x_pde, x_ic, x_data, T_data):
        loss_f = self.loss_pde(x_pde)
        loss_i = self.loss_ic(x_ic)
        loss_d = self.loss_data(x_data, T_data)
        
        # Loss weights - these may need tuning
        w_pde = 1.0
        w_ic = 10.0
        w_data = 100.0 # Weight data loss highest
        
        total_loss = w_pde * loss_f + w_ic * loss_i + w_data * loss_d
        
        if self.iter % 100 == 0:
            print(f"Iter {self.iter}: Total Loss: {total_loss:.4e}, "
                  f"PDE: {loss_f.item():.4e}, IC: {loss_i.item():.4e}, Data: {loss_d.item():.4e}, "
                  f"Learned k: {self.k.item():.4f}")
        
        self.iter += 1
        return total_loss
    
    def predict(self, x_data):
        return self.dnn(x_data)

# --- Training Setup ---
early_stopping_patience = 2000
best_loss = float('inf')
patience_counter = 0
best_model_state = None

layers = np.array([2, 40, 40, 40, 40, 1])
PINN = InversePINN(layers)

# Add the trainable parameter self.k to the optimizer
optimizer = torch.optim.Adam(
    list(PINN.dnn.parameters()) + [PINN.k],
    lr=learning_rate
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)

# --- Training Loop ---
print("Starting training...")
start_time = time.time()
for i in range(steps):
    optimizer.zero_grad()
    loss = PINN.loss(X_data_train, X_ic_train, X_data_train, T_data_train)
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss
        patience_counter = 0
        best_model_state = PINN.dnn.state_dict()
        best_k = PINN.k.clone()
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping at step {i}. Best loss: {best_loss:.4e}")
            if best_model_state:
                PINN.dnn.load_state_dict(best_model_state)
                PINN.k.data = best_k.data
            break

training_time = time.time() - start_time
print(f"Training finished in {training_time:.2f} seconds.")

# --- Evaluation ---
PINN.dnn.eval()
learned_k = PINN.k.item()
print("\n--- Results ---")
print(f"True Thermal Conductivity (k): {TRUE_K}")
print(f"Estimated Thermal Conductivity (k): {learned_k:.4f}")
print(f"Error: {abs(learned_k - TRUE_K) / TRUE_K * 100:.2f}%")


# --- Visualization ---
def plot_results(pinn_model, df_long_data, x_coords, t_coords):
    print("\nGenerating comparison plots...")
    pinn_model.dnn.eval()
    
    plot_times = [t_coords[0], t_coords[len(t_coords) // 3], t_coords[len(t_coords) * 2 // 3], t_coords[-1]]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, t in enumerate(plot_times):
            # Get ground truth data for this timestamp from the long dataframe
            true_data_slice = df_long_data[df_long_data['Timestamp'] == t]
            true_x = true_data_slice['x']
            true_temp = true_data_slice['temperature']
            
            # Prepare input for PINN prediction
            x_plot = np.array(x_coords)
            t_plot = np.full_like(x_plot, t)
            
            x_plot_norm = (x_plot - min_x) / x_scale
            t_plot_norm = (t_plot - min_t) / t_scale
            
            X_infer = torch.tensor(np.stack([x_plot_norm, t_plot_norm], axis=1), dtype=torch.float32).to(device)
            
            T_pred = pinn_model.predict(X_infer).cpu().numpy().flatten()
            
            # Plotting
            ax = axes[i]
            ax.plot(true_x, true_temp, 'o', color='blue', label='Ground Truth Data', markersize=4)
            ax.plot(x_plot, T_pred, 'r-', label='PINN Prediction', linewidth=2)
            ax.set_title(f'Temperature at t = {t:.2f} s')
            ax.set_xlabel('Position (x)')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            
    plt.suptitle(f'Inverse PINN: Ground Truth vs. Prediction (Learned k = {learned_k:.4f})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Run the plotting function
plot_results(PINN, df_long, x_values, t_values)