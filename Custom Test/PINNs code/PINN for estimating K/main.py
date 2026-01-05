import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN(nn.Module):
    """Physics-Informed Neural Network"""
    def __init__(self, hidden_layers=[128, 128, 128, 128]):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_size = 2  # Input: [x, t]
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Learnable thermal conductivity parameter
        self.log_k = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)
    
    @property
    def thermal_conductivity(self):
        return torch.exp(self.log_k)

def compute_physics_loss(model, x, t):
    """Compute physics loss based on heat equation: du/dt = k * d²u/dx²"""
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    
    u = model(x, t)
    
    # First derivatives
    du_dt = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    
    # Second derivative
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0]
    
    # Heat equation residual
    k = model.thermal_conductivity
    residual = du_dt - k * d2u_dx2
    
    return torch.mean(residual**2)

def generate_training_data(n_initial=200, n_boundary=200, n_collocation=5000, n_interior=1000, k_true=10.0):
    """Generate training data for heat equation with true thermal conductivity k_true"""
    
    # Initial condition: t=0, x ∈ [0, 1]
    x_initial = torch.linspace(0, 1, n_initial).reshape(-1, 1)
    t_initial = torch.zeros(n_initial, 1)
    u_initial = torch.sin(np.pi * x_initial)
    
    # Boundary conditions: x=0 and x=1, t ∈ [0, 0.5]
    t_boundary = torch.linspace(0, 0.5, n_boundary//2).reshape(-1, 1)
    x_boundary_left = torch.zeros(n_boundary//2, 1)
    x_boundary_right = torch.ones(n_boundary//2, 1)
    x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
    t_boundary_full = torch.cat([t_boundary, t_boundary], dim=0)
    u_boundary = torch.zeros(n_boundary, 1)
    
    # Collocation points for physics loss - focus on region where solution is active
    x_collocation = torch.rand(n_collocation, 1)
    t_collocation = torch.rand(n_collocation, 1) * 0.3  # Focus on early times
    
    # Dense interior data from analytical solution: u(x,t) = sin(πx)exp(-k*π²*t)
    # Use a grid for better coverage
    n_grid = int(np.sqrt(n_interior))
    x_grid = torch.linspace(0.1, 0.9, n_grid)
    t_grid = torch.linspace(0.001, 0.2, n_grid)
    X_grid, T_grid = torch.meshgrid(x_grid, t_grid, indexing='ij')
    x_interior = X_grid.reshape(-1, 1)
    t_interior = T_grid.reshape(-1, 1)
    u_interior = torch.sin(np.pi * x_interior) * torch.exp(-k_true * np.pi**2 * t_interior)
    
    return {
        'initial': (x_initial, t_initial, u_initial),
        'boundary': (x_boundary, t_boundary_full, u_boundary),
        'interior': (x_interior, t_interior, u_interior),
        'collocation': (x_collocation, t_collocation)
    }

def train_pinn(model, data, n_epochs=10000):
    """Train the PINN model using Adam optimizer"""
    
    x_init, t_init, u_init = data['initial']
    x_bound, t_bound, u_bound = data['boundary']
    x_interior, t_interior, u_interior = data['interior']
    x_coll, t_coll = data['collocation']
    
    history = {'total': [], 'data': [], 'physics': [], 'interior': [], 'k': []}
    
    # Use separate optimizers with different learning rates
    optimizer_net = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    optimizer_k = torch.optim.Adam([model.log_k], lr=1e-2)
    
    # Learning rate schedulers
    scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, patience=1000, factor=0.5, min_lr=1e-6)
    scheduler_k = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_k, patience=1000, factor=0.5, min_lr=1e-6)
    
    best_k_error = float('inf')
    best_k = 1.0
    
    for epoch in range(n_epochs):
        optimizer_net.zero_grad()
        optimizer_k.zero_grad()
        
        # Data loss (initial and boundary conditions)
        u_init_pred = model(x_init, t_init)
        u_bound_pred = model(x_bound, t_bound)
        u_interior_pred = model(x_interior, t_interior)
        
        loss_init = torch.mean((u_init_pred - u_init)**2)
        loss_bound = torch.mean((u_bound_pred - u_bound)**2)
        loss_interior = torch.mean((u_interior_pred - u_interior)**2)
        
        # Physics loss
        loss_physics = compute_physics_loss(model, x_coll, t_coll)
        
        # Weighted total loss
        # Interior loss is most important for identifying k
        loss = loss_init + loss_bound + 100.0 * loss_interior + 0.1 * loss_physics
        
        loss.backward()
        optimizer_net.step()
        optimizer_k.step()
        
        scheduler_net.step(loss)
        scheduler_k.step(loss_interior)
        
        # Record history
        history['total'].append(loss.item())
        history['data'].append((loss_init + loss_bound).item())
        history['interior'].append(loss_interior.item())
        history['physics'].append(loss_physics.item())
        history['k'].append(model.thermal_conductivity.item())
        
        # Track best k
        k_error = abs(model.thermal_conductivity.item() - 10.0)
        if k_error < best_k_error:
            best_k_error = k_error
            best_k = model.thermal_conductivity.item()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"  Total Loss: {loss.item():.6e}")
            print(f"  Interior Loss: {loss_interior.item():.6e}")
            print(f"  Physics Loss: {loss_physics.item():.6e}")
            print(f"  k (current): {model.thermal_conductivity.item():.4f}")
            print(f"  k (best): {best_k:.4f} (error: {best_k_error:.4f})")
            print(f"  LR_net: {optimizer_net.param_groups[0]['lr']:.2e}, LR_k: {optimizer_k.param_groups[0]['lr']:.2e}")
            print()
    
    return history

def plot_results(model, history, true_k=10.0):
    """Visualize results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Solution field
    x_plot = torch.linspace(0, 1, 100).reshape(-1, 1)
    t_plot = torch.linspace(0, 0.3, 100).reshape(-1, 1)
    X, T = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing='ij')
    
    with torch.no_grad():
        U_pred = model(X.reshape(-1, 1), T.reshape(-1, 1)).reshape(X.shape)
    U_true = torch.sin(np.pi * X) * torch.exp(-true_k * np.pi**2 * T)
    
    c = axes[0, 0].contourf(T.numpy(), X.numpy(), U_pred.numpy(), levels=50, cmap='hot')
    axes[0, 0].set_xlabel('Time (t)')
    axes[0, 0].set_ylabel('Position (x)')
    axes[0, 0].set_title('Predicted Temperature Field')
    plt.colorbar(c, ax=axes[0, 0])
    
    # Plot 2: True solution
    c2 = axes[0, 1].contourf(T.numpy(), X.numpy(), U_true.numpy(), levels=50, cmap='hot')
    axes[0, 1].set_xlabel('Time (t)')
    axes[0, 1].set_ylabel('Position (x)')
    axes[0, 1].set_title('True Temperature Field')
    plt.colorbar(c2, ax=axes[0, 1])
    
    # Plot 3: Error
    error = np.abs(U_pred.numpy() - U_true.numpy())
    c3 = axes[0, 2].contourf(T.numpy(), X.numpy(), error, levels=50, cmap='viridis')
    axes[0, 2].set_xlabel('Time (t)')
    axes[0, 2].set_ylabel('Position (x)')
    axes[0, 2].set_title('Absolute Error')
    plt.colorbar(c3, ax=axes[0, 2])
    
    # Plot 4: Loss history
    axes[1, 0].semilogy(history['total'], label='Total Loss', alpha=0.7)
    axes[1, 0].semilogy(history['interior'], label='Interior Loss', alpha=0.7)
    axes[1, 0].semilogy(history['physics'], label='Physics Loss', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss History')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Thermal conductivity estimate
    axes[1, 1].plot(history['k'], linewidth=2)
    axes[1, 1].axhline(y=true_k, color='r', linestyle='--', linewidth=2, label=f'True k = {true_k}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Thermal Conductivity (k)')
    axes[1, 1].set_title(f'Estimated k (Final: {history["k"][-1]:.4f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 20])
    
    # Plot 6: Temperature profiles at different times
    times = [0.01, 0.05, 0.1, 0.15, 0.2]
    for t_val in times:
        t_tensor = torch.ones_like(x_plot) * t_val
        with torch.no_grad():
            u_pred = model(x_plot, t_tensor)
        u_true = np.sin(np.pi * x_plot.detach().numpy()) * np.exp(-true_k * np.pi**2 * t_val)
        axes[1, 2].plot(x_plot.numpy(), u_pred.numpy(), label=f't={t_val}', linestyle='-', linewidth=2)
        axes[1, 2].plot(x_plot.numpy(), u_true, linestyle='--', alpha=0.5, linewidth=1)
    
    axes[1, 2].set_xlabel('Position (x)')
    axes[1, 2].set_ylabel('Temperature (u)')
    axes[1, 2].set_title('Temperature Profiles (solid: pred, dashed: true)')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("Physics-Informed Neural Network for Thermal Conductivity Estimation")
    print("Heat Equation: du/dt = k * d²u/dx²")
    print("=" * 70)
    print()
    
    # Create model
    model = PINN(hidden_layers=[128, 128, 128, 128])
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Initial k = {model.thermal_conductivity.item():.4f}")
    print()
    
    # Generate training data with k=10
    data = generate_training_data(
        n_initial=200, 
        n_boundary=200, 
        n_collocation=5000,
        n_interior=1024,
        k_true=10.0
    )
    print("Training data generated:")
    print(f"  Initial conditions: {data['initial'][0].shape[0]} points")
    print(f"  Boundary conditions: {data['boundary'][0].shape[0]} points")
    print(f"  Interior data points: {data['interior'][0].shape[0]} points")
    print(f"  Collocation points: {data['collocation'][0].shape[0]} points")
    print(f"  True thermal conductivity: k = 10.0")
    print()
    
    # Train model
    print("Starting training...")
    print()
    history = train_pinn(model, data, n_epochs=10000)
    
    # Final results
    print("=" * 70)
    print("Training Complete!")
    print(f"Final estimated thermal conductivity: k = {model.thermal_conductivity.item():.4f}")
    print(f"True thermal conductivity: k = 10.0")
    print(f"Absolute error: {abs(model.thermal_conductivity.item() - 10.0):.4f}")
    print(f"Relative error: {abs(model.thermal_conductivity.item() - 10.0)/10.0 * 100:.2f}%")
    print("=" * 70)
    
    # Plot results
    plot_results(model, history, true_k=10.0)