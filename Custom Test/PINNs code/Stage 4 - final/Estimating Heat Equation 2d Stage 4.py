import torch
import numpy as np
import matplotlib.pyplot as plt
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
THERMAL_CONDUCTIVITY = 10.0

# Map parameter name → true value
true_values = {
    "lam": THERMAL_CONDUCTIVITY,
}


def generate_training_data(n_initial=100, n_boundary=100, n_collocation=1000, n_interior=1000, k_true=10.0):
    """Generate training data for heat equation"""
    # Spatial domain: x ∈ [0, 1], temporal domain: t ∈ [0, 1]
    
    # Initial condition: t=0, x ∈ [0, 1]
    x_initial = torch.rand(n_initial, 1)
    t_initial = torch.zeros(n_initial, 1)
    u_initial = torch.sin(np.pi * x_initial)  # u(x, 0) = sin(πx)
    
    # Boundary conditions: x=0 and x=1, t ∈ [0, 1]
    t_boundary = torch.rand(n_boundary, 1)
    x_boundary_left = torch.zeros(n_boundary//2, 1)
    x_boundary_right = torch.ones(n_boundary//2, 1)
    x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
    t_boundary_full = torch.cat([t_boundary[:n_boundary//2], t_boundary[n_boundary//2:]], dim=0)
    u_boundary = torch.zeros(n_boundary, 1)  # u(0, t) = u(1, t) = 0
    
    # Collocation points for physics loss
    x_collocation = torch.rand(n_collocation, 1)
    t_collocation = torch.rand(n_collocation, 1)
    
    # Interior data points from analytical solution: u(x,t) = sin(πx)exp(-k*π²*t)
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


class PINN(nn.Module):
    """
    1D PINN with trainable thermal conductivity parameter.
    Heat equation: u_t = k * u_xx
    """
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        hidden_dim=128,
        num_hidden=4,
        activation="tanh",
        eps=1e-12,
    ):
        super().__init__()
        self.eps = eps

        # --- MLP ---
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "silu":
            self.activation = torch.nn.functional.silu
        elif activation == "sin":
            self.activation = torch.sin
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Learnable thermal conductivity (log-parameterized for positivity)
        self.log_k = nn.Parameter(torch.tensor(0.0))

    @property
    def k(self):
        return torch.exp(self.log_k)

    def forward(self, x, t):
        out = torch.cat([x, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    # 1D Heat equation: u_t = k * u_xx
    def loss_PDE(self, x, t):
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = self.forward(x, t)

        # Compute derivatives
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        # PDE residual
        residual = u_t - self.k * u_xx
        return torch.mean(residual ** 2)

    def loss_data(self, x, t, u_obs):
        u = self.forward(x, t)
        return torch.mean((u - u_obs) ** 2)


def main():
    torch.manual_seed(seeds_num)

    model = PINN(
        input_dim=2,
        output_dim=1,
        hidden_dim=128,
        num_hidden=4,
        activation="tanh"
    ).to(device)

    def get_k():
        return float(model.k.detach().cpu().item())

    true_k = THERMAL_CONDUCTIVITY

    history = {
        'total': [], 
        'data': [], 
        'physics': [], 
        'k': [],
        'time_sec': []
    }

    # Generate training data
    print("Generating training data...")
    data = generate_training_data(
        n_initial=100, 
        n_boundary=100, 
        n_collocation=5000,
        n_interior=1024,
        k_true=true_k
    )
    
    x_init, t_init, u_init = data['initial']
    x_bound, t_bound, u_bound = data['boundary']
    x_interior, t_interior, u_interior = data['interior']
    x_coll, t_coll = data['collocation']
    
    # Move to device
    x_init, t_init, u_init = x_init.to(device), t_init.to(device), u_init.to(device)
    x_bound, t_bound, u_bound = x_bound.to(device), t_bound.to(device), u_bound.to(device)
    x_interior, t_interior, u_interior = x_interior.to(device), t_interior.to(device), u_interior.to(device)
    x_coll, t_coll = x_coll.to(device), t_coll.to(device)

    print(f"\nTraining data sizes:")
    print(f"  Initial: {x_init.shape[0]}")
    print(f"  Boundary: {x_bound.shape[0]}")
    print(f"  Interior: {x_interior.shape[0]}")
    print(f"  Collocation: {x_coll.shape[0]}")

    t_start = default_timer()
    
    # Separate optimizers with different learning rates
    optimizer_net = torch.optim.Adam(
        [p for (n, p) in model.named_parameters() if p.requires_grad and not n.endswith("log_k")], 
        lr=1e-3
    )
    optimizer_k = torch.optim.Adam([model.log_k], lr=1e-2)

    scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_net, patience=1000, factor=0.5, min_lr=1e-6
    )
    scheduler_k = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_k, patience=1000, factor=0.5, min_lr=1e-6
    )

    n_epochs = 10000
    
    best_k_error = float('inf')
    best_k = get_k()

    print(f"\nInitial k: {get_k():.4f}")
    print(f"Target k: {true_k:.4f}\n")

    for i in range(n_epochs):
        optimizer_net.zero_grad()
        optimizer_k.zero_grad()

        # Data losses
        loss_init = model.loss_data(x_init, t_init, u_init)
        loss_bound = model.loss_data(x_bound, t_bound, u_bound)
        loss_interior = model.loss_data(x_interior, t_interior, u_interior)
        
        # Physics loss on collocation points
        loss_physics = model.loss_PDE(x_coll, t_coll)

        # Combined loss - emphasize interior data and physics
        loss = loss_init + loss_bound + (100.0 * loss_interior) + (10.0 * loss_physics)

        loss.backward()
        optimizer_net.step()
        optimizer_k.step()
        
        scheduler_net.step(loss)
        scheduler_k.step(loss_interior)
        
        # Track best k
        current_k = get_k()
        k_error = abs(current_k - true_k)
        if k_error < best_k_error:
            best_k_error = k_error
            best_k = current_k
        
        # Record history
        elapsed_time = default_timer() - t_start
        history['total'].append(loss.item())
        history['data'].append((loss_init + loss_bound + loss_interior).item())
        history['physics'].append(loss_physics.item())
        history['k'].append(current_k)
        history['time_sec'].append(elapsed_time)
        
        print(
            f"[Epoch {i+1:05d}] L={loss:.3e} "
            f"(Li={float(loss_init):.3e}, Lb={float(loss_bound):.3e}, "
            f"Lint={float(loss_interior):.3e}, Lphy={float(loss_physics):.3e}) | "
            f"k={current_k:.4f} (true={true_k:.2f}, best={best_k:.4f}, err={k_error:.4f})"
        )

    # Final results
    t_total = default_timer() - t_start
    print(f"\n{'='*70}")
    print(f"Total training time: {t_total/60:.2f} min")
    print(f"Final k: {get_k():.6f}")
    print(f"True k: {true_k:.6f}")
    print(f"Absolute Error: {abs(get_k() - true_k):.6f}")
    print(f"Relative Error: {abs(get_k() - true_k)/true_k*100:.2f}%")
    print(f"Best k achieved: {best_k:.6f} (error: {best_k_error:.6f})")
    print(f"{'='*70}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Parameter convergence
    axes[0, 0].plot(history["time_sec"], history["k"], label="Estimated k", linewidth=2)
    axes[0, 0].axhline(y=true_k, color="black", linestyle="--", linewidth=2, label=f"True k = {true_k}")
    axes[0, 0].set_xlabel("Time (sec)")
    axes[0, 0].set_ylabel("Thermal Conductivity (k)")
    axes[0, 0].set_title("Convergence of k")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_ylim([0, 20])
    
    # Loss history
    axes[0, 1].semilogy(history['time_sec'], history['total'], label='Total Loss', alpha=0.7, linewidth=2)
    axes[0, 1].semilogy(history['time_sec'], history['data'], label='Data Loss', alpha=0.7, linewidth=2)
    axes[0, 1].semilogy(history['time_sec'], history['physics'], label='Physics Loss', alpha=0.7, linewidth=2)
    axes[0, 1].set_xlabel('Time (sec)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss History')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Predicted temperature field
    x_plot = torch.linspace(0, 1, 100).reshape(-1, 1).to(device)
    t_plot = torch.linspace(0, 0.3, 100).reshape(-1, 1).to(device)
    X_pred, T_pred = torch.meshgrid(x_plot.squeeze(), t_plot.squeeze(), indexing='ij')
    
    with torch.no_grad():
        U_pred = model(X_pred.reshape(-1, 1), T_pred.reshape(-1, 1)).reshape(X_pred.shape)
    
    # True solution
    U_true = torch.sin(np.pi * X_pred) * torch.exp(-true_k * np.pi**2 * T_pred)
    
    c = axes[1, 0].contourf(T_pred.cpu().numpy(), X_pred.cpu().numpy(), U_pred.cpu().numpy(), levels=50, cmap='hot')
    axes[1, 0].set_xlabel('Time (t)')
    axes[1, 0].set_ylabel('Position (x)')
    axes[1, 0].set_title('Predicted Temperature Field')
    plt.colorbar(c, ax=axes[1, 0])
    
    # True vs predicted temperature profiles
    times = [0.01, 0.05, 0.1, 0.15, 0.2]
    for t_val in times:
        t_tensor = torch.ones_like(x_plot) * t_val
        with torch.no_grad():
            u_pred = model(x_plot, t_tensor)
        
        u_true = torch.sin(np.pi * x_plot) * torch.exp(-true_k * np.pi**2 * t_val)
        
        axes[1, 1].plot(x_plot.cpu().numpy(), u_pred.cpu().numpy(), label=f't={t_val} (pred)', linewidth=2)
        axes[1, 1].plot(x_plot.cpu().numpy(), u_true.cpu().numpy(), '--', label=f't={t_val} (true)', linewidth=1, alpha=0.7)
    
    axes[1, 1].set_xlabel('Position (x)')
    axes[1, 1].set_ylabel('Temperature (u)')
    axes[1, 1].set_title('Temperature Profiles: Predicted vs True')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()

    save_path = "convergence_k_simple.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()