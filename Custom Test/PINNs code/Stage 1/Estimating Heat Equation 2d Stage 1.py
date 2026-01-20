import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Set default dtype and seeds
torch.set_default_dtype(torch.float32)
seeds_num = 666
torch.manual_seed(seeds_num)
np.random.seed(seeds_num)

# === True Constants (for validation) ===
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
THERMAL_CONDUCTIVITY = 0.1  # Updated to match new data!
Q = 2.192

true_values = {
    "rho": DENSITY,
    "cp":  SPECIFIC_HEAT_CAPACITY,
    "lam": THERMAL_CONDUCTIVITY,
}


def load_data(filepath):
    return pd.read_csv(filepath)


def prepare_training_data(df):
    """Parse CSV into coordinate arrays and temperature values."""
    all_columns = df.columns.tolist()
    
    all_coords = []
    all_temps = []
    
    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(all_columns)):
            column = all_columns[i]
            x, y = map(float, column.strip("()").split(","))
            temp = df.iloc[idx, i]
            all_coords.append([x, y, t_raw])
            all_temps.append([temp])
    
    return np.array(all_coords), np.array(all_temps)


class PINN(nn.Module):
    """
    PINN with ONE trainable physical parameter among: rho, cp, lam.
    Uses LOG-PARAMETERIZATION for better optimization.
    """
    def __init__(
        self,
        input_dim=3,
        output_dim=1,
        hidden_dim=100,
        num_hidden=4,  # Increased depth
        activation="tanh",
        learn_param="lam",
        true_rho=DENSITY,
        true_cp=SPECIFIC_HEAT_CAPACITY,
        true_lam=THERMAL_CONDUCTIVITY,
        init_ranges=None,
    ):
        super().__init__()

        self.learn_param = learn_param.lower().strip()
        if self.learn_param not in {"rho", "cp", "lam", "none"}:
            raise ValueError("learn_param must be one of: 'rho', 'cp', 'lam', 'none'")

        # --- MLP with Xavier initialization ---
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Xavier initialization
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "silu":
            self.activation = torch.nn.functional.silu
        elif activation == "sin":
            self.activation = torch.sin
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Default init ranges (in original parameter space)
        if init_ranges is None:
            init_ranges = {
                "rho": (0.5, 5.0),
                "cp":  (0.1, 2.5),
                "lam": (0.01, 1.0),  # Adjusted for λ=0.1
            }

        # Fixed values as buffers
        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

        # Create trainable parameter in LOG space
        self.log_rho_param = None
        self.log_cp_param  = None
        self.log_lam_param = None

        if self.learn_param != "none":
            lo, hi = init_ranges[self.learn_param]
            init_val = lo + (hi - lo) * torch.rand(1)
            log_init_val = torch.log(init_val).float()

            if self.learn_param == "rho":
                self.log_rho_param = nn.Parameter(log_init_val)
            elif self.learn_param == "cp":
                self.log_cp_param = nn.Parameter(log_init_val)
            elif self.learn_param == "lam":
                self.log_lam_param = nn.Parameter(log_init_val)

    @property
    def rho(self):
        if self.log_rho_param is not None:
            return torch.exp(self.log_rho_param)
        return self.rho_fixed

    @property
    def cp(self):
        if self.log_cp_param is not None:
            return torch.exp(self.log_cp_param)
        return self.cp_fixed

    @property
    def lam(self):
        if self.log_lam_param is not None:
            return torch.exp(self.log_lam_param)
        return self.lam_fixed

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, y, t):
        """PDE residual: rho*cp*u_t - lam*(u_xx + u_yy) - Q = 0"""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                    create_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                   retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                    create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                                   create_graph=True)[0]

        residual = (self.rho * self.cp * u_t) - (self.lam * (u_xx + u_yy)) - Q
        return torch.mean(residual ** 2)

    def loss_initial(self, x, y, t, u_init):
        """Initial condition loss."""
        u = self.forward(x, y, t)
        return torch.mean((u - u_init) ** 2)

    def loss_bounds(self, x, y, t):
        """Zero Neumann BC: du/dn = 0 at boundaries."""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                   create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                   create_graph=True, retain_graph=True)[0]

        return torch.mean(u_x ** 2) + torch.mean(u_y ** 2)

    def loss_data(self, x, y, t, u_obs):
        """Data fitting loss."""
        u = self.forward(x, y, t)
        return torch.mean((u - u_obs) ** 2)


def pde_residual_rmse(model, x, y, t):
    """Compute RMSE of PDE residual."""
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model.forward(x, y, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                               retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                               retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), 
                               create_graph=True)[0]

    r = (model.rho * model.cp * u_t) - (model.lam * (u_xx + u_yy)) - Q
    return torch.sqrt(torch.mean(r ** 2)).detach()


def field_l2_error(model, x, y, t, u_obs):
    """Compute relative L2 error."""
    with torch.no_grad():
        u_pred = model.forward(x, y, t)
        num = torch.norm(u_pred - u_obs)
        den = torch.norm(u_obs) + 1e-12
        return (num / den).detach()


def main(param_to_learn, data_filepath):
    print(f"\n{'='*60}")
    print(f"Training PINN to learn: {param_to_learn}")
    print(f"True value: {true_values[param_to_learn]}")
    print(f"{'='*60}\n")
    
    # Load and prepare data
    df = load_data(data_filepath)
    coords_array, tempValues = prepare_training_data(df)
    
    print(f"Data shape: {coords_array.shape[0]} points")
    print(f"X range: [{coords_array[:, 0].min():.2f}, {coords_array[:, 0].max():.2f}]")
    print(f"Y range: [{coords_array[:, 1].min():.2f}, {coords_array[:, 1].max():.2f}]")
    print(f"T range: [{coords_array[:, 2].min():.2f}, {coords_array[:, 2].max():.2f}]")
    print(f"Temp range: [{tempValues.min():.2f}, {tempValues.max():.2f}]")
    
    # Convert to tensors
    X_train = torch.from_numpy(coords_array).float().to(device)
    U_train = torch.from_numpy(tempValues).float().to(device)
    x_train = X_train[:, 0:1]
    y_train = X_train[:, 1:2]
    t_train = X_train[:, 2:3]

    # Boundary points
    boundary_mask = (
        np.isclose(coords_array[:, 0], 0) | 
        np.isclose(coords_array[:, 0], coords_array[:, 0].max()) |
        np.isclose(coords_array[:, 1], 0) | 
        np.isclose(coords_array[:, 1], coords_array[:, 1].max())
    )
    X_boundary = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
    x_boundary = X_boundary[:, 0:1]
    y_boundary = X_boundary[:, 1:2]
    t_boundary = X_boundary[:, 2:3]

    # Initial points (t=0)
    initial_mask = np.isclose(coords_array[:, 2], 0)
    X_initial = torch.from_numpy(coords_array[initial_mask]).float().to(device)
    U_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
    x_initial = X_initial[:, 0:1]
    y_initial = X_initial[:, 1:2]
    t_initial = X_initial[:, 2:3]

    print(f"\nBoundary points: {boundary_mask.sum()}")
    print(f"Initial points: {initial_mask.sum()}")

    # Initialize model
    torch.manual_seed(seeds_num)
    model = PINN(
        learn_param=param_to_learn,
        input_dim=3,
        output_dim=1,
        hidden_dim=100,
        num_hidden=4,
        activation='tanh'
    ).to(device)

    # Loss weights
    lambda_d = 1.0    # Data
    lambda_r = 0.1    # PDE (start smaller, increase)
    lambda_b = 0.1    # Boundary
    lambda_i = 1.0    # Initial condition

    # History tracking
    history = {
        "L_total": [], "L_pde": [], "L_ic": [], "L_bc": [], "L_data": [],
        "rho": [], "cp": [], "lam": [], "time_sec": [],
    }

    t_start = default_timer()

    # ==================== Phase 1: Adam ====================
    print("\n--- Phase 1: Adam Optimizer ---")
    adam_iters = 15000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    early_stop_patience = 3000

    for it in range(adam_iters):
        optimizer.zero_grad()

        Lr = model.loss_PDE(x_train, y_train, t_train)
        Li = model.loss_initial(x_initial, y_initial, t_initial, U_initial)
        Lb = model.loss_bounds(x_boundary, y_boundary, t_boundary)
        Ld = model.loss_data(x_train, y_train, t_train, U_train)

        L = (lambda_r * Lr) + (lambda_i * Li) + (lambda_b * Lb) + (lambda_d * Ld)
        L.backward()
        optimizer.step()
        scheduler.step(L)

        curr = float(L.detach())

        # Early stopping
        if curr < best_loss - 1e-8:
            best_loss = curr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        elapsed = default_timer() - t_start
        history["L_total"].append(curr)
        history["L_pde"].append(float(Lr.detach()))
        history["L_ic"].append(float(Li.detach()))
        history["L_bc"].append(float(Lb.detach()))
        history["L_data"].append(float(Ld.detach()))
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["time_sec"].append(elapsed)

        if it % 500 == 0 or it == adam_iters - 1:
            param_val = float(getattr(model, param_to_learn).detach())
            true_val = true_values[param_to_learn]
            rel_err = abs(param_val - true_val) / true_val * 100
            print(
                f"[Adam {it:05d}] L={curr:.3e} | "
                f"Ld={float(Ld):.3e} Lr={float(Lr):.3e} | "
                f"{param_to_learn}={param_val:.4f} (true={true_val}, err={rel_err:.1f}%)"
            )

        if patience_counter >= early_stop_patience:
            print(f"[Adam] Early stopping at iter {it}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ==================== Phase 2: L-BFGS ====================
    print("\n--- Phase 2: L-BFGS Optimizer ---")
    lbfgs_iters = 1000
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        Lr = model.loss_PDE(x_train, y_train, t_train)
        Li = model.loss_initial(x_initial, y_initial, t_initial, U_initial)
        Lb = model.loss_bounds(x_boundary, y_boundary, t_boundary)
        Ld = model.loss_data(x_train, y_train, t_train, U_train)
        L = (lambda_r * Lr) + (lambda_i * Li) + (lambda_b * Lb) + (lambda_d * Ld)
        L.backward()
        return L

    best_lbfgs_loss = float("inf")
    lbfgs_counter = 0

    for it in range(lbfgs_iters):
        L = optimizer.step(closure)
        curr = float(L.detach())

        if curr < best_lbfgs_loss - 1e-10:
            best_lbfgs_loss = curr
            lbfgs_counter = 0
        else:
            lbfgs_counter += 1

        # Recompute individual losses for logging
        with torch.no_grad():
            Lr = model.loss_PDE(x_train, y_train, t_train)
            Li = model.loss_initial(x_initial, y_initial, t_initial, U_initial)
            Lb = model.loss_bounds(x_boundary, y_boundary, t_boundary)
            Ld = model.loss_data(x_train, y_train, t_train, U_train)

        elapsed = default_timer() - t_start
        history["L_total"].append(curr)
        history["L_pde"].append(float(Lr.detach()))
        history["L_ic"].append(float(Li.detach()))
        history["L_bc"].append(float(Lb.detach()))
        history["L_data"].append(float(Ld.detach()))
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["time_sec"].append(elapsed)

        if it % 50 == 0 or it == lbfgs_iters - 1:
            param_val = float(getattr(model, param_to_learn).detach())
            true_val = true_values[param_to_learn]
            rel_err = abs(param_val - true_val) / true_val * 100
            print(
                f"[LBFGS {it:04d}] L={curr:.3e} | "
                f"{param_to_learn}={param_val:.4f} (err={rel_err:.1f}%)"
            )

        if lbfgs_counter >= 100:
            print(f"[L-BFGS] Early stopping at iter {it}")
            break

    # ==================== Results ====================
    t_total = default_timer() - t_start
    
    final_param = float(getattr(model, param_to_learn).detach())
    true_param = true_values[param_to_learn]
    final_error = abs(final_param - true_param) / true_param * 100

    print(f"\n{'='*60}")
    print(f"RESULTS for {param_to_learn}")
    print(f"{'='*60}")
    print(f"True value:      {true_param}")
    print(f"Estimated value: {final_param:.6f}")
    print(f"Relative error:  {final_error:.2f}%")
    print(f"Training time:   {t_total/60:.2f} min")
    print(f"Final data loss: {history['L_data'][-1]:.3e}")
    print(f"Final PDE loss:  {history['L_pde'][-1]:.3e}")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Parameter convergence
    ax = axes[0]
    ax.plot(history["time_sec"], history[param_to_learn], 'b-', linewidth=2, label=f'Estimated {param_to_learn}')
    ax.axhline(y=true_param, color='r', linestyle='--', linewidth=2, label=f'True {param_to_learn}')
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel(param_to_learn)
    ax.set_title(f"Convergence of {param_to_learn}")
    ax.legend()
    ax.grid(True)

    # Loss convergence
    ax = axes[1]
    ax.semilogy(history["time_sec"], history["L_total"], 'b-', label='Total', alpha=0.8)
    ax.semilogy(history["time_sec"], history["L_data"], 'g-', label='Data', alpha=0.8)
    ax.semilogy(history["time_sec"], history["L_pde"], 'r-', label='PDE', alpha=0.8)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Convergence")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = f"/mnt/user-data/outputs/convergence_{param_to_learn}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to: {save_path}")
    plt.close()

    return final_param, final_error, history


if __name__ == "__main__":
    data_file = "temperature_output copy 3.csv"
    
    results = {}
    for param in ["lam", "cp", "rho"]:
        est_val, error, hist = main(param, data_file)
        results[param] = {"estimated": est_val, "error": error}
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for param, res in results.items():
        print(f"{param}: estimated={res['estimated']:.4f}, true={true_values[param]}, error={res['error']:.1f}%")