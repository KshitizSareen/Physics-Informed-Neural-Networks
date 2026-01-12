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

# Map parameter name → true value
true_values = {
    "rho": DENSITY,
    "cp":  SPECIFIC_HEAT_CAPACITY,
    "lam": THERMAL_CONDUCTIVITY,
}


def load_data(filepath="temperature_output.csv"):
    return pd.read_csv(filepath)

# === Prepare Training Data ===
def prepare_training_data(df):
    xValues = set()
    tValues = set()
    all_columns = df.columns.tolist()

    all_coords = []
    all_temps = []
    for idx in range(len(df)):
        t_raw = df["Timestamp"].iloc[idx]
        for i in range(2, len(all_columns)):
            column = all_columns[i]
            # Parse as single x coordinate instead of (x,y)
            x, y = map(float, column.strip("()").split(","))
            temp = df.iloc[idx, i]

            xValues.add(x)
            tValues.add(t_raw)
            all_coords.append([x, t_raw])
            all_temps.append([temp])
    return (
        np.array(all_coords), np.array(all_temps)
    )


# Load raw data
df = load_data() 
coords_array, tempValues = prepare_training_data(df)

print("\n=== BASIC DATA CHECK ===")
print(f"Total rows in CSV: {len(df)}")
print(f"Total columns in CSV: {len(df.columns)}")
print(f"First few column names: {df.columns[:5].tolist()}")
print(f"coords_array shape: {coords_array.shape}")
print(f"tempValues shape: {tempValues.shape}")
print(f"Unique (x,t) pairs: {len(np.unique(coords_array, axis=0))}")
print(f"Total coords: {len(coords_array)}")
if len(np.unique(coords_array, axis=0)) != len(coords_array):
    print("WARNING: You have duplicate (x,t) coordinates!")
print("="*23)

# After: coords_array, tempValues = prepare_training_data(df)
print("\n" + "="*60)
print("DATA SPATIAL STRUCTURE ANALYSIS")
print("="*60)

unique_x = np.sort(np.unique(coords_array[:, 0]))
unique_t = np.sort(np.unique(coords_array[:, 1]))

print(f"Number of spatial points: {len(unique_x)}")
print(f"Number of time points: {len(unique_t)}")
print(f"Total data points: {len(coords_array)}")

# Check spatial variation at different times
for t_idx in [0, len(unique_t)//4, len(unique_t)//2, 3*len(unique_t)//4, -1]:
    t_val = unique_t[t_idx]
    mask = np.isclose(coords_array[:, 1], t_val, atol=1e-6)
    
    x_at_t = coords_array[mask, 0]
    temp_at_t = tempValues[mask]
    
    print(f"\nAt t_index={t_idx} (t={t_val:.4f}):")
    print(f"  Number of points at this time: {len(x_at_t)}")
    print(f"  Number of UNIQUE x values: {len(np.unique(x_at_t))}")
    print(f"  Temp range: [{temp_at_t.min():.6f}, {temp_at_t.max():.6f}]")
    print(f"  Temp std: {temp_at_t.std():.6e}")
    
    # If there are duplicates, something is wrong
    if len(x_at_t) != len(np.unique(x_at_t)):
        print(f"  WARNING: Duplicate x values at this timestep!")
        unique_x_here, counts = np.unique(x_at_t, return_counts=True)
        print(f"  Example: x={unique_x_here[0]:.6f} appears {counts[0]} times")
    
    # Only proceed if we have unique x values
    unique_x_at_t = np.unique(x_at_t)
    if len(unique_x_at_t) >= 3:
        # Get temperature at unique x locations (average if duplicates)
        temp_at_unique_x = []
        for x_val in unique_x_at_t:
            x_mask = np.isclose(x_at_t, x_val, atol=1e-10)
            temp_at_unique_x.append(np.mean(temp_at_t[x_mask]))
        temp_at_unique_x = np.array(temp_at_unique_x)
        
        print(f"  After removing duplicates: {len(temp_at_unique_x)} points")
        print(f"  Temp range (unique x): [{temp_at_unique_x.min():.6f}, {temp_at_unique_x.max():.6f}]")
        print(f"  Temp std (unique x): {temp_at_unique_x.std():.6e}")
        
        # Now compute derivatives
        if temp_at_unique_x.std() > 1e-10:
            first_deriv = np.diff(temp_at_unique_x)
            dx = np.mean(np.diff(unique_x_at_t))
            
            print(f"  Spatial spacing dx: {dx:.6e}")
            print(f"  Max |ΔT| between adjacent points: {np.max(np.abs(first_deriv)):.6e}")
            
            if len(first_deriv) >= 2:
                second_deriv = np.diff(first_deriv)
                approx_d2T_dx2 = second_deriv / (dx**2)
                print(f"  Mean |d²T/dx²| (approx): {np.mean(np.abs(approx_d2T_dx2)):.6e}")
                print(f"  Max |d²T/dx²| (approx): {np.max(np.abs(approx_d2T_dx2)):.6e}")
                
                if np.mean(np.abs(approx_d2T_dx2)) > 1e-6:
                    implied_lam = Q / np.mean(np.abs(approx_d2T_dx2))
                    print(f"  Implied λ from Q/|d²T/dx²|: {implied_lam:.3e}")
        else:
            print(f"  Temperature is spatially UNIFORM at this time")

print("="*60)

minX, maxX = coords_array[:, 0].min(), coords_array[:, 0].max()
minT, maxT = coords_array[:, 1].min(), coords_array[:, 1].max()
minTemp, maxTemp = tempValues.min(), tempValues.max()

coords_array[:, 0] = (2 * ((coords_array[:, 0] - minX) / (maxX - minX))) - 1
coords_array[:, 1] = (2 * ((coords_array[:, 1] - minT) / (maxT - minT))) - 1
tempValues = (2 * ((tempValues - minTemp) / (maxTemp - minTemp))) - 1

# --- All training data (for data loss only) ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
t_train_Nu = X_train_Nu_tensor[:, 1:2]

# --- Boundary points: x = 0 or x = max ---
eps = 1e-6

boundary_mask = (
    np.isclose(coords_array[:, 0], -1.0, atol=eps) | np.isclose(coords_array[:, 0], 1.0, atol=eps)
)

initial_mask = np.isclose(coords_array[:, 1], -1.0, atol=eps)

X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
t_train_boundary = X_train_boundary_tensor[:, 1:2]

X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
t_train_initial = X_train_initial_tensor[:, 1:2]

# === Generate Collocation Points ===
def generate_collocation_points(n_points=10000, method='random'):
    """
    Generate collocation points in the domain for PDE loss.
    
    Args:
        n_points: Number of collocation points
        method: 'random' for random sampling, 'grid' for uniform grid, 'lhs' for Latin Hypercube
    
    Returns:
        x_colloc, t_colloc: Tensors of collocation points (normalized to [-1, 1])
    """
    if method == 'random':
        # Random uniform sampling
        x_colloc = torch.rand(n_points, 1, device=device) * 2 - 1  # [-1, 1]
        t_colloc = torch.rand(n_points, 1, device=device) * 2 - 1  # [-1, 1]
    
    elif method == 'grid':
        # Uniform grid
        n_x = int(np.sqrt(n_points))
        n_t = n_points // n_x
        x_grid = torch.linspace(-1, 1, n_x, device=device)
        t_grid = torch.linspace(-1, 1, n_t, device=device)
        X_grid, T_grid = torch.meshgrid(x_grid, t_grid, indexing='ij')
        x_colloc = X_grid.reshape(-1, 1)
        t_colloc = T_grid.reshape(-1, 1)
    
    elif method == 'lhs':
        # Latin Hypercube Sampling (simple version)
        n_x = int(np.sqrt(n_points))
        n_t = n_points // n_x
        
        # Stratified sampling
        x_strata = torch.linspace(0, 1, n_x + 1, device=device)
        t_strata = torch.linspace(0, 1, n_t + 1, device=device)
        
        x_colloc = []
        t_colloc = []
        for i in range(n_x):
            for j in range(n_t):
                x_sample = x_strata[i] + torch.rand(1, device=device) * (x_strata[i+1] - x_strata[i])
                t_sample = t_strata[j] + torch.rand(1, device=device) * (t_strata[j+1] - t_strata[j])
                x_colloc.append(x_sample)
                t_colloc.append(t_sample)
        
        x_colloc = torch.stack(x_colloc) * 2 - 1  # Scale to [-1, 1]
        t_colloc = torch.stack(t_colloc) * 2 - 1
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return x_colloc, t_colloc


class PINN(nn.Module):
    """
    1D PINN with exactly ONE trainable physical parameter among: rho, cp, lam.
    The rest are fixed to provided true values.

    Stage 1 reparameterization:
      rho = exp(rho_hat), cp = exp(cp_hat), lam = exp(lam_hat)
    so learned parameter is unconstrained but physical parameter stays positive.

    learn_param: "rho" | "cp" | "lam" | "none"
    """
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        hidden_dim=100,
        num_hidden=3,
        activation="tanh",
        learn_param="rho",
        true_rho=DENSITY,
        true_cp=SPECIFIC_HEAT_CAPACITY,
        true_lam=THERMAL_CONDUCTIVITY,
        init_ranges=None,
        eps=1e-12,
    ):
        super().__init__()
        self.eps = eps

        self.learn_param = learn_param.lower().strip()
        if self.learn_param not in {"rho", "cp", "lam", "none"}:
            raise ValueError("learn_param must be one of: 'rho', 'cp', 'lam', 'none'")

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

        # --- Defaults for init ranges (PHYSICAL space) ---
        if init_ranges is None:
            init_ranges = {
                "rho": (0.5, 5.0),
                "cp":  (0.1, 2.5),
                "lam": (5.0, 15.0),
            }

        # --- Fixed TRUE values as buffers (physical space) ---
        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

        # --- Trainable log-parameters (hat variables). Only one is a Parameter. ---
        self.rho_hat = None
        self.cp_hat  = None
        self.lam_hat = None

        if self.learn_param != "none":
            lo, hi = init_ranges[self.learn_param]
            init_phys = (lo + (hi - lo) * torch.rand(1)).float()
            init_hat = torch.log(init_phys + self.eps)

            if self.learn_param == "rho":
                self.rho_hat = nn.Parameter(init_hat)
            elif self.learn_param == "cp":
                self.cp_hat = nn.Parameter(init_hat)
            elif self.learn_param == "lam":
                self.lam_hat = nn.Parameter(init_hat)

        self.epoch = 0

    # --- Physical parameters (always positive) ---
    @property
    def rho(self):
        if self.rho_hat is not None:
            return torch.exp(self.rho_hat)
        return self.rho_fixed

    @property
    def cp(self):
        if self.cp_hat is not None:
            return torch.exp(self.cp_hat)
        return self.cp_fixed

    @property
    def lam(self):
        if self.lam_hat is not None:
            return torch.exp(self.lam_hat)
        return self.lam_fixed

    def forward(self, x, t):
        out = torch.cat([x, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        self.epoch+=1
        return out

    def loss_PDE(self, x, t,epoch):
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, t)

        dx_factor = 2.0 / (maxX - minX)
        dt_factor = 2.0 / (maxT - minT)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = u_t * dt_factor * ((maxTemp - minTemp) / 2)
        u_xx = u_xx * (dx_factor ** 2) * ((maxTemp - minTemp) / 2)

        temporal_term = self.rho * self.cp * u_t
        diffusion_term = self.lam * u_xx
        source_term = Q
        
        residual = temporal_term - diffusion_term - source_term

        if epoch % 100 == 0:
            mean_temporal = torch.mean(temporal_term)
            mean_diffusion = torch.mean(diffusion_term)
            mean_residual = torch.mean(residual)
            print(f"\n  === PDE Balance ===")
            print(f"  ρ·cp·∂u/∂t = {mean_temporal:.3f}")
            print(f"  λ·∂²u/∂x² = {mean_diffusion:.3f}")
            print(f"  Q = {Q:.3f}")
            print(f"  Residual = {mean_residual:.3f}")
            print(f"  λ = {float(self.lam):.3f}")
        
        # CRITICAL: Normalize each term by its expected magnitude
        # This prevents the network from exploiting scale differences
        temporal_scale = abs(Q) + 0.1  # Q dominates, so scale by Q
        diffusion_scale = abs(Q) + 0.1
        
        normalized_residual = residual / temporal_scale
        
        return torch.mean(normalized_residual ** 2)

    def loss_initial(self, x, t):
        u = self.forward(x, t)
        u = 0.5 * (u + 1) * (maxTemp - minTemp) + minTemp
        return torch.mean((u - INITIAL_TEMP) ** 2)

    def loss_bounds(self, x, t):
        x = x.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_x = ((maxTemp - minTemp) / 2) * u_x * (2.0 / (maxX - minX))

        return torch.mean(u_x ** 2)

    def loss_data(self, x, t, u_obs):
        u = self.forward(x, t)
        return torch.mean((u - u_obs) ** 2)


    # Update losses():
    def losses(self, x_colloc, t_colloc, x_data, t_data, u_data, x0, t0, xb, tb,it):
        Lr = self.loss_PDE(x_colloc, t_colloc,it)
        Li = self.loss_initial(x0, t0)
        Lb = self.loss_bounds(xb, tb)
        Ld = self.loss_data(x_data, t_data, u_data)
        return Lr, Li, Lb, Ld



def main(param_to_learn, n_colloc=10000, colloc_method='random'):
    torch.manual_seed(seeds_num)



    model = PINN(
        learn_param=param_to_learn,
        input_dim=2,
        output_dim=1,
        hidden_dim=100,
        num_hidden=3,
        activation="tanh"
    ).to(device)

    lambda_d, lambda_r, lambda_b, lambda_i = 1.0, 1.0, 1.0, 1.0

    if param_to_learn == "lam":
        lambda_r = 10.0  # Increase PDE weight

    true_vals = {
        "rho": float(DENSITY),
        "cp": float(SPECIFIC_HEAT_CAPACITY),
        "lam": float(THERMAL_CONDUCTIVITY),
    }

    history = {
        "L_total": [], "L_pde": [], "L_ic": [], "L_bc": [], "L_data": [],
        "rho": [], "cp": [], "lam": [],
        "time_sec": [],
    }

    t_start = default_timer()

    def get_param_val():
        v = getattr(model, param_to_learn)
        return float(v.detach().cpu().item())

    def compute_adaptive_lambdas(Lr, Li, Lb, Ld, include_hat_params=False):
        losses = {"data": Ld, "pde": Lr, "bc": Lb, "ic": Li}

        if include_hat_params:
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            params = [p for (n, p) in model.named_parameters()
                      if p.requires_grad and not n.endswith("_hat")]

        grads = {}
        for name, term in losses.items():
            g = torch.autograd.grad(
                term, params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            gn = torch.zeros((), device=device)
            cnt = 0
            for gi in g:
                if gi is not None:
                    gn = gn + gi.norm()
                    cnt += 1
            grads[name] = gn / max(cnt, 1)

        total = sum(grads.values())
        lambdas = {k: total / (grads[k] + 1e-8) for k in grads}
        Z = sum(lambdas.values()) + 1e-12
        lambdas = {k: (v / Z) for k, v in lambdas.items()}
        return lambdas

    lbfgs_iters = 5000
    log_every = 10

    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=1,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    best_loss = float("inf")
    loss_min_delta = 1e-10
    loss_patience = 200
    loss_counter = 0

    param_min_delta = 1e-8
    param_patience = 40
    param_counter = 0
    prev_param = None

    current_lambdas = {"pde": 0.25, "ic": 0.25, "bc": 0.25, "data": 0.25}

    # Generate collocation points once (or regenerate periodically)
    print(f"Generating {n_colloc} collocation points using '{colloc_method}' method...")
    x_colloc, t_colloc = generate_collocation_points(n_colloc, method=colloc_method)
    print(f"Collocation points shape: x={x_colloc.shape}, t={t_colloc.shape}")

    if param_to_learn == "lam":
        lambda_r = 1000.0  # Make PDE loss dominant
        lambda_d = 1.0
    else:
        lambda_r = 1.0
        lambda_d = 1.0

    def closure():
        lbfgs_opt.zero_grad()

        Lr, Li, Lb, Ld = model.losses(
            x_colloc, t_colloc,  # Collocation points for PDE
            x_train_Nu, t_train_Nu, U_train_Nu,  # Data points for fitting
            x_train_initial, t_train_initial,  # Initial condition
            x_train_boundary, t_train_boundary  # Boundary condition
            ,it
        )

        L = (
            lambda_r * current_lambdas["pde"]  * Lr +
            lambda_i * current_lambdas["ic"]   * Li +
            lambda_b * current_lambdas["bc"]   * Lb +
            lambda_d * current_lambdas["data"] * Ld 
        )

        L.backward()
        return L


    for it in range(lbfgs_iters):
        # Optionally regenerate collocation points every N iterations for better coverage
        if it > 0 and it % 500 == 0:
            print(f"[iter {it}] Regenerating collocation points...")
            x_colloc, t_colloc = generate_collocation_points(n_colloc, method=colloc_method)

        with torch.enable_grad():
            if param_to_learn != "lam":
                Lr, Li, Lb, Ld = model.losses(
                    x_colloc, t_colloc,
                    x_train_Nu, t_train_Nu, U_train_Nu,
                    x_train_initial, t_train_initial,
                    x_train_boundary, t_train_boundary,-1
                )
                current_lambdas = compute_adaptive_lambdas(Lr, Li, Lb, Ld, include_hat_params=False)
            else:
                # Fixed weights that heavily favor PDE
                current_lambdas = {"pde": 0.90, "ic": 0.03, "bc": 0.03, "data": 0.04}

        L = lbfgs_opt.step(closure)
        curr = float(L.detach())

        elapsed = default_timer() - t_start
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["time_sec"].append(elapsed)

        if it % log_every == 0 or it == lbfgs_iters - 1:

            pv = get_param_val()

            history["L_total"].append(curr)

            if curr < best_loss - loss_min_delta:
                best_loss = curr
                loss_counter = 0
            else:
                loss_counter += 1

            if prev_param is None:
                prev_param = pv
                param_counter = 0
            else:
                if abs(pv - prev_param) < param_min_delta:
                    param_counter += 1
                else:
                    param_counter = 0
                prev_param = pv

            print(
                f"[LBFGS {it:05d}] L={curr:.3e} "
                f"{param_to_learn}={pv:.6f} p_streak={param_counter}/{param_patience} | "
                f"best={best_loss:.3e} l_streak={loss_counter}/{loss_patience} | "
                f"lams=({current_lambdas['pde']:.2f},{current_lambdas['ic']:.2f},{current_lambdas['bc']:.2f},{current_lambdas['data']:.2f})"
            )

            if param_counter >= param_patience:
                print(f"[LBFGS] Early stopping: {param_to_learn} converged.")
                break

            if loss_counter >= loss_patience:
                print(f"[LBFGS] Early stopping: loss plateau.")
                break
            # After first loss computation in training loop
            if it == 0:
                print("\n" + "=" * 50)
                print("IDENTIFIABILITY CHECK")
                print("=" * 50)
                
                # Try different lambda values and see how much loss changes
                original_lam = float(model.lam.detach())
                test_lams = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]
                
                # Temporarily disable gradients for the parameter
                if model.lam_hat is not None:
                    original_requires_grad = model.lam_hat.requires_grad
                    model.lam_hat.requires_grad_(False)
                    original_value = model.lam_hat.clone()
                
                for test_lam in test_lams:
                    if model.lam_hat is not None:
                        model.lam_hat.data = torch.log(torch.tensor([test_lam], device=device, dtype=torch.float32))
                    
                    Lr_test, Li_test, Lb_test, Ld_test = model.losses(
                        x_colloc, t_colloc,
                        x_train_Nu, t_train_Nu, U_train_Nu,
                        x_train_initial, t_train_initial,
                        x_train_boundary, t_train_boundary,-1
                    )
                    total_test = float(Lr_test + Li_test + Lb_test + Ld_test)
                    
                    print(f"  λ={test_lam:6.2f} → Loss={total_test:.3e} (PDE={float(Lr_test):.3e}, Data={float(Ld_test):.3e})")
                
                # Restore original value and gradient state
                if model.lam_hat is not None:
                    model.lam_hat.data = original_value.data
                    model.lam_hat.requires_grad_(original_requires_grad)
                
                print("=" * 50 + "\n")

    t_total = default_timer() - t_start
    print(f"Total training time: {t_total/60:.2f} min")
    print(f"wall-clock time (sec): {t_total:.2f}")

    plt.figure(figsize=(7, 4))
    plt.plot(history["time_sec"], history[param_to_learn], label=f"Estimated {param_to_learn}", linewidth=2)
    plt.axhline(y=true_vals[param_to_learn], color="black", linestyle="--", linewidth=2, label=f"True {param_to_learn}")
    plt.xlabel("Time (sec)")
    plt.ylabel(param_to_learn)
    plt.title(f"Convergence of {param_to_learn}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = f"convergence_{param_to_learn}_colloc.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved parameter convergence plot to: {save_path}")
    plt.show()


# Run with collocation points
for param in ["lam"]:
    main(param, n_colloc=10000, colloc_method='random')