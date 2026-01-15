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


def load_data(filepath="temperature_output copy 2.csv"):
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

minX,maxX = coords_array[:,0].min(),coords_array[:,0].max()
minY,maxY = coords_array[:,1].min(),coords_array[:,1].max()
minT,maxT = coords_array[:,2].min(),coords_array[:,2].max()
minTemp,maxTemp = tempValues.min(),tempValues.max()

coords_array[:,0] = (2 * ((coords_array[:,0]-minX) / (maxX - minX))) - 1
coords_array[:,1] = (2 * ((coords_array[:,1]-minY) / (maxY - minY))) - 1
coords_array[:,2] = (2 * ((coords_array[:,2]-minT) / (maxT - minT))) - 1
tempValues =  (2 * ((tempValues-minTemp) / (maxTemp - minTemp))) - 1

# --- All training data ---
X_train_Nu_tensor = torch.from_numpy(coords_array).float().to(device)
U_train_Nu = torch.from_numpy(tempValues).float().to(device)
x_train_Nu = X_train_Nu_tensor[:, 0:1]
y_train_Nu = X_train_Nu_tensor[:, 1:2]
t_train_Nu = X_train_Nu_tensor[:, 2:3]


# --- Boundary points: x = 0 or x = max, y = 0 or y = max ---
eps = 1e-6

boundary_mask = (
    np.isclose(coords_array[:, 0], -1.0, atol=eps) | np.isclose(coords_array[:, 0], 1.0, atol=eps) |
    np.isclose(coords_array[:, 1], -1.0, atol=eps) | np.isclose(coords_array[:, 1], 1.0, atol=eps)
)

initial_mask = np.isclose(coords_array[:, 2], -1.0, atol=eps)

X_train_boundary_tensor = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
U_train_boundary = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
x_train_boundary = X_train_boundary_tensor[:, 0:1]
y_train_boundary = X_train_boundary_tensor[:, 1:2]
t_train_boundary = X_train_boundary_tensor[:, 2:3]

X_train_initial_tensor = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U_train_initial = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x_train_initial = X_train_initial_tensor[:, 0:1]
y_train_initial = X_train_initial_tensor[:, 1:2]
t_train_initial = X_train_initial_tensor[:, 2:3]

print(f"Domain: x=[{minX}, {maxX}], y=[{minY}, {maxY}], t=[{minT}, {maxT}]")
print(f"Temp range: [{minTemp}, {maxTemp}]")
print(f"dx_factor={2.0/(maxX-minX)}, dy_factor={2.0/(maxY-minY)}, dt_factor={2.0/(maxT-minT)}")

N_COLLOCATION = 10000

def generate_collocation_points(n_points, device='cuda'):
    """
    Generate random collocation points in normalized [-1, 1] domain.
    """
    x_coll = 2 * torch.rand(n_points, 1, device=device) - 1
    y_coll = 2 * torch.rand(n_points, 1, device=device) - 1
    t_coll = 2 * torch.rand(n_points, 1, device=device) - 1
    
    return x_coll, y_coll, t_coll

x_coll, y_coll, t_coll = generate_collocation_points(
    n_points=N_COLLOCATION,
    device=device
)


class PINN(nn.Module):
    """
    PINN with exactly ONE trainable physical parameter among: rho, cp, lam.
    """
    def __init__(
        self,
        input_dim=3,
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

        if init_ranges is None:
            init_ranges = {
                "rho": (0.5, 5.0),
                "cp":  (0.1, 2.5),
                "lam": (0.1, 50.0),
            }

        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

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

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, y, t):
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, y, t)

        dx_factor = 2.0 / (maxX-minX)
        dy_factor = 2.0 / (maxY-minY)
        dt_factor = 2.0 / (maxT-minT)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        u_t = u_t * dt_factor * ((maxTemp-minTemp) / 2)
        u_xx = u_xx * (dx_factor ** 2) * ((maxTemp-minTemp) / 2)
        u_yy = u_yy * (dy_factor ** 2) * ((maxTemp-minTemp) / 2)

        residual = (self.rho * self.cp * u_t) - (self.lam * (u_xx + u_yy)) - Q
        return torch.mean(residual ** 2)

    def loss_initial(self, x, y, t):
        u = self.forward(x, y, t)
        x_denorm = (((x+1)/2)*(maxX-minX))+minX 
        y_denorm = (((y+1)/2)*(maxY-minY))+minY 
        u = 0.5 * (u + 1) * (maxTemp-minTemp) + minTemp
        return torch.mean((u - (20 + (x_denorm ** 2) + (y_denorm ** 2)))**2)

    def loss_bounds(self, x, y, t):
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        u_x = ((maxTemp-minTemp)/2) * u_x * (2.0 / (maxX-minX))
        u_y = ((maxTemp-minTemp)/2) * u_y * (2.0 / (maxY-minY))

        return torch.mean(u_x ** 2) + torch.mean(u_y ** 2)

    def loss_data(self, x, y, t, u_obs):
        u = self.forward(x, y, t)
        return torch.mean((u - u_obs) ** 2)

    def losses(self, x_data, y_data, t_data, u_data, 
               x_ic, y_ic, t_ic, 
               x_bc, y_bc, t_bc,
               x_coll, y_coll, t_coll):
        Lr = self.loss_PDE(x_coll, y_coll, t_coll)
        Li = self.loss_initial(x_ic, y_ic, t_ic)
        Lb = self.loss_bounds(x_bc, y_bc, t_bc)
        Ld = self.loss_data(x_data, y_data, t_data, u_data)
        return Lr, Li, Lb, Ld


def resample_collocation_points():
    return generate_collocation_points(
        n_points=N_COLLOCATION,
        device=device
    )


def main(param_to_learn):
    global x_coll, y_coll, t_coll
    
    torch.manual_seed(seeds_num)

    model = PINN(
        learn_param=param_to_learn,
        input_dim=3,
        output_dim=1,
        hidden_dim=128,
        num_hidden=4,
        activation="tanh"
    ).to(device)

    lambda_d, lambda_r, lambda_b, lambda_i = 1.0, 1.0, 1.0, 1.0

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

    # ---------------------------
    # Training hyperparameters
    # ---------------------------
    phase1_iters = 1000   # Data-only pre-training
    phase2_iters = 20000   # Combined training with Adam
    phase3_iters = 2000   # L-BFGS refinement
    log_every = 10
    resample_every = 500

    current_lambdas = {"pde": 0.25, "ic": 0.25, "bc": 0.25, "data": 0.25}

    best_loss = float("inf")
    loss_min_delta = 1e-10
    loss_patience = 2000
    loss_counter = 0
    param_min_delta = 1e-8
    param_patience = 40
    param_counter = 0
    prev_param = None

    # ==========================================================================
    # PHASE 1: Data-Only Pre-training (Network parameters only, NO physics)
    # ==========================================================================
    print("=" * 60)
    print("PHASE 1: Data-Only Pre-training (Network only)")
    print("=" * 60)
    print("Training network to fit data WITHOUT physics loss...")
    print("Physical parameter is FROZEN during this phase.")

    # Only optimize network parameters, NOT the physical parameter
    network_params = [p for (n, p) in model.named_parameters() 
                      if p.requires_grad and not n.endswith("_hat")]
    
    adam_opt_phase1 = torch.optim.Adam(network_params, lr=1e-3)

    for it in range(phase1_iters):
        adam_opt_phase1.zero_grad()

        # Only data loss (+ IC and BC for good initialization)
        Ld = model.loss_data(x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu)
        Li = model.loss_initial(x_train_initial, y_train_initial, t_train_initial)
        Lb = model.loss_bounds(x_train_boundary, y_train_boundary, t_train_boundary)
        
        # Combined loss without PDE
        L = Ld + 0.1 * Li + 0.1 * Lb

        L.backward()
        adam_opt_phase1.step()

        curr = float(L.detach())

        if it % log_every == 0:
            elapsed = default_timer() - t_start
            history["rho"].append(float(model.rho.detach()))
            history["cp"].append(float(model.cp.detach()))
            history["lam"].append(float(model.lam.detach()))
            history["time_sec"].append(elapsed)

            pv = get_param_val()
            history["L_total"].append(curr)
            history["L_pde"].append(0.0)  # No PDE loss in phase 1
            history["L_ic"].append(float(Li.detach()))
            history["L_bc"].append(float(Lb.detach()))
            history["L_data"].append(float(Ld.detach()))

            current_lr = adam_opt_phase1.param_groups[0]['lr']
            print(
                f"[Phase1 {it:05d}] L={curr:.3e} "
                f"(Ld={float(Ld):.3e}, Li={float(Li):.3e}, Lb={float(Lb):.3e}) | "
                f"{param_to_learn}={pv:.6f} (frozen) | "
                f"lr={current_lr:.2e}"
            )

    print(f"\nPhase 1 complete. Network pre-trained on data.")
    print(f"Data loss: {float(Ld):.6e}")

    # ==========================================================================
    # PHASE 2: Combined Training (Network + Physical Parameter + Physics Loss)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Combined Training (Network + Physics + Parameter)")
    print("=" * 60)
    print("Now training BOTH network AND physical parameter with ALL losses...")

    # Optimize ALL parameters (network + physical)
    adam_opt_phase2 = torch.optim.Adam(model.parameters(), lr=1e-2)
    # Reset early stopping counters
    best_loss = float("inf")
    loss_counter = 0
    param_counter = 0
    prev_param = None

    for it in range(phase2_iters):
        # Resample collocation points periodically
        if it > 0 and it % resample_every == 0:
            x_coll, y_coll, t_coll = resample_collocation_points()
            print(f"  [Resampled collocation points at iteration {it}]")

        adam_opt_phase2.zero_grad()

        Lr, Li, Lb, Ld = model.losses(
            x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
            x_train_initial, y_train_initial, t_train_initial,
            x_train_boundary, y_train_boundary, t_train_boundary,
            x_coll, y_coll, t_coll
        )

        current_lambdas = compute_adaptive_lambdas(Lr, Li, Lb, Ld, include_hat_params=False)

        L = (
            lambda_r * current_lambdas["pde"] * Lr +
            lambda_i * current_lambdas["ic"] * Li +
            lambda_b * current_lambdas["bc"] * Lb +
            lambda_d * current_lambdas["data"] * Ld
        )

        L.backward()
        adam_opt_phase2.step()

        curr = float(L.detach())

        if it % log_every == 0:
            elapsed = default_timer() - t_start
            history["rho"].append(float(model.rho.detach()))
            history["cp"].append(float(model.cp.detach()))
            history["lam"].append(float(model.lam.detach()))
            history["time_sec"].append(elapsed)

            pv = get_param_val()
            history["L_total"].append(curr)
            history["L_pde"].append(float(Lr.detach()))
            history["L_ic"].append(float(Li.detach()))
            history["L_bc"].append(float(Lb.detach()))
            history["L_data"].append(float(Ld.detach()))

            current_lr = adam_opt_phase2.param_groups[0]['lr']
            print(
                f"[Phase2 {it:05d}] L={curr:.3e} "
                f"(Ld={float(Ld):.3e}, Lr={float(Lr):.3e}, Lb={float(Lb):.3e}, Li={float(Li):.3e}) | "
                f"{param_to_learn}={pv:.6f} p_streak={param_counter}/{param_patience} | "
                f"best={best_loss:.3e} l_streak={loss_counter}/{loss_patience} | "
                f"lams=({current_lambdas['pde']:.2f},{current_lambdas['ic']:.2f},{current_lambdas['bc']:.2f},{current_lambdas['data']:.2f}) | "
                f"lr={current_lr:.2e}"
            )

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

            if param_counter >= param_patience:
                print(f"[Phase2] Early stopping: {param_to_learn} converged.")
                break

            if loss_counter >= loss_patience:
                print(f"[Phase2] Early stopping: loss plateau.")
                break

    # ---------------------------
    # Results and Plotting
    # ---------------------------
    t_total = default_timer() - t_start
    print(f"\nTotal training time: {t_total/60:.2f} min")
    print(f"wall-clock time (sec): {t_total:.2f}")

    final_param = get_param_val()
    true_val = true_vals[param_to_learn]
    rel_error = abs(final_param - true_val) / true_val * 100
    print(f"Final {param_to_learn}: {final_param:.6f} (true: {true_val:.6f}, error: {rel_error:.2f}%)")

    # Plot convergence
    plt.figure(figsize=(10, 5))
    
    # Add vertical lines for phase transitions
    phase1_end = phase1_iters // log_every
    phase2_end = phase1_end + (phase2_iters // log_every)
    
    plt.subplot(1, 2, 1)
    plt.plot(history["time_sec"], history[param_to_learn], label=f"Estimated {param_to_learn}", linewidth=2)
    plt.axhline(y=true_vals[param_to_learn], color="black", linestyle="--", linewidth=2, label=f"True {param_to_learn}")
    plt.xlabel("Time (sec)")
    plt.ylabel(param_to_learn)
    plt.title(f"Convergence of {param_to_learn}")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(history["time_sec"], history["L_total"], label="Total Loss")
    plt.semilogy(history["time_sec"], history["L_data"], label="Data Loss", alpha=0.7)
    plt.semilogy(history["time_sec"], history["L_pde"], label="PDE Loss", alpha=0.7)
    plt.xlabel("Time (sec)")
    plt.ylabel("Loss")
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = f"convergence_{param_to_learn}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved parameter convergence plot to: {save_path}")
    plt.show()


for param in ["rho","cp","lam"]:
    main(param)