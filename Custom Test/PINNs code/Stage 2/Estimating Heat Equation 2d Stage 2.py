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


def load_data(filepath="temperature_output copy.csv"):
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

# --- No normalization ---

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

k = np.random.rand()

class PINN(nn.Module):
    """
    PINN with exactly ONE trainable physical parameter among: rho, cp, lam.
    The rest are fixed to provided true values.

    Stage 1 reparameterization:
      rho = exp(rho_hat), cp = exp(cp_hat), lam = exp(lam_hat)
    so learned parameter is unconstrained but physical parameter stays positive.

    learn_param: "rho" | "cp" | "lam" | "none"
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
        init_ranges=None,  # ranges in PHYSICAL space, e.g. {"rho":(0.5,5.0), ...}
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
                "lam": (0.1, 50.0),
            }

        # --- Fixed TRUE values as buffers (physical space) ---
        # We store them as positive scalars.
        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

        # --- Trainable log-parameters (hat variables). Only one is a Parameter. ---
        self.rho_hat = None
        self.cp_hat  = None
        self.lam_hat = None

        if self.learn_param != "none":
            lo, hi = init_ranges[self.learn_param]
            # init in physical space, then convert to log-space
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

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    # PDE: rho*cp*u_t - lam*(u_xx + u_yy) - Q = 0
    def loss_PDE(self, x, y, t):
        # Make sure these are leaf tensors requiring grad (robust for LBFGS)
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

        u_t = u_t * dt_factor
        u_xx = u_xx * (dx_factor ** 2)
        u_yy = u_yy * (dy_factor ** 2)

        # Scale temperature derivative back to physical units if temp is normalized
        u_t = u_t * ((maxTemp-minTemp) / 2)
        u_xx = u_xx * ((maxTemp-minTemp) / 2)
        u_yy = u_yy * ((maxTemp-minTemp) / 2)

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

        u_x = ((maxTemp-minTemp)/2) *  u_x * (2.0 / (maxX-minX))
        u_y = ((maxTemp-minTemp)/2) * u_y * (2.0 / (maxY-minY))

        return torch.mean(u_x ** 2) + torch.mean(u_y ** 2)

    def loss_data(self, x, y, t, u_obs):
        u = self.forward(x, y, t)
        return torch.mean((u - u_obs) ** 2)

    def losses(self, x_all, y_all, t_all, u_all, x0, y0, t0, xb, yb, tb):
        Lr = self.loss_PDE(x_all, y_all, t_all)
        Li = self.loss_initial(x0, y0, t0)
        Lb = self.loss_bounds(xb, yb, tb)
        Ld = self.loss_data(x_all, y_all, t_all, u_all)
        return Lr, Li, Lb, Ld


def main(param_to_learn):
    # ---------------------------
    # Stage 0 training setup
    # ---------------------------
    torch.manual_seed(seeds_num)

    model = PINN(
        learn_param=param_to_learn,
        input_dim=3,
        output_dim=1,
        hidden_dim=100,
        num_hidden=3,
        activation='tanh'
    ).to(device)

    # Fixed lambda weights (Stage 0 control)
    lambda_d, lambda_r, lambda_b, lambda_i = 1.0, 1.0, 1.0, 1.0

    history = {
        "L_total": [], "L_pde": [], "L_ic": [], "L_bc": [], "L_data": [],
        "rho": [], "cp": [], "lam": [],
        "pde_residual_rmse": [], "field_l2": [], "time_sec": [],
    }

    t_start = default_timer()

    # ---------------------------
    # Param-convergence early stopping config
    # ---------------------------
    # Stop if the learned parameter changes by less than param_min_delta
    # for param_patience "logging checks" in a row.
    param_patience = 30              # 30 * 100 = 3000 Adam steps of "no movement"
    param_min_delta = 1e-6           # absolute change threshold in parameter value
    param_counter = 0
    prev_param_val = None

    def get_learned_scalar_value():
        # model.rho / model.cp / model.lam are properties -> always return a scalar tensor
        v = getattr(model, param_to_learn)  # tensor shape [1]
        return float(v.detach().cpu().item())

    # ---------------------------
    # Phase 1: Adam (full loss) + Early Stopping (loss + param convergence)
    # ---------------------------
    adam_iters = 20000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Loss early stopping config
    early_stop_patience = 2000
    min_delta = 1e-8
    check_every = 10

    best_loss = float("inf")
    best_state = None
    patience_counter = 0


    # ---------------------------
    # Phase 2: L-BFGS + Early Stopping (loss + param convergence)
    # ---------------------------
    lbfgs_iters = 2000
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=1,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    lbfgs_patience = 200
    lbfgs_min_delta = 1e-10
    best_lbfgs_loss = float("inf")
    lbfgs_counter = 0

    # Param ES for LBFGS (checks happen every log interval)
    lbfgs_log_every = 10
    lbfgs_param_patience = 40         # 40 logs * 10 iters/log = 400 LBFGS iterations "no movement"
    lbfgs_param_min_delta = 1e-8
    lbfgs_param_counter = 0
    lbfgs_prev_param_val = None

    def closure():
        optimizer.zero_grad()
        Lr, Li, Lb, Ld = model.losses(
            x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
            x_train_initial, y_train_initial, t_train_initial,
            x_train_boundary, y_train_boundary, t_train_boundary
        )
        L = (lambda_r * Lr) + (lambda_i * Li) + (lambda_b * Lb) + (lambda_d * Ld)
        L.backward()
        return L

    for it in range(lbfgs_iters):
        L = optimizer.step(closure)
        curr = float(L.detach())

        # Loss-based LBFGS early stopping
        if curr < best_lbfgs_loss - lbfgs_min_delta:
            best_lbfgs_loss = curr
            lbfgs_counter = 0
        else:
            lbfgs_counter += 1

        Lr, Li, Lb, Ld = model.losses(
            x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
            x_train_initial, y_train_initial, t_train_initial,
            x_train_boundary, y_train_boundary, t_train_boundary
        )

        # param convergence (LBFGS)
        param_val = get_learned_scalar_value()
        if lbfgs_prev_param_val is None:
            lbfgs_prev_param_val = param_val
            lbfgs_param_counter = 0
        else:
            if abs(param_val - lbfgs_prev_param_val) < lbfgs_param_min_delta:
                lbfgs_param_counter += 1
            else:
                lbfgs_param_counter = 0
            lbfgs_prev_param_val = param_val

        elapsed = default_timer() - t_start
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["time_sec"].append(elapsed)

        # Logging + param convergence checks
        if it % lbfgs_log_every == 0 or it == lbfgs_iters - 1:
            print(
                f"[LBFGS {it:04d}] L={curr:.3e} "
                f"(Ld={float(Ld):.3e}, Lr={float(Lr):.3e}, Lb={float(Lb):.3e}, Li={float(Li):.3e}) | "
                f"rho={float(model.rho):.6f} cp={float(model.cp):.6f} lam={float(model.lam):.6f} | "
                f"{param_to_learn}={param_val:.6f} Δ<{lbfgs_param_min_delta:g}? streak={lbfgs_param_counter}/{lbfgs_param_patience} | "
                f"best={best_lbfgs_loss:.3e} loss_patience={lbfgs_counter}/{lbfgs_patience}"
            )

        if lbfgs_param_counter >= lbfgs_param_patience:
            print(
                f"[L-BFGS] Early stopping (param converged) at iter={it}: "
                f"{param_to_learn} changed < {lbfgs_param_min_delta:g} for {lbfgs_param_patience} logs."
            )
            break

        if lbfgs_counter >= lbfgs_patience:
            print(f"[L-BFGS] Early stopping (loss) at iter={it} best_loss={best_lbfgs_loss:.3e}")
            break


    t_total = default_timer() - t_start
    print(f"Total training time: {t_total/60:.2f} min")

    print(f"wall-clock time (sec): {t_total:.2f}")

    plt.figure(figsize=(7, 4))

    # Learned trajectory
    plt.plot(
        history["time_sec"],
        history[param_to_learn],
        label=f"Estimated {param_to_learn}",
        linewidth=2
    )

    # True value (horizontal dashed line)
    plt.axhline(
        y=true_values[param_to_learn],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"True {param_to_learn}"
    )

    plt.xlabel("Time (sec)")
    plt.ylabel(param_to_learn)
    plt.title(f"Convergence of {param_to_learn}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # File name: clean, reproducible, paper-friendly
    save_path = f"convergence_{param_to_learn}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved parameter convergence plot to: {save_path}")

    plt.show()

for param in ["rho","cp","lam"]:
    main(param)