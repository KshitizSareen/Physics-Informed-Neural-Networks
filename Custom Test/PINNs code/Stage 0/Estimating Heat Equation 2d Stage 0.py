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
    """
    PINN with exactly ONE trainable physical parameter among: rho, cp, lam.
    The rest are fixed to provided true values.

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
        # optional: random init range for the learned parameter
        init_ranges=None,  # e.g. {"rho": (0.5,5.0), "cp": (0.1,2.5), "lam": (0.1,50.0)}
    ):
        super().__init__()

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

        # --- Defaults for init ranges ---
        if init_ranges is None:
            init_ranges = {
                "rho": (0.5, 5.0),
                "cp":  (0.1, 2.5),
                "lam": (0.1, 50.0),
            }

        # --- Fixed values as buffers (non-trainable, device-aware) ---
        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

        # --- Create exactly one trainable Parameter (or none) ---
        self.rho_param = None
        self.cp_param  = None
        self.lam_param = None

        if self.learn_param != "none":
            lo, hi = init_ranges[self.learn_param]
            init_val = lo + (hi - lo) * torch.rand(1)
            init_val = init_val.float()

            if self.learn_param == "rho":
                self.rho_param = nn.Parameter(init_val)
            elif self.learn_param == "cp":
                self.cp_param = nn.Parameter(init_val)
            elif self.learn_param == "lam":
                self.lam_param = nn.Parameter(init_val)

        self.epoch = 0

    # --- Convenient properties to use in PDE ---
    @property
    def rho(self):
        return self.rho_param if self.rho_param is not None else self.rho_fixed

    @property
    def cp(self):
        return self.cp_param if self.cp_param is not None else self.cp_fixed

    @property
    def lam(self):
        return self.lam_param if self.lam_param is not None else self.lam_fixed

    def forward(self, x, y, t):
        out = torch.cat([x, y, t], dim=-1)
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    # PDE: rho*cp*u_t - lam*(u_xx + u_yy) - Q = 0
    def loss_PDE(self, x, y, t):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        residual = (self.rho * self.cp * u_t) - (self.lam * (u_xx + u_yy)) - Q
        return torch.mean(residual ** 2)

    def loss_initial(self, x, y, t):
        u = self.forward(x, y, t)
        return torch.mean((u - INITIAL_TEMP) ** 2)

    def loss_bounds(self, x, y, t):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

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
    
# Metrics helpers
def pde_residual_rmse(model: PINN, x, y, t):
    # RMSE of residual (not squared mean)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model.forward(x, y, t)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    r = (model.rho * model.cp * u_t) - (model.lam * (u_xx + u_yy)) - Q
    return torch.sqrt(torch.mean(r ** 2)).detach()

def field_l2_error(model: PINN, x, y, t, u_obs):
    # L2 relative error on the observed points
    with torch.no_grad():
        u_pred = model.forward(x, y, t)
        num = torch.norm(u_pred - u_obs)
        den = torch.norm(u_obs) + 1e-12
        return (num / den).detach()

def rel_err(est, true):
    return abs(float(est) - float(true)) / (abs(float(true)) + 1e-12)


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
    param_check_every = 100          # same cadence as loss logging
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
    check_every = 100

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for it in range(adam_iters):
        optimizer.zero_grad()

        Lr, Li, Lb, Ld = model.losses(
            x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
            x_train_initial, y_train_initial, t_train_initial,
            x_train_boundary, y_train_boundary, t_train_boundary
        )

        L = (lambda_r * Lr) + (lambda_i * Li) + (lambda_b * Lb) + (lambda_d * Ld)
        L.backward()
        optimizer.step()

        curr = float(L.detach())

        # ----- Loss-based early stopping -----
        if curr < best_loss - min_delta:
            best_loss = curr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += check_every

        # ----- Param-convergence early stopping -----
        # check if parameter stopped moving
        param_val = get_learned_scalar_value()
        if prev_param_val is None:
            prev_param_val = param_val
            param_counter = 0
        else:
            if abs(param_val - prev_param_val) < param_min_delta:
                param_counter += 1
            else:
                param_counter = 0
            prev_param_val = param_val

        # ----- Logging -----
        elapsed = default_timer() - t_start
        history["L_total"].append(curr)
        history["L_pde"].append(float(Lr.detach()))
        history["L_ic"].append(float(Li.detach()))
        history["L_bc"].append(float(Lb.detach()))
        history["L_data"].append(float(Ld.detach()))
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["pde_residual_rmse"].append(float(pde_residual_rmse(model, x_train_Nu, y_train_Nu, t_train_Nu)))
        history["field_l2"].append(float(field_l2_error(model, x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu)))
        history["time_sec"].append(elapsed)

        # Evaluate periodically (loss ES + param-convergence ES)
        if (it % check_every == 0) or (it == adam_iters - 1):
            print(
                f"[Adam {it:06d}] L={curr:.3e} "
                f"(Ld={float(Ld):.3e}, Lr={float(Lr):.3e}, Lb={float(Lb):.3e}, Li={float(Li):.3e}) | "
                f"rho={float(model.rho):.6f} cp={float(model.cp):.6f} lam={float(model.lam):.6f} | "
                f"{param_to_learn}={param_val:.6f} Δ<{param_min_delta:g}? streak={param_counter}/{param_patience} | "
                f"best={best_loss:.3e} loss_patience={patience_counter}/{early_stop_patience}"
            )

        # ----- Stop conditions -----
        if patience_counter >= early_stop_patience:
            print(f"[Adam] Early stopping (loss) at iter={it} best_loss={best_loss:.3e}")
            break

        if param_counter >= param_patience:
            print(
                f"[Adam] Early stopping (param converged) at iter={it}: "
                f"{param_to_learn} changed < {param_min_delta:g} for {param_patience} checks "
                f"({param_patience*check_every} steps)."
            )
            break

    # Restore best Adam model before L-BFGS
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Reset param convergence tracker for LBFGS
    param_counter = 0
    prev_param_val = None

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
        history["L_total"].append(curr)
        history["L_pde"].append(float(Lr.detach()))
        history["L_ic"].append(float(Li.detach()))
        history["L_bc"].append(float(Lb.detach()))
        history["L_data"].append(float(Ld.detach()))
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["pde_residual_rmse"].append(float(pde_residual_rmse(model, x_train_Nu, y_train_Nu, t_train_Nu)))
        history["field_l2"].append(float(field_l2_error(model, x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu)))
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

    print("\n=== Stage 0 Report ===")
    print(f"final field L2 error: {history['field_l2'][-1]:.3e}")
    print(f"final PDE residual RMSE: {history['pde_residual_rmse'][-1]:.3e}")
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