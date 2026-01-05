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
        u = 0.5 * (u + 1) * (maxTemp-minTemp) + minTemp
        return torch.mean((u - INITIAL_TEMP) ** 2)

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

    # ---------------------------
    # Adaptive lambda computation (safe outside closure)
    # ---------------------------
    def compute_adaptive_lambdas(Lr, Li, Lb, Ld, include_hat_params=False):
        losses = {"data": Ld, "pde": Lr, "bc": Lb, "ic": Li}

        # If include_hat_params=False, we balance using ONLY field-network params,
        # excluding the scalar inverse param (*_hat) so lambdas don't go crazy.
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
        lambdas = {k: (v / Z) for k, v in lambdas.items()}  # sum=1
        return lambdas

    # ---------------------------
    # L-BFGS only
    # ---------------------------
    lbfgs_iters = 5000
    log_every = 10

    lbfgs_opt = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=1,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    # Early stopping (loss plateau)
    best_loss = float("inf")
    loss_min_delta = 1e-10
    loss_patience = 200  # counts LOG CHECKS
    loss_counter = 0

    # Early stopping (param convergence)
    param_min_delta = 1e-8
    param_patience = 40  # counts LOG CHECKS
    param_counter = 0
    prev_param = None

    # Start with uniform weights
    current_lambdas = {"pde": 0.25, "ic": 0.25, "bc": 0.25, "data": 0.25}

    def closure():
        lbfgs_opt.zero_grad()

        Lr, Li, Lb, Ld = model.losses(
            x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
            x_train_initial, y_train_initial, t_train_initial,
            x_train_boundary, y_train_boundary, t_train_boundary
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
        # Update lambdas ONCE per outer iteration (freeze inside closure)
        with torch.enable_grad():
            Lr, Li, Lb, Ld = model.losses(
                x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
                x_train_initial, y_train_initial, t_train_initial,
                x_train_boundary, y_train_boundary, t_train_boundary
            )
            current_lambdas = compute_adaptive_lambdas(Lr, Li, Lb, Ld, include_hat_params=False)

        L = lbfgs_opt.step(closure)
        curr = float(L.detach())

        # Track time/params (cheap)
        elapsed = default_timer() - t_start
        history["rho"].append(float(model.rho.detach()))
        history["cp"].append(float(model.cp.detach()))
        history["lam"].append(float(model.lam.detach()))
        history["time_sec"].append(elapsed)

        # Log + early stop every log_every
        if it % log_every == 0 or it == lbfgs_iters - 1:
            # recompute losses for reporting (no grads needed)
            with torch.enable_grad():
                Lr, Li, Lb, Ld = model.losses(
                    x_train_Nu, y_train_Nu, t_train_Nu, U_train_Nu,
                    x_train_initial, y_train_initial, t_train_initial,
                    x_train_boundary, y_train_boundary, t_train_boundary
                )

            pv = get_param_val()

            history["L_total"].append(curr)
            history["L_pde"].append(float(Lr.detach()))
            history["L_ic"].append(float(Li.detach()))
            history["L_bc"].append(float(Lb.detach()))
            history["L_data"].append(float(Ld.detach()))

            # ---- loss plateau ES (per LOG) ----
            if curr < best_loss - loss_min_delta:
                best_loss = curr
                loss_counter = 0
            else:
                loss_counter += 1

            # ---- param convergence ES (per LOG) ----
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
                f"(Ld={float(Ld):.3e}, Lr={float(Lr):.3e}, Lb={float(Lb):.3e}, Li={float(Li):.3e}) | "
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

    # ---------------------------
    # Plot + save
    # ---------------------------
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

    save_path = f"convergence_{param_to_learn}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved parameter convergence plot to: {save_path}")
    plt.show()


for param in [ "lam"]:
    main(param)

