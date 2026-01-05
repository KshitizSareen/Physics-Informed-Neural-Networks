import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from timeit import default_timer

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

seeds_num = 666
torch.manual_seed(seeds_num)
np.random.seed(seeds_num)

# === Constants ===
DENSITY = 1.68
SPECIFIC_HEAT_CAPACITY = 0.96
Q = 2.192
INITIAL_TEMP = 21.23
THERMAL_CONDUCTIVITY = 10

true_values = {"rho": DENSITY, "cp": SPECIFIC_HEAT_CAPACITY, "lam": THERMAL_CONDUCTIVITY}

def load_data(filepath="temperature_output.csv"):
    return pd.read_csv(filepath)

def prepare_training_data(df):
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

# -------------------------
# Load + normalize ONLY temperature
# -------------------------
df = load_data()
coords_array, tempValues = prepare_training_data(df)

minX, maxX = coords_array[:, 0].min(), coords_array[:, 0].max()
minY, maxY = coords_array[:, 1].min(), coords_array[:, 1].max()
minT, maxT = coords_array[:, 2].min(), coords_array[:, 2].max()
minTemp, maxTemp = tempValues.min(), tempValues.max()

tempValues = (2 * ((tempValues - minTemp) / (maxTemp - minTemp))) - 1  # [-1,1]

X_train_tensor = torch.from_numpy(coords_array).float().to(device)
U_train = torch.from_numpy(tempValues).float().to(device)

x_train = X_train_tensor[:, 0:1]
y_train = X_train_tensor[:, 1:2]
t_train = X_train_tensor[:, 2:3]

# Boundary + Initial masks in PHYSICAL coords
eps = 1e-6
boundary_mask = (
    np.isclose(coords_array[:, 0], minX, atol=eps) | np.isclose(coords_array[:, 0], maxX, atol=eps) |
    np.isclose(coords_array[:, 1], minY, atol=eps) | np.isclose(coords_array[:, 1], maxY, atol=eps)
)
initial_mask = np.isclose(coords_array[:, 2], minT, atol=eps)

Xb = torch.from_numpy(coords_array[boundary_mask]).float().to(device)
Ub = torch.from_numpy(tempValues[boundary_mask]).float().to(device)
xb, yb, tb = Xb[:, 0:1], Xb[:, 1:2], Xb[:, 2:3]

X0 = torch.from_numpy(coords_array[initial_mask]).float().to(device)
U0 = torch.from_numpy(tempValues[initial_mask]).float().to(device)
x0, y0, t0 = X0[:, 0:1], X0[:, 1:2], X0[:, 2:3]

# -------------------------
# Model
# -------------------------
class PINN(nn.Module):
    """
    PINN with Fourier features.
    Exactly ONE trainable physical parameter among: rho, cp, lam.
    Others fixed to true values.

    Parameterization: phys = exp(hat) to keep positive.
    """
    def __init__(
        self,
        m=20,                      # number of Fourier frequencies
        hidden_dim=100,
        num_hidden=3,
        activation="tanh",
        learn_param="rho",
        true_rho=DENSITY,
        true_cp=SPECIFIC_HEAT_CAPACITY,
        true_lam=THERMAL_CONDUCTIVITY,
        init_ranges=None,
        eps=1e-12,
        B_scale=3.0,               # frequency bandwidth
    ):
        super().__init__()
        self.eps = eps
        self.learn_param = learn_param.lower().strip()
        if self.learn_param not in {"rho", "cp", "lam", "none"}:
            raise ValueError("learn_param must be one of: 'rho','cp','lam','none'")

        # ---- Fourier matrix B as BUFFER (moves with .to(device)) ----
        B = torch.randn(m, 3, dtype=torch.float32) * B_scale
        self.register_buffer("B", B)  # shape (m,3)

        input_dim = 2 * m

        # ---- MLP ----
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "silu":
            self.activation = torch.nn.functional.silu
        elif activation == "sin":
            self.activation = torch.sin
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if init_ranges is None:
            init_ranges = {"rho": (0.5, 5.0), "cp": (0.1, 2.5), "lam": (0.1, 50.0)}

        self.register_buffer("rho_fixed", torch.tensor([float(true_rho)], dtype=torch.float32))
        self.register_buffer("cp_fixed",  torch.tensor([float(true_cp)],  dtype=torch.float32))
        self.register_buffer("lam_fixed", torch.tensor([float(true_lam)], dtype=torch.float32))

        self.rho_hat = None
        self.cp_hat = None
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

    @property
    def rho(self):
        return torch.exp(self.rho_hat) if self.rho_hat is not None else self.rho_fixed

    @property
    def cp(self):
        return torch.exp(self.cp_hat) if self.cp_hat is not None else self.cp_fixed

    @property
    def lam(self):
        return torch.exp(self.lam_hat) if self.lam_hat is not None else self.lam_fixed

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=-1)   # (N,3)
        proj = (2.0 * torch.pi) * (X @ self.B.t())  # (N,m)
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N,2m)

        out = feats
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)
        return out

    def loss_PDE(self, x, y, t):
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        u_y = torch.autograd.grad(u, y, torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

        # Undo temperature normalization for derivatives
        temp_scale = (maxTemp - minTemp) / 2.0
        u_t  = u_t  * temp_scale
        u_xx = u_xx * temp_scale
        u_yy = u_yy * temp_scale

        r = (self.rho * self.cp * u_t) - (self.lam * (u_xx + u_yy)) - Q
        return torch.mean(r ** 2)

    def loss_initial(self, x, y, t):
        u = self.forward(x, y, t)
        u = 0.5 * (u + 1) * (maxTemp - minTemp) + minTemp
        return torch.mean((u - INITIAL_TEMP) ** 2)

    def loss_bounds(self, x, y, t):
        x = x.detach().clone().requires_grad_(True)
        y = y.detach().clone().requires_grad_(True)
        t = t.detach().clone().requires_grad_(True)

        u = self.forward(x, y, t)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

        temp_scale = (maxTemp - minTemp) / 2.0
        u_x = u_x * temp_scale
        u_y = u_y * temp_scale
        return torch.mean(u_x ** 2) + torch.mean(u_y ** 2)

    def loss_data(self, x, y, t, u_obs):
        return torch.mean((self.forward(x, y, t) - u_obs) ** 2)

    def losses(self, x_all, y_all, t_all, u_all, x0, y0, t0, xb, yb, tb):
        Lr = self.loss_PDE(x_all, y_all, t_all)
        Li = self.loss_initial(x0, y0, t0)
        Lb = self.loss_bounds(xb, yb, tb)
        Ld = self.loss_data(x_all, y_all, t_all, u_all)
        return Lr, Li, Lb, Ld

# -------------------------
# Train (LBFGS only, as in your code)
# -------------------------
def main(param_to_learn):
    torch.manual_seed(seeds_num)

    model = PINN(learn_param=param_to_learn, m=20, hidden_dim=100, num_hidden=3, activation="tanh").to(device)

    lambda_d = lambda_r = lambda_b = lambda_i = 1.0
    history = {"rho": [], "cp": [], "lam": [], "time_sec": []}

    def get_param_val():
        return float(getattr(model, param_to_learn).detach().cpu().item())

    t_start = default_timer()

    lbfgs_iters = 2000
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=1, history_size=100, line_search_fn="strong_wolfe")

    # start equal weights
    current_lambdas = {"pde": 0.25, "ic": 0.25, "bc": 0.25, "data": 0.25}

    def closure():
        optimizer.zero_grad()
        Lr, Li, Lb, Ld = model.losses(x_train, y_train, t_train, U_train, x0, y0, t0, xb, yb, tb)
        L = (
            lambda_r * current_lambdas["pde"]  * Lr +
            lambda_i * current_lambdas["ic"]   * Li +
            lambda_b * current_lambdas["bc"]   * Lb +
            lambda_d * current_lambdas["data"] * Ld
        )
        L.backward()
        return L

    for it in range(lbfgs_iters):
        # update grad-balanced lambdas once per outer iter (not inside closure)
        with torch.enable_grad():
            Lr, Li, Lb, Ld = model.losses(x_train, y_train, t_train, U_train, x0, y0, t0, xb, yb, tb)
            losses = {"data": Ld, "pde": Lr, "bc": Lb, "ic": Li}

            # exclude *_hat (inverse scalar) from balancing if desired
            params = [p for (n, p) in model.named_parameters() if p.requires_grad and not n.endswith("_hat")]

            grads = {}
            for name, term in losses.items():
                g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True, create_graph=False)
                gn = torch.zeros((), device=device)
                cnt = 0
                for gi in g:
                    if gi is not None:
                        gn = gn + gi.norm()
                        cnt += 1
                grads[name] = gn / max(cnt, 1)

            total_grad = sum(grads.values())
            lambdas = {k: total_grad / (grads[k] + 1e-8) for k in grads}
            Z = sum(lambdas.values()) + 1e-12
            current_lambdas = {k: float((v / Z).detach().cpu().item()) for k, v in lambdas.items()}  # store as floats

        L = optimizer.step(closure)
        curr = float(L.detach().cpu().item())

        elapsed = default_timer() - t_start
        history["rho"].append(float(model.rho.detach().cpu().item()))
        history["cp"].append(float(model.cp.detach().cpu().item()))
        history["lam"].append(float(model.lam.detach().cpu().item()))
        history["time_sec"].append(elapsed)

        if it % 10 == 0 or it == lbfgs_iters - 1:
            pv = get_param_val()
            print(
                f"[LBFGS {it:04d}] L={curr:.3e} | {param_to_learn}={pv:.6f} | "
                f"lams(pde,ic,bc,data)=({current_lambdas['pde']:.2f},{current_lambdas['ic']:.2f},"
                f"{current_lambdas['bc']:.2f},{current_lambdas['data']:.2f})"
            )

    t_total = default_timer() - t_start
    print(f"Total training time: {t_total/60:.2f} min")

    plt.figure(figsize=(7, 4))
    plt.plot(history["time_sec"], history[param_to_learn], label=f"Estimated {param_to_learn}", linewidth=2)
    plt.axhline(y=true_values[param_to_learn], color="black", linestyle="--", linewidth=2, label=f"True {param_to_learn}")
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

for param in ["rho", "cp", "lam"]:
    main(param)
