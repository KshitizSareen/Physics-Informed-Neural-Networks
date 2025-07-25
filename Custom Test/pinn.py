import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Exact solution for a damped oscillator equation used for benchmarking the PINN
def exact_solution(d, w0, t):
    assert d < w0  # ensure underdamping condition
    w = np.sqrt(w0**2 - d**2)  # damped angular frequency
    phi = np.arctan(-d / w)    # phase shift due to damping
    A = 1 / (2 * np.cos(phi))  # amplitude factor
    cos = torch.cos(phi + w * t)
    exp = torch.exp(-d * t)
    u = exp * 2 * A * cos      # exact solution: decaying cosine wave
    return u

# Fully connected neural network (FCN) used as a PINN
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        # First layer (input to hidden)
        layers = [nn.Linear(N_INPUT, N_HIDDEN), nn.Tanh()]
        # Hidden layers (N_LAYERS - 1 blocks of Linear + Tanh)
        layers += [layer for _ in range(N_LAYERS - 1) for layer in (nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh())]
        # Final layer (hidden to output)
        layers += [nn.Linear(N_HIDDEN, N_OUTPUT)]
        # Combine into one sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":

    torch.manual_seed(123)  # for reproducibility

    pinn = FCN(1, 1, 32, 3)  # input: t, output: u(t)

    # Initial condition point: t = 0
    t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

    # Physics domain: 300 time points between t = 0 and t = 1
    t_physics = torch.linspace(0, 1, 300).view(-1, 1).requires_grad_(True)

    # Damping coefficient and natural frequency
    d, w0 = 2, 20
    mu, k = 2 * d, w0**2  # for the ODE: u'' + mu u' + k u = 0

    # Testing time points for plotting the exact solution
    t_test = torch.linspace(0, 1, 300).view(-1, 1)
    u_exact = exact_solution(d, w0, t_test)

    # Optimizer
    optimiser = torch.optim.Adam(pinn.parameters(), lr=1e-3)

    for i in range(15001):
        optimiser.zero_grad()

        # Loss weights (initial condition and physics residual)
        lambda1, lambda2 = 1e-1, 1e-4

        # Evaluate the model at the initial boundary t=0
        u = pinn(t_boundary)  # This gives u(t=0)

        # Loss1 enforces the initial condition: u(0) = 1
        # torch.squeeze converts vector to scalar
        loss1 = (torch.squeeze(u) - 1)**2

        # Compute du/dt at t = 0 using autograd
        # This calculates ∂u/∂t at t=0 using the chain rule on the computation graph
        # Arguments:
        #   - outputs: u (network output)
        #   - inputs: t_boundary (input variable)
        #   - grad_outputs: torch.ones_like(u) simulates ∂L/∂u = 1 in chain rule
        #   - create_graph=True allows higher-order derivatives later
        dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
        loss2 = (torch.squeeze(dudt)-0)**2

        # Evaluate the neural network over the full physics domain t ∈ [0, 1]
        # This gives predicted values u(t) for all 300 time steps
        u = pinn(t_physics)

        # Compute the first derivative ∂u/∂t at all 300 time points using autograd
        # This returns a tensor dudt of shape (300, 1), representing u'(t)
        dudt = torch.autograd.grad(
            u,                      # output of the network
            t_physics,              # input with respect to which we compute the gradient
            torch.ones_like(u),     # vector-Jacobian product multiplier (∂L/∂u = 1)
            create_graph=True       # retain computation graph for higher-order gradients
        )[0]

        # Compute the second derivative ∂²u/∂t² at all 300 time points using autograd
        # This returns d2udt2 of shape (300, 1), representing u''(t)
        d2udt2 = torch.autograd.grad(
            dudt,                   # first derivative (∂u/∂t)
            t_physics,              # same input variable
            torch.ones_like(dudt),  # ∂L/∂(∂u/∂t) = 1 to get full Jacobian-vector product
            create_graph=True       # needed if you plan to compute further derivatives
        )[0]

        # Define the physics loss based on the ODE: u'' + μu' + ku = 0
        # This term penalizes the network when it violates the physical law
        # Equivalent to: mean((u'' + μu' + ku)^2)
        loss3 = torch.mean((d2udt2 + mu * dudt + k * u)**2)

        # Total loss combines:
        # - loss1: initial condition u(0) = 1
        # - loss2: initial time derivative ∂u/∂t (t=0) = 0
        # - loss3: physics-based residual enforcing the ODE u'' + μu' + ku = 0
        loss = loss1 + lambda1 * loss2 + lambda2 * loss3

        # Backpropagate the total loss to compute gradients w.r.t. all model parameters
        loss.backward()

        # Update model parameters using the optimizer (Adam)
        optimiser.step()

        # Plot the predicted vs exact solution every 10 epochs
        if i % 500 == 0:
            with torch.no_grad():
                u_pred = pinn(t_test)
            plt.figure(figsize=(6, 4))
            plt.plot(t_test.numpy(), u_exact.numpy(), label='Exact Solution', linewidth=2)
            plt.plot(t_test.numpy(), u_pred.numpy(), label='PINN Prediction', linestyle='--')
            plt.xlabel('t')
            plt.ylabel('u(t)')
            plt.title(f'PINN vs Exact at Epoch {i}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()



