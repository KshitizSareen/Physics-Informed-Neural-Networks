import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)



def train_model(X, y, col_name, batch_size=256, epochs=1000):
    # Initialize model, loss, optimizer
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Normalize input X and output y ---
    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    # --- Convert to PyTorch tensors ---
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)

    # --- Create batches ---
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --- Training loop ---
    best_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in dataloader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = model.state_dict()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # --- Restore best model state ---
    model.load_state_dict(best_model_state)

    # --- Save model to disk ---
    torch.save(model.state_dict(), f"model_{col_name}.pt")
