"""Step 3: train a baseline LSTM to predict RUL on C-MAPSS FD001.

Run from the project root:  python src/train_lstm.py
Needs data/processed/fd001_windows.npz from preprocess_fd001.py
Saves the best model to models/lstm_fd001.pt and a plot to figures/.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---- Settings ----
HIDDEN = 64        # size of the LSTM's memory
LAYERS = 2         # stacked LSTM layers
DROPOUT = 0.2      # randomly switches off units during training to reduce overfitting
BATCH = 256
LR = 1e-3          # learning rate
EPOCHS = 50
PATIENCE = 8       # stop if validation hasn't improved for this many epochs
SEED = 42
# ------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

# Use the Apple Silicon GPU if available, otherwise the CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

d = np.load("data/processed/fd001_windows.npz")
X_tr, y_tr = torch.tensor(d["X_train"]), torch.tensor(d["y_train"])
X_va, y_va = torch.tensor(d["X_val"]), torch.tensor(d["y_val"])
X_te, y_te = torch.tensor(d["X_test"]), d["y_test"]
n_features = X_tr.shape[2]

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=1024)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, cycles, HIDDEN)
        last = out[:, -1, :]           # the LSTM's summary after the final cycle
        return self.head(last).squeeze(-1)


model = LSTMRegressor(n_features).to(device)
loss_fn = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)


def predict(X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy()


def rmse(pred, true) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def phm08_score(pred, true) -> float:
    """Asymmetric scoring function from the PHM08 challenge:
    predicting failure too LATE (d > 0) is penalised more than too early."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


Path("models").mkdir(exist_ok=True)
best_val, wait, history = float("inf"), 0, []

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimiser.step()

    train_rmse = rmse(predict(X_tr), y_tr.numpy())
    val_rmse = rmse(predict(X_va), y_va.numpy())
    history.append((train_rmse, val_rmse))
    print(f"Epoch {epoch:2d} | train RMSE {train_rmse:6.2f} | val RMSE {val_rmse:6.2f}")

    if val_rmse < best_val:
        best_val, wait = val_rmse, 0
        torch.save(model.state_dict(), "models/lstm_fd001.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"Early stopping: no improvement for {PATIENCE} epochs")
            break

# ---- Evaluate the best model on the test engines ----
model.load_state_dict(torch.load("models/lstm_fd001.pt", map_location=device))
pred_te = np.clip(predict(X_te), 0, None)
print(f"\nBest validation RMSE: {best_val:.2f}")
print(f"TEST RMSE:        {rmse(pred_te, y_te):.2f}")
print(f"TEST PHM08 score: {phm08_score(pred_te, y_te):.1f}  (lower is better)")

# ---- Plots for the write-up ----
Path("figures").mkdir(exist_ok=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
h = np.array(history)
a1.plot(h[:, 0], label="train")
a1.plot(h[:, 1], label="validation")
a1.set_xlabel("Epoch"); a1.set_ylabel("RMSE (cycles)"); a1.set_title("Training curve"); a1.legend()

order = np.argsort(y_te)
a2.plot(y_te[order], label="true RUL", linewidth=2)
a2.plot(pred_te[order], "o", markersize=4, label="predicted RUL")
a2.set_xlabel("Test engine (sorted by true RUL)"); a2.set_ylabel("RUL (cycles)")
a2.set_title("Test set: predicted vs true"); a2.legend()
fig.tight_layout()
fig.savefig("figures/lstm_fd001_results.png", dpi=150)
print("Saved figures/lstm_fd001_results.png")
