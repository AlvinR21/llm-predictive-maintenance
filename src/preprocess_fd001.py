"""Step 2: prepare C-MAPSS FD001 for the LSTM.

Run from the project root:  python src/preprocess_fd001.py
Reads data/raw/, writes data/processed/fd001_windows.npz

What it does:
  1. Drops columns that carry no information about wear.
  2. Caps RUL (early life is treated as "healthy").
  3. Splits training ENGINES into train/validation (by engine, not by row,
     so the model is never validated on an engine it has trained on).
  4. Scales sensors to 0-1, fitting the scaler on the training engines only.
  5. Slices each engine's history into fixed-length windows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---- Settings you may want to change (and justify in your write-up) ----
WINDOW = 30        # cycles per input sequence
RUL_CAP = 125      # common choice in the literature; check papers for your justification
VAL_FRACTION = 0.2 # share of training engines held out for validation
SEED = 42          # makes the train/validation split repeatable
# ------------------------------------------------------------------------

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name, sep=r"\s+", header=None).iloc[:, :26]
    df.columns = COLS
    return df


train = load("train_FD001.txt")
test = load("test_FD001.txt")
rul_test = pd.read_csv(RAW / "RUL_FD001.txt", header=None)[0].to_numpy()

# ---- 1. Feature selection ----
# Flat columns found in step 1, plus op1/op2: FD001 runs at a single operating
# condition, so these settings are essentially noise here.
DROP = ["op1", "op2", "op3", "s1", "s5", "s10", "s16", "s18", "s19"]

# Check s6 before deciding: it varies very slightly
print("s6 std:", round(train["s6"].std(), 6), "| unique values:", train["s6"].nunique())
DROP.append("s6")  # remove this line if you decide to keep s6

FEATURES = [c for c in COLS[2:] if c not in DROP]
print(f"Using {len(FEATURES)} features:", FEATURES)

# ---- 2. RUL labels (capped) ----
train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
train["RUL"] = train["RUL"].clip(upper=RUL_CAP)
rul_test = np.minimum(rul_test, RUL_CAP)

# ---- 3. Split by engine ----
rng = np.random.default_rng(SEED)
units = train["unit"].unique()
val_units = rng.choice(units, size=int(len(units) * VAL_FRACTION), replace=False)
tr = train[~train["unit"].isin(val_units)].copy()
va = train[train["unit"].isin(val_units)].copy()
print(f"Train engines: {tr['unit'].nunique()} | Validation engines: {va['unit'].nunique()}")

# ---- 4. Scale (fit on training engines only, to avoid data leakage) ----
scaler = MinMaxScaler()
tr[FEATURES] = scaler.fit_transform(tr[FEATURES])
va[FEATURES] = scaler.transform(va[FEATURES])
test[FEATURES] = scaler.transform(test[FEATURES])


# ---- 5. Windows ----
def make_windows(df: pd.DataFrame):
    """Every possible WINDOW-length slice of every engine; label = RUL at the window's last cycle."""
    X, y = [], []
    for _, d in df.groupby("unit"):
        f = d[FEATURES].to_numpy()
        r = d["RUL"].to_numpy()
        for end in range(WINDOW, len(d) + 1):
            X.append(f[end - WINDOW:end])
            y.append(r[end - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def last_windows(df: pd.DataFrame):
    """One window per test engine: its final WINDOW cycles (padded at the start if shorter)."""
    X = []
    for _, d in df.groupby("unit"):
        f = d[FEATURES].to_numpy()
        if len(f) < WINDOW:
            pad = np.repeat(f[:1], WINDOW - len(f), axis=0)
            f = np.vstack([pad, f])
        X.append(f[-WINDOW:])
    return np.array(X, dtype=np.float32)


X_tr, y_tr = make_windows(tr)
X_va, y_va = make_windows(va)
X_te = last_windows(test)
y_te = rul_test.astype(np.float32)

short = (test.groupby("unit").size() < WINDOW).sum()
print(f"Test engines shorter than {WINDOW} cycles (padded): {short}")

print("\nShapes  (windows, cycles, features):")
print("  train:", X_tr.shape, "labels:", y_tr.shape)
print("  val:  ", X_va.shape, "labels:", y_va.shape)
print("  test: ", X_te.shape, "labels:", y_te.shape)

np.savez_compressed(
    OUT / "fd001_windows.npz",
    X_train=X_tr, y_train=y_tr,
    X_val=X_va, y_val=y_va,
    X_test=X_te, y_test=y_te,
    features=np.array(FEATURES),
)
print(f"\nSaved {OUT / 'fd001_windows.npz'}")
