"""Step 1: load and explore C-MAPSS FD001.

Run from the project root:  python src/explore_fd001.py
Expects train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/raw/
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/raw")

# 26 columns: engine id, cycle, 3 operational settings, 21 sensors
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]


def load(name: str) -> pd.DataFrame:
    # Files are space-separated with trailing spaces, so read with a regex
    # separator and keep only the 26 real columns.
    df = pd.read_csv(DATA_DIR / name, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = COLS
    return df


train = load("train_FD001.txt")
test = load("test_FD001.txt")
rul_test = pd.read_csv(DATA_DIR / "RUL_FD001.txt", header=None, names=["RUL"])

# Training engines run until failure, so RUL = last cycle - current cycle
train["RUL"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]

print("Train shape:", train.shape, "| engines:", train["unit"].nunique())
print("Test shape: ", test.shape, "| engines:", test["unit"].nunique())
print("Test RUL labels:", len(rul_test))
print("\nCycles per engine (train):")
print(train.groupby("unit")["cycle"].max().describe())

# Sensors that never change carry no information - note these for later
flat = [c for c in COLS[2:] if train[c].std() < 1e-4]
print("\nNear-constant columns (candidates to drop):", flat)

# Plot a few sensors for 5 engines against RUL to see degradation trends
Path("figures").mkdir(exist_ok=True)
sensors = ["s2", "s3", "s4", "s7", "s11", "s12"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, s in zip(axes.flat, sensors):
    for u in range(1, 6):
        d = train[train["unit"] == u]
        ax.plot(d["RUL"], d[s], linewidth=0.8)
    ax.invert_xaxis()  # time runs left to right towards failure
    ax.set_title(s)
    ax.set_xlabel("RUL (cycles)")
fig.suptitle("FD001: sensor readings as engines approach failure (5 engines)")
fig.tight_layout()
fig.savefig("figures/fd001_sensor_trends.png", dpi=150)
print("\nSaved figures/fd001_sensor_trends.png")
