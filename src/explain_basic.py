"""Step 4: connect the LSTM to Mistral (no RAG yet).

Run from the project root:  python src/explain_basic.py            (engine 1)
                            python src/explain_basic.py 34         (any test engine 1-100)

This is the "LLM-only" baseline: Mistral explains the LSTM's prediction using
only the numbers we give it, with no retrieved documents. Later we add RAG and
compare the two, which gives you an evaluation for the dissertation.

Needs: Ollama running with the mistral model, and `pip install ollama`.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import ollama

# ---- Must match train_lstm.py and preprocess_fd001.py ----
HIDDEN, LAYERS, DROPOUT, RUL_CAP = 64, 2, 0.2, 125
# ----------------------------------------------------------
LLM_MODEL = "mistral"

# What each kept sensor measures. Taken from the sensor table in
# Saxena et al. (2008) - check these against the paper before citing them.
SENSOR_INFO = {
    "s2": "T24, total temperature at LPC outlet (°R)",
    "s3": "T30, total temperature at HPC outlet (°R)",
    "s4": "T50, total temperature at LPT outlet (°R)",
    "s7": "P30, total pressure at HPC outlet (psia)",
    "s8": "Nf, physical fan speed (rpm)",
    "s9": "Nc, physical core speed (rpm)",
    "s11": "Ps30, static pressure at HPC outlet (psia)",
    "s12": "phi, ratio of fuel flow to Ps30 (pps/psi)",
    "s13": "NRf, corrected fan speed (rpm)",
    "s14": "NRc, corrected core speed (rpm)",
    "s15": "BPR, bypass ratio",
    "s17": "htBleed, bleed enthalpy",
    "s20": "W31, HPT coolant bleed (lbm/s)",
    "s21": "W32, LPT coolant bleed (lbm/s)",
}


class LSTMRegressor(nn.Module):  # same as in train_lstm.py
    def __init__(self, n_features: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=LAYERS,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def sensor_summary(window: np.ndarray, features: list[str]) -> str:
    """Describe each sensor's level and trend over the 30-cycle window.
    Values are scaled 0-1, where 0/1 = lowest/highest seen in training data."""
    lines = []
    for i, f in enumerate(features):
        level = window[-5:, i].mean()                      # recent level
        change = level - window[:5, i].mean()              # change across window
        trend = "rising" if change > 0.05 else "falling" if change < -0.05 else "stable"
        lines.append(f"- {f} ({SENSOR_INFO.get(f, 'unknown')}): "
                     f"level {level:.2f} of training range, {trend} ({change:+.2f})")
    return "\n".join(lines)


def build_prompt(engine: int, pred: float, summary: str) -> str:
    return f"""You are a maintenance assistant for turbofan aircraft engines.
An LSTM model has analysed the last 30 operating cycles of engine {engine}.

Predicted remaining useful life (RUL): {pred:.0f} cycles.
(Predictions are capped at {RUL_CAP} cycles; {RUL_CAP} means "healthy, no clear degradation yet".)

Sensor readings over the last 30 cycles:
{summary}

In under 200 words:
1. State the health status of the engine in one sentence.
2. Name the sensors that most suggest degradation, and why.
3. Recommend a maintenance action and how urgently it is needed.
Only use the information above. If something is uncertain, say so."""


def main():
    engine = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    d = np.load("data/processed/fd001_windows.npz")
    features = [str(f) for f in d["features"]]
    window = d["X_test"][engine - 1]
    true_rul = float(d["y_test"][engine - 1])

    model = LSTMRegressor(len(features))
    model.load_state_dict(torch.load("models/lstm_fd001.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred = float(model(torch.tensor(window[None])).item()) * RUL_CAP
    pred = max(pred, 0.0)

    prompt = build_prompt(engine, pred, sensor_summary(window, features))
    print("=" * 70)
    print(f"Engine {engine} | predicted RUL {pred:.0f} | true RUL {true_rul:.0f}")
    print("=" * 70)
    print("PROMPT SENT TO MISTRAL:\n" + prompt)
    print("=" * 70)
    print("MISTRAL'S EXPLANATION:\n")

    reply = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    print(reply["message"]["content"])


if __name__ == "__main__":
    main()
