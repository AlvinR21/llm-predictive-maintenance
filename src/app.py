"""Step 7: Streamlit interface for the LLM-augmented predictive maintenance system.

Run from the project root:   streamlit run src/app.py
It opens in your browser at http://localhost:8501

Needs everything from earlier steps (processed data, LSTM model, FAISS index) and
Ollama running with the chosen model(s). Reuses all logic from explain_rag.py, so the
app shows exactly what the evaluation measured.
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch

from explain_rag import RUL_CAP, NOISE_THRESHOLD, explain, load_all, top_sensors
from explain_basic import SENSOR_INFO

st.set_page_config(page_title="LLM Predictive Maintenance", layout="wide")


# ---------------------------------------------------------------- data (cached)
@st.cache_resource(show_spinner="Loading LSTM, data and knowledge index...")
def resources():
    return load_all()


@st.cache_data(show_spinner="Predicting RUL for all test engines...")
def fleet_predictions():
    d, features, lstm, *_ = resources()
    with torch.no_grad():
        preds = lstm(torch.tensor(d["X_test"])).numpy() * RUL_CAP
    preds = np.clip(preds, 0, None)
    df = pd.DataFrame({
        "engine": np.arange(1, len(preds) + 1),
        "predicted_RUL": preds.round(0).astype(int),
        "true_RUL": d["y_test"].astype(int),
    })
    df["band"] = df["predicted_RUL"].apply(band_label)
    return df


def band_label(pred: float) -> str:
    if pred < 30:
        return "CRITICAL"
    if pred < 100:
        return "DEGRADING"
    return "HEALTHY"


BAND_COLOUR = {"CRITICAL": "red", "DEGRADING": "orange", "HEALTHY": "green"}

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Settings")
fleet = fleet_predictions()
engine = st.sidebar.selectbox(
    "Test engine", fleet["engine"],
    format_func=lambda e: f"Engine {e}  (predicted RUL {fleet.loc[e - 1, 'predicted_RUL']})")
model_name = st.sidebar.radio("Language model", ["llama3.1", "mistral"],
                              help="Llama 3.1 performed best in the evaluation.")
use_rag = st.sidebar.toggle("Use retrieval (RAG)", value=True)
use_guard = st.sidebar.toggle("Use guardrail", value=True,
                              help="Rewrites answers that recommend instrument checks.")
compare = st.sidebar.checkbox("Also show the other setting (RAG on vs off)")
st.sidebar.caption("Each explanation takes roughly 20-60 seconds on a laptop.")

# ---------------------------------------------------------------- header
st.title("LLM-Augmented Predictive Maintenance")
st.caption("LSTM remaining-useful-life prediction on NASA C-MAPSS FD001, "
           "explained by a local LLM with retrieval from maintenance documents.")

tab_engine, tab_fleet, tab_about = st.tabs(["Engine", "Fleet overview", "About"])

# ---------------------------------------------------------------- engine tab
with tab_engine:
    row = fleet.loc[engine - 1]
    pred, true = int(row["predicted_RUL"]), int(row["true_RUL"])
    band = row["band"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL", f"{pred} cycles")
    c2.metric("True RUL", f"{true} cycles", delta=f"{pred - true:+d} error",
              delta_color="off")
    c3.markdown(f"**Health band**  \n:{BAND_COLOUR[band]}[**{band}**]")
    c4.caption(f"RUL is capped at {RUL_CAP} cycles, so {RUL_CAP} means "
               f"'no clear degradation yet'.")

    # Sensor trends over the 30-cycle window
    d, features, *_ = resources()
    window = d["X_test"][engine - 1]
    tops = top_sensors(window, features)

    st.subheader("Largest sensor changes (last 30 cycles)")
    cols = st.columns(3)
    for col, (s, desc, dirn, ch) in zip(cols, tops):
        small = abs(ch) < NOISE_THRESHOLD
        col.markdown(f"**{s}**: {desc}  \n{dirn} ({ch:+.2f})"
                     + ("  \n:gray[within normal variation]" if small else ""))
        i = features.index(s)
        col.line_chart(pd.DataFrame({s: window[:, i]},
                                    index=pd.RangeIndex(-29, 1, name="cycle")),
                       height=160)

    with st.expander("All 14 sensors (scaled 0 = lowest, 1 = highest in training data)"):
        st.line_chart(pd.DataFrame(window, columns=features,
                                   index=pd.RangeIndex(-29, 1, name="cycle")))
        st.caption("; ".join(f"{f}: {SENSOR_INFO.get(f, '')}" for f in features))

    # Explanation
    st.subheader("Explanation")
    settings = [(use_rag, "RAG" if use_rag else "LLM only")]
    if compare:
        settings.append((not use_rag, "LLM only" if use_rag else "RAG"))

    if st.button("Explain this engine", type="primary"):
        for rag, label in settings:
            key = (engine, model_name, rag, use_guard)
            if key not in st.session_state:
                with st.spinner(f"{model_name} is writing the {label} explanation..."):
                    st.session_state[key] = explain(engine, model_name,
                                                    rag=rag, use_guard=use_guard)

    shown = [(lbl, st.session_state.get((engine, model_name, rag, use_guard)))
             for rag, lbl in settings]
    if not any(r for _, r in shown):
        st.info("Choose your settings on the left, then press **Explain this engine**.")
    out_cols = st.columns(len(shown))
    for col, (label, r) in zip(out_cols, shown):
        if not r:
            continue
        with col:
            st.markdown(f"#### {label}  ·  {r['model']}")
            if r["guard"] and r["flagged_before_guard"]:
                st.warning(f"Guardrail: {len(r['flagged_before_guard'])} instrument-check "
                           f"action(s) were flagged and the answer was rewritten.")
            st.markdown(r["final_answer"].replace("\n", "  \n"))
            st.caption(f"Direction check: {r['direction_score']}/3 key sensors reported "
                       f"correctly · instrument actions left: {len(r['flagged_after_guard'])}")
            if r["guard"] and r["flagged_before_guard"]:
                with st.expander("First answer (before guardrail)"):
                    st.markdown(r["raw_answer"].replace("\n", "  \n"))
            if r["hits"]:
                with st.expander(f"Retrieved passages ({len(r['hits'])})"):
                    for n, (p, s, q) in enumerate(r["hits"], start=1):
                        kind = "meaning" if q.startswith("turbofan") else "action"
                        st.markdown(f"**[{n}] {p['source']}, page {p['page']}** "
                                    f"· {kind} search · similarity {s:.2f}")
                        st.caption(p["text"])
            with st.expander("Prompt sent to the model"):
                st.code(r["prompt"], language=None)

# ---------------------------------------------------------------- fleet tab
with tab_fleet:
    st.subheader("All 100 test engines")
    counts = fleet["band"].value_counts()
    a, b, c = st.columns(3)
    a.metric("Critical (< 30 cycles)", int(counts.get("CRITICAL", 0)))
    b.metric("Degrading (30-99)", int(counts.get("DEGRADING", 0)))
    c.metric("Healthy (100+)", int(counts.get("HEALTHY", 0)))

    rmse = float(np.sqrt(((fleet["predicted_RUL"] - fleet["true_RUL"]) ** 2).mean()))
    st.caption(f"LSTM test RMSE on these engines: {rmse:.2f} cycles "
               f"(rounded predictions, so it may differ slightly from training output).")
    st.scatter_chart(fleet, x="true_RUL", y="predicted_RUL", color="band", height=320)
    st.dataframe(fleet.sort_values("predicted_RUL"), hide_index=True,
                 width="stretch")

# ---------------------------------------------------------------- about tab
with tab_about:
    st.markdown(f"""
**How it works**
1. An LSTM reads the last 30 cycles of 14 sensors and predicts remaining useful life (RUL).
2. The three sensors that changed most are found; changes under {NOISE_THRESHOLD} are treated
   as normal variation.
3. With RAG on, two searches (what the symptoms *mean*, and what to *do*) retrieve passages from
   the FAA Powerplant Handbook, the C-MAPSS paper (Saxena et al., 2008) and FAA AC 33.90-1A.
4. A local LLM (via Ollama) writes a STATUS / KEY SENSORS / ACTIONS explanation.
5. A rule-based guardrail flags actions that recommend instrument checks (the simulated sensors are
   accurate) and asks the model once to rewrite them.

**Limitations**
- Explanations are advisory and generated by a language model; they are not certified maintenance
  guidance.
- C-MAPSS is simulated data; real engines need manufacturer maintenance manuals.
- The knowledge base contains little on compressor wear, so retrieved passages are often only
  partly relevant (see the evaluation).
""")
