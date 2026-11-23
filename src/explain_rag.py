"""Step 5b: explain an LSTM prediction with RAG, a guardrail, and a choice of LLM.

Run from the project root:
    python src/explain_rag.py 34                       RAG explanation (mistral, guardrail on)
    python src/explain_rag.py 34 --both                LLM-only AND RAG, for comparison
    python src/explain_rag.py 34 --model llama3.1      use another Ollama model
    python src/explain_rag.py 34 --no-guard            switch the guardrail off
    python src/explain_rag.py 34 --show-prompt         also print the full RAG prompt

Needs: data/index/ from build_index.py, models/lstm_fd001.pt, Ollama running the model.
Reuses the model, sensor summary and settings from explain_basic.py.
evaluate.py imports explain() from here to run many engines at once.
"""
import json
import re
import sys
from functools import lru_cache

import faiss
import numpy as np
import torch
import ollama
from sentence_transformers import SentenceTransformer

from explain_basic import LSTMRegressor, SENSOR_INFO, RUL_CAP, sensor_summary

DEFAULT_MODEL = "mistral"
PER_QUERY = 2      # passages taken from each of the two searches
PER_SOURCE = 2     # max passages from any single document
TEMPERATURE = 0    # 0 = same answer every run (reproducible results for evaluation)

# Changes smaller than this (on the 0-1 scaled range) are treated as normal variation.
# Chosen from the evaluation data: healthy engines' largest changes were about 0.05-0.09,
# near-failure engines about 0.2-0.3. State this threshold and its basis in the write-up.
NOISE_THRESHOLD = 0.10

# Guardrail: in C-MAPSS the sensors are accurate, so actions about instruments are wrong.
INSTRUMENT_TERMS = re.compile(
    r"\b(probe|transmitter|indicator|gauge|(?:re)?calibrat\w*|thermocouple|instrument\w*|"
    r"wiring|pressure line|sensor (?:fault|error|accuracy))\b", re.IGNORECASE)


# ---------------------------------------------------------------- loading
@lru_cache(maxsize=1)
def load_all():
    d = np.load("data/processed/fd001_windows.npz")
    features = [str(f) for f in d["features"]]
    model = LSTMRegressor(len(features))
    model.load_state_dict(torch.load("models/lstm_fd001.pt", map_location="cpu"))
    model.eval()
    data = json.loads(open("data/index/passages.json").read())
    index = faiss.read_index("data/index/knowledge.faiss")
    embedder = SentenceTransformer(data["embed_model"])
    return d, features, model, index, data["passages"], embedder


# ---------------------------------------------------------------- sensors
def changes(window: np.ndarray) -> np.ndarray:
    return window[-5:].mean(axis=0) - window[:5].mean(axis=0)


def top_sensors(window, features, n=3):
    """[(sensor, description, direction, change)] for the n largest changes."""
    ch = changes(window)
    out = []
    for i in np.argsort(-np.abs(ch))[:n]:
        desc = SENSOR_INFO.get(features[i], features[i]).split(" (")[0]
        out.append((features[i], desc, "rising" if ch[i] > 0 else "falling", float(ch[i])))
    return out


def key_changes_text(tops) -> str:
    return "\n".join(
        f"{r}. {s} ({d}): {dirn} ({c:+.2f})"
        + ("  [small: within normal variation]" if abs(c) < NOISE_THRESHOLD else "")
        for r, (s, d, dirn, c) in enumerate(tops, start=1))


def health_band(pred: float) -> str:
    if pred < 30:
        return ("CRITICAL: failure is close. Recommend urgent maintenance before further operation.")
    if pred < 100:
        return ("DEGRADING: clear wear is developing. Recommend planned maintenance soon.")
    return ("HEALTHY / EARLY LIFE: no clear degradation yet. Do not describe the engine as "
            "degrading or deteriorating. Unless a key change is outside normal variation, "
            "the only action needed is routine monitoring and scheduled inspections.")


# ---------------------------------------------------------------- retrieval
def make_queries(tops, pred):
    """Two searches: what the symptoms MEAN, and what to DO about them.
    Note for the write-up: the second query adds generic maintenance vocabulary
    (inspection, borescope, cleaning, repair) - a query-expansion design choice."""
    symptoms = "; ".join(f"{d} {'increasing' if dirn == 'rising' else 'decreasing'}"
                         for _, d, dirn, _ in tops)
    stage = ("near failure" if pred < 30 else "degrading" if pred < 100 else "healthy")
    meaning = (f"turbofan engine {stage}: {symptoms}. What engine module degradation "
               f"(compressor or turbine flow and efficiency loss) causes these changes?")
    action = (f"turbine engine maintenance for internal wear of compressor or turbine "
              f"({symptoms}): inspection, borescope, cleaning, repair procedures")
    return meaning, action


def retrieve(queries, index, passages, embedder):
    hits, used_ids, per_source = [], set(), {}
    for q in queries:
        qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
        scores, ids = index.search(qv, min(200, index.ntotal))
        taken = 0
        for s, i in zip(scores[0], ids[0]):
            if i < 0 or i in used_ids:
                continue
            src = passages[i]["source"]
            if per_source.get(src, 0) >= PER_SOURCE:
                continue
            used_ids.add(i)
            per_source[src] = per_source.get(src, 0) + 1
            hits.append((passages[i], float(s), q))
            taken += 1
            if taken == PER_QUERY:
                break
    return hits


# ---------------------------------------------------------------- prompting
def build_prompt(engine, pred, summary, keys, hits=None):
    band = health_band(pred)
    """One prompt for both conditions, so the only difference is the retrieved passages."""
    if hits:
        context = "\n\n".join(f"[{n}] ({p['source']}, page {p['page']})\n{p['text']}"
                              for n, (p, _, _) in enumerate(hits, start=1))
        ref_block = f"\nReference passages:\n{context}\n"
        action_rule = ("Each action must cite the passage that supports it as [1], [2], etc. "
                       "Only use passage content that matches the observed symptoms; "
                       "ignore rows or passages about symptoms this engine does not show. "
                       "If no passage fits, say so.")
    else:
        ref_block = ""
        action_rule = "Base each action only on the sensor data above."

    return f"""You are a maintenance assistant for turbofan aircraft engines.
An LSTM model has analysed the last 30 operating cycles of engine {engine}.

Predicted remaining useful life (RUL): {pred:.0f} cycles.
(Predictions are capped at {RUL_CAP} cycles; {RUL_CAP} means "healthy, no clear degradation yet".)

Health band from the prediction: {band}

Largest sensor changes (most important first):
{keys}

Important context: these readings come from NASA's C-MAPSS engine simulation, where the
sensors are accurate. Sensor changes reflect real wear inside the engine (for example loss of
compressor or turbine efficiency), NOT faulty gauges, probes or wiring. Do not recommend
instrument checks or re-calibration.

All sensor readings over the last 30 cycles:
{summary}
{ref_block}
Answer in under 200 words, in exactly this format:
STATUS: one sentence on the engine's health and urgency.
KEY SENSORS: the three largest changes above, with direction exactly as given, and what each suggests.
ACTIONS: at most 3 maintenance actions, numbered 1-3, most important first, no repeats. For each,
state which observed symptom it addresses. {action_rule}
If no maintenance is needed, write "ACTIONS: None beyond routine monitoring." instead.
Do not invent information."""


def chat(model_name, messages):
    reply = ollama.chat(model=model_name, messages=messages,
                        options={"temperature": TEMPERATURE})
    return reply["message"]["content"]


def actions_section(answer: str) -> str:
    m = re.search(r"ACTIONS:(.*)", answer, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else answer


NEGATION = re.compile(r"\b(not|no|never|instead|rather|avoid|without)\b", re.IGNORECASE)


def recommends_instrument_check(action: str) -> bool:
    """True if some clause of the action mentions an instrument term WITHOUT a negation.
    So "check the pressure transmitter" is flagged, but
    "(since instrument checks are not allowed)" is not."""
    clauses = re.split(r"[.;()]", action)
    return any(INSTRUMENT_TERMS.search(c) and not NEGATION.search(c) for c in clauses)


def flagged_actions(answer: str) -> list[str]:
    """Numbered action lines that recommend instrument checks."""
    lines = re.split(r"\n\s*(?=\d+[.)])", actions_section(answer))
    return [l.strip() for l in lines if recommends_instrument_check(l)]


def guard(model_name, prompt, answer):
    """If any action is an instrument check, ask the model once to replace it."""
    bad = flagged_actions(answer)
    if not bad:
        return answer, bad
    fix = ("These actions recommend instrument or sensor checks, which are not allowed "
           "because the C-MAPSS sensors are accurate:\n- " + "\n- ".join(bad) +
           "\nRewrite your whole answer in the same format, replacing those actions with "
           "actions about the engine's mechanical condition. Keep the same citation rules.")
    revised = chat(model_name, [{"role": "user", "content": prompt},
                                {"role": "assistant", "content": answer},
                                {"role": "user", "content": fix}])
    return revised, bad


# ---------------------------------------------------------------- checks
def direction_score(answer: str, tops) -> int:
    """How many of the 3 key sensors are reported with the correct direction (0-3)."""
    m = re.search(r"KEY SENSORS:(.*?)(ACTIONS:|$)", answer, re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else answer
    score = 0
    for s, _, dirn, _ in tops:
        wrong = "falling" if dirn == "rising" else "rising"
        lines = [l for l in block.splitlines() if re.search(rf"\b{s}\b", l)]
        if lines and dirn in lines[0].lower() and wrong not in lines[0].lower():
            score += 1
    return score


# ---------------------------------------------------------------- main entry
def explain(engine: int, model_name=DEFAULT_MODEL, rag=True, use_guard=True):
    """Run one engine through one setup. Returns a dict (used by evaluate.py)."""
    d, features, lstm, index, passages, embedder = load_all()
    window = d["X_test"][engine - 1]
    true_rul = float(d["y_test"][engine - 1])
    with torch.no_grad():
        pred = max(float(lstm(torch.tensor(window[None])).item()) * RUL_CAP, 0.0)

    tops = top_sensors(window, features)
    keys = key_changes_text(tops)
    summary = sensor_summary(window, features)
    hits = retrieve(make_queries(tops, pred), index, passages, embedder) if rag else []
    prompt = build_prompt(engine, pred, summary, keys, hits if rag else None)

    raw = chat(model_name, [{"role": "user", "content": prompt}])
    final, bad = guard(model_name, prompt, raw) if use_guard else (raw, flagged_actions(raw))
    return {
        "engine": engine, "model": model_name, "condition": "rag" if rag else "llm_only",
        "guard": use_guard, "pred_rul": round(pred, 1), "true_rul": true_rul,
        "top_sensors": tops, "hits": hits, "prompt": prompt,
        "raw_answer": raw, "final_answer": final,
        "flagged_before_guard": bad, "flagged_after_guard": flagged_actions(final),
        "direction_score": direction_score(final, tops),
    }


def show(r, show_prompt=False):
    line = "=" * 70
    print(f"\n{line}\n{r['condition'].upper()} | model {r['model']} | guard {'on' if r['guard'] else 'off'}")
    print(f"Engine {r['engine']} | predicted RUL {r['pred_rul']:.0f} | true RUL {r['true_rul']:.0f}")
    print(line)
    if r["hits"]:
        print("RETRIEVED PASSAGES:")
        for n, (p, s, q) in enumerate(r["hits"], start=1):
            kind = "meaning" if q.startswith("turbofan") else "action"
            print(f"  [{n}] ({kind} search) {p['source']} p.{p['page']} "
                  f"(similarity {s:.2f}): {p['text'][:140]}...")
        print()
    if show_prompt:
        print("PROMPT:\n" + r["prompt"] + "\n" + line)
    if r["guard"] and r["flagged_before_guard"]:
        print("FIRST ANSWER (before guardrail):\n" + r["raw_answer"])
        print(f"\nGUARDRAIL flagged {len(r['flagged_before_guard'])} action(s); asked for a rewrite.\n")
        print("REVISED ANSWER:")
    print(r["final_answer"])
    print(f"\nDirection check: {r['direction_score']}/3 key sensors correct | "
          f"instrument actions left: {len(r['flagged_after_guard'])}")


def main():
    args = sys.argv[1:]
    engine = int(next((a for a in args if a.isdigit()), 1))
    model_name = args[args.index("--model") + 1] if "--model" in args else DEFAULT_MODEL
    use_guard = "--no-guard" not in args

    d, features, *_ = load_all()
    print("INDEXED SOURCES:", sorted({p["source"] for p in load_all()[4]}))
    if "--both" in args:
        show(explain(engine, model_name, rag=False, use_guard=use_guard))
    show(explain(engine, model_name, rag=True, use_guard=use_guard),
         show_prompt="--show-prompt" in args)


if __name__ == "__main__":
    main()
