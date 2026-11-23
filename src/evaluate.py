"""Step 6: evaluate the explanation pipeline over several engines and setups.

Run from the project root:
    python src/evaluate.py                                  10 engines, mistral + llama3.1
    python src/evaluate.py --models mistral                 one model only
    python src/evaluate.py --engines 34,80,12               your own engine list
    python src/evaluate.py --tag v2                         name this run (default: date-time)

For every engine x model x condition (LLM-only, RAG) it records the prediction,
retrieved passages, the answer before and after the guardrail, and two automatic
scores. It writes:
    results/eval_results_<tag>.csv  one row per run, with EMPTY columns for your manual scores
    results/eval_full_<tag>.json    everything, including full answers and prompts
Each run gets its own files, so earlier results are never overwritten.

Expect it to take a while on a laptop (roughly 20-60 s per LLM call).
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from explain_rag import explain, load_all

args = sys.argv[1:]
models = (args[args.index("--models") + 1].split(",") if "--models" in args
          else ["mistral", "llama3.1"])

if "--engines" in args:
    engines = [int(e) for e in args[args.index("--engines") + 1].split(",")]
else:
    # 10 engines spread from near-failure to healthy, chosen by true RUL
    y_test = load_all()[0]["y_test"]
    order = np.argsort(y_test)
    engines = sorted({int(order[i]) + 1 for i in np.linspace(0, len(order) - 1, 10).astype(int)})

tag = args[args.index("--tag") + 1] if "--tag" in args else datetime.now().strftime("%Y%m%d_%H%M")
csv_path = Path(f"results/eval_results_{tag}.csv")
json_path = Path(f"results/eval_full_{tag}.json")

print(f"Engines: {engines}\nModels: {models}\n")
Path("results").mkdir(exist_ok=True)
rows, full = [], []
total = len(engines) * len(models) * 2
done, start = 0, time.time()

for engine in engines:
    for model_name in models:
        for rag in (False, True):
            r = explain(engine, model_name, rag=rag, use_guard=True)
            done += 1
            print(f"[{done}/{total}] engine {engine} | {model_name} | {r['condition']} | "
                  f"directions {r['direction_score']}/3 | guardrail fired: "
                  f"{bool(r['flagged_before_guard'])} | {time.time() - start:.0f}s elapsed")
            rows.append({
                "engine": engine, "model": model_name, "condition": r["condition"],
                "true_rul": r["true_rul"], "pred_rul": r["pred_rul"],
                "abs_error": round(abs(r["pred_rul"] - r["true_rul"]), 1),
                "top_sensors": "; ".join(f"{s} {d}" for s, _, d, _ in r["top_sensors"]),
                "passages": "; ".join(f"{p['source']} p.{p['page']}" for p, _, _ in r["hits"]),
                "direction_score_0to3": r["direction_score"],
                "instrument_actions_before_guard": len(r["flagged_before_guard"]),
                "instrument_actions_after_guard": len(r["flagged_after_guard"]),
                "final_answer": r["final_answer"],
                # ---- fill these in yourself after reading each answer ----
                "MANUAL_relevant_passages_0to4": "",
                "MANUAL_citations_supported_yes_no_partly": "",
                "MANUAL_action_quality_0to2": "",
                "MANUAL_notes": "",
            })
            full.append({k: v for k, v in r.items()})

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json_path.write_text(json.dumps(full, indent=1, default=str))

print("\nSUMMARY (automatic scores)")
for model_name in models:
    for cond in ("llm_only", "rag"):
        sel = [r for r in rows if r["model"] == model_name and r["condition"] == cond]
        dirs = np.mean([r["direction_score_0to3"] for r in sel])
        fired = sum(r["instrument_actions_before_guard"] > 0 for r in sel)
        left = sum(r["instrument_actions_after_guard"] for r in sel)
        print(f"  {model_name:10s} {cond:9s} | mean directions {dirs:.2f}/3 | "
              f"guardrail fired on {fired}/{len(sel)} | instrument actions left {left}")
print(f"\nSaved {csv_path} and {json_path}")
