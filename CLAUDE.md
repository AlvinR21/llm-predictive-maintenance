# Project: LLM for Predictive Maintenance (final-year capstone)

BSc Computer Science final-year project, University of West London. Submission due January 2027.

## What the project is
- A hybrid system: an LSTM model predicts Remaining Useful Life (RUL) of turbofan engines,
  and a RAG pipeline with a local open-source LLM explains the predictions and suggests
  maintenance actions.
- Dataset: NASA C-MAPSS FD001. Main reference: Saxena, A., Goebel, K., Simon, D. and
  Eklund, N. (2008) 'Damage propagation modeling for aircraft engine run-to-failure
  simulation', 2008 International Conference on Prognostics and Health Management, IEEE.
- Front end: Streamlit (next step).

## Tech stack
- Python 3.11, PyTorch (LSTM; Apple Silicon "mps" device where possible)
- FAISS + sentence-transformers (all-MiniLM-L6-v2): retrieval, written in plain Python
  (LangChain dropped for transparency and stability)
- Ollama, local models: Llama 3.1 8B (best, default) and Mistral 7B (comparison)
- Streamlit: web interface

## Environment
- MacBook Air (Apple Silicon), Python 3.11, virtual environment in `venv/`.

## Folder layout
- `data/raw/` C-MAPSS text files; `data/processed/` windows (.npz)
- `data/knowledge/` RAG documents: FAA AMT Handbook - Powerplant (FAA-H-8083-32B),
  Saxena et al. 2008 (C-MAPSS.pdf), FAA AC 33.90-1A
- `data/index/` FAISS index + passages.json (built by build_index.py)
- `models/` lstm_fd001.pt; `figures/` plots; `results/` evaluation CSV/JSON
- `src/` scripts (all run from the project root):
  - explore_fd001.py, preprocess_fd001.py, train_lstm.py (Phase 1)
  - explain_basic.py: LLM-only explanation baseline
  - build_index.py: PDF -> chunks -> embeddings -> FAISS (turbine-only filter,
    number-table/contents-page filter, hyphen fix)
  - explain_rag.py: two searches (meaning + action), max 2 passages per source,
    shared prompt for LLM-only vs RAG, health bands, noise threshold 0.10,
    instrument-check guardrail (one rewrite), direction check. --model, --both, --no-guard
  - evaluate.py: 10 engines x 2 models x 2 conditions -> results/eval_results_<tag>.csv

## Current stage
- Phase 1 done: LSTM test RMSE 13.26, PHM08 score 308 (FD001).
- Phase 2 done: RAG pipeline + evaluation. Final run = v3 (results/eval_results_v3.csv).
  Best config: Llama 3.1 + RAG + guardrail (30/30 directions, 0 instrument actions left,
  5/5 healthy engines correct, 0 contradictory answers).
- Key findings: RAG alone introduced instrument-check advice (0/20 runs without retrieval vs
  10/20 with, in v1) because the handbook troubleshooting table (p.427-428) dominates
  retrieval; guardrail fixes most; health bands fixed over-alarming on healthy engines;
  Mistral produced contradictory/format-leak answers (13/20), Llama none.
- Lessons logged: first LSTM collapsed to the mean (unscaled target); prompt wording could
  not fix a knowledge-base gap; keyword guardrail had a false positive (negation) and a
  miss ("recalibrating"), both fixed.
- Next: manual scoring of eval_results_v3.csv (Llama + RAG rows at least), Streamlit app,
  then the dissertation write-up.

## How to help me
- I'm still building my Python confidence. Explain what code does and why,
  and let me write or change parts myself rather than generating everything at once.
- Keep changes small and runnable; one step at a time.
- This is assessed work: I need to understand and be able to defend every part of it.
