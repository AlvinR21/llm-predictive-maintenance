# Project: LLM for Predictive Maintenance (final-year capstone)

BSc Computer Science final-year project, University of West London. Submission due January 2027.

## What the project is
- A hybrid system: an LSTM model predicts Remaining Useful Life (RUL) of turbofan engines,
  and a RAG pipeline with an open-source LLM (Mistral 7B or Llama 3 8B) explains the
  predictions and suggests maintenance actions.
- Dataset: NASA C-MAPSS (start with FD001). Cite: Saxena, A. and Goebel, K. (2008)
  Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.
- Front end: Streamlit.

## Tech stack (from README)
- Python 3.11
- PyTorch: LSTM RUL predictor (Apple Silicon, so use the "mps" device where possible)
- FAISS + sentence-transformers: document retrieval
- Ollama + Mistral 7B: language model, run locally
- LangChain: RAG orchestration
- Streamlit: web interface

## Environment
- MacBook Air (Apple Silicon), Python 3.11, virtual environment in `venv/`.

## Folder layout
- `venv/` virtual environment (not committed to git)
- `data/raw/` original C-MAPSS text files (not committed to git)
- `src/` Python modules and scripts
- `notebooks/` exploration
- `figures/` plots for the dissertation

## Current stage
- Phase 1: Foundation (in progress). Project folder, venv, .gitignore and README are done.
- Done: explore FD001 (`src/explore_fd001.py`); preprocessing with 14 sensors,
  RUL cap 125, 30-cycle windows, 80/20 split by engine (`src/preprocess_fd001.py`);
  baseline LSTM (`src/train_lstm.py`, target scaled 0-1, validation RMSE about 13.6).
- Lesson logged: the first LSTM collapsed to predicting the mean because the
  target was unscaled; fixed by scaling RUL to 0-1.
- Next: Phase 2, the RAG pipeline (Ollama + Mistral 7B, FAISS, LangChain), then Streamlit.

## How to help me
- I'm still building my Python confidence. Explain what code does and why,
  and let me write or change parts myself rather than generating everything at once.
- Keep changes small and runnable; one step at a time.
- This is assessed work: I need to understand and be able to defend every part of it.
