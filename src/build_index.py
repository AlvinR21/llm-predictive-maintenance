"""Step 5a: build the RAG knowledge index (run once, or again after adding documents).

Run from the project root:  python src/build_index.py
Reads every PDF in data/knowledge/, writes data/index/ (FAISS index + passages).

Pipeline: PDF pages -> text -> overlapping passages -> embeddings -> FAISS index.
Every passage keeps its file name and page number so answers can cite sources.
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

KNOWLEDGE = Path("data/knowledge")
INDEX_DIR = Path("data/index")
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast embedding model (downloads once, ~90 MB)
CHUNK_CHARS = 1000                # passage length in characters
OVERLAP = 200                     # overlap so sentences cut at a boundary aren't lost
MIN_CHARS = 200                   # skip near-empty pages (covers, figure-only pages)

# Optional: only index some pages of a file, to keep the knowledge base on-topic.
# Page numbers are the PDF viewer's numbers (1 = first page of the file), inclusive.
# Example:  PAGE_RANGES = {"amt_powerplant_handbook.pdf": [(15, 120), (400, 460)]}
# Leave empty to index every page of every PDF.
PAGE_RANGES: dict[str, list[tuple[int, int]]] = {}

# Topic filter: the handbook covers piston engines and propellers too, which are
# irrelevant to turbofans. A passage is dropped if it mentions piston-engine terms
# more often than turbine-engine terms. Set to False to index everything.
TURBINE_ONLY = True
PISTON_TERMS = ["reciprocating", "piston", "cylinder", "carburetor", "magneto",
                "propeller", "spark plug", "crankshaft", "valve clearance"]
TURBINE_TERMS = ["turbine", "compressor", "turbofan", "turbojet", "combustor",
                 "combustion chamber", "exhaust gas temperature", "egt", "bleed",
                 "engine pressure ratio", "n1", "n2", "hot section"]


def is_turbine_passage(text: str) -> bool:
    t = text.lower()
    piston = sum(t.count(w) for w in PISTON_TERMS)
    turbine = sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in TURBINE_TERMS)
    return turbine >= piston


def wanted(file: str, page: int) -> bool:
    ranges = PAGE_RANGES.get(file)
    return ranges is None or any(a <= page <= b for a, b in ranges)


# Skip chunks that are mostly numbers (e.g. tables of fault coefficients): the LLM
# cannot use them and they take retrieval slots from explanatory text.
MAX_DIGIT_RATIO = 0.15


def is_number_table(text: str) -> bool:
    """Mostly-number tables, and tables of contents (rows of dot leaders '.......')."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return True
    digits = sum(c.isdigit() for c in chars) / len(chars)
    dots = text.count(".....")
    return digits > MAX_DIGIT_RATIO or dots >= 3


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)   # rejoin words hyphenated across lines
    # Some PDFs (e.g. the NASA copy of Saxena et al. 2008) extract hyphens as "0":
    # "run0to0failure" -> "run-to-failure", "C0MAPSS" -> "C-MAPSS"
    text = re.sub(r"(?<=[A-Za-z])0(?=[A-Za-z])", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk(text: str):
    start = 0
    while start < len(text):
        yield text[start:start + CHUNK_CHARS]
        start += CHUNK_CHARS - OVERLAP


passages = []
dropped = 0
tables = 0
pdfs = sorted(KNOWLEDGE.glob("*.pdf"))
if not pdfs:
    raise SystemExit(f"No PDFs found in {KNOWLEDGE}/")

for pdf in pdfs:
    reader = PdfReader(pdf)
    used = 0
    for i, page in enumerate(reader.pages, start=1):
        if not wanted(pdf.name, i):
            continue
        text = clean(page.extract_text() or "")
        if len(text) < MIN_CHARS:
            continue
        used += 1
        for piece in chunk(text):
            if is_number_table(piece):
                tables += 1
                continue
            if TURBINE_ONLY and not is_turbine_passage(piece):
                dropped += 1
                continue
            passages.append({"source": pdf.name, "page": i, "text": piece})
    print(f"{pdf.name}: {len(reader.pages)} pages, {used} indexed")

print(f"Total passages kept: {len(passages)} | dropped as piston/propeller content: {dropped}"
      f" | dropped as number tables / contents pages: {tables}")
print(f"Embedding with {EMBED_MODEL} (first run downloads the model)...")

model = SentenceTransformer(EMBED_MODEL)
vectors = model.encode([p["text"] for p in passages], batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
vectors = np.asarray(vectors, dtype="float32")

# Inner product on normalised vectors = cosine similarity
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

INDEX_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(INDEX_DIR / "knowledge.faiss"))
(INDEX_DIR / "passages.json").write_text(json.dumps(
    {"embed_model": EMBED_MODEL, "passages": passages}, indent=1))
print(f"Saved {index.ntotal} vectors to {INDEX_DIR}/")

# Quick sanity check: does a turbine-maintenance question find sensible pages?
q = "high pressure compressor degradation inspection turbine engine"
qv = model.encode([q], normalize_embeddings=True).astype("float32")
scores, ids = index.search(qv, min(3, index.ntotal))
print(f"\nTest search: '{q}'")
for s, i in zip(scores[0], ids[0]):
    p = passages[i]
    print(f"  [{p['source']} p.{p['page']}] score {s:.2f}: {p['text'][:120]}...")
