# Week 3 — Day 5: Vector Indexing & Meta FAISS

Two runnable demos in one script:

1. Encode six real product descriptions with `all-MiniLM-L6-v2` and retrieve the top three using **exact cosine similarity** via `faiss.IndexFlatIP` with normalized embeddings.
2. Create **10,000 synthetic vectors**, compare exact search to IVF approximate search (`IndexIVFFlat`, 100 clusters), and display **recall@5** for `nprobe=1, 5, 20`.

## Setup

Python 3.10+ recommended. In a terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python day5_faiss.py
```

Windows activation: `.venv\\Scripts\\activate`.

The model is downloaded on first run; run once before recording. Subsequent runs use the local cache.

## Important distinctions

- The first six vectors are real embeddings; the 10,000 vectors in the second experiment are **synthetic**, not shopping products.
- `IndexFlatIP` performs exact nearest-neighbor search using inner product. On unit-normalized vectors, inner product equals cosine similarity.
- `IndexIVFFlat` partitions vectors into clusters; `nprobe` sets how many clusters to search. It can miss exact nearest neighbors.
- Recall@5 shown here is for **one** test query. It is not a production-quality evaluation or a speed benchmark.
- FAISS is an indexing/search library, not a full vector database.
