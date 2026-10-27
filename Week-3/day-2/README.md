# Week 3 Day 2 — Semantic vs Keyword Search

BuildLifeandAI | Free Python demo, no paid API and no GPU required.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python day2_search.py
```

On Windows, activate with `.venv\\Scripts\\activate`.

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` and may take a few minutes. Subsequent runs are faster.

## What the project shows

1. **Basic substring keyword search:** Searching for `Sakinalu` returns `[]` because that exact text is absent.
2. **Semantic search:** The pretrained embedding model ranks all five products by cosine similarity. The scores and ordering are computed live; **do not assume Chekkinalu ranks first**.
3. **Synonym normalization:** A known mapping `sakinalu → chekkinalu` retrieves the intended listing with simple keyword matching.

The two regional terms are an illustrative naming case. This demo does not establish that the model independently understands their relationship.

## Record a clean demo

Install dependencies and run once before recording. Then rerun `python day2_search.py` and capture the three labeled sections. Show real scores and the Chekkinalu rank.

## Scope

This is an educational demonstration, not production BM25, a vector database, or an evaluation suite. For production retrieval, use representative labeled queries, consider lexical retrieval/BM25, synonym expansion, hybrid retrieval, filters and reranking.
