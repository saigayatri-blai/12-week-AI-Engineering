# Week 3 · Day 1 — What Are Embeddings?

A small, free Python demonstration for **BuildLifeandAI's 24-Week AI Engineering Roadmap**.

## Goal
Explore whether an embedding model considers the regional snack names **Sakinalu** and **Chekkinalu** related, and distinguish shared-description similarity from names-only similarity.

## Setup
Requires Python 3.10+ and internet access for the **first model download**. No paid API key or GPU required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python day1_embeddings.py
```

## Experiments
1. Encode three product descriptions and inspect the `(3, 384)` embedding shape.
2. Compare Sakinalu/Chekkinalu descriptions and an unrelated smartphone cable using cosine similarity.
3. Remove the shared descriptions and compare the two regional names alone.

## Interpretation
Cosine similarity measures vector alignment, **not verified synonymy**. Shared wording in descriptions can inflate apparent similarity. The names-only test helps check whether the model learned a relationship independently, though a single pair is not a complete evaluation. Record the actual scores rather than using invented outputs.

## Files
- `day1_embeddings.py` — complete runnable demonstration
- `requirements.txt` — dependency
- `demo_teleprompter.md` — under-three-minute narration
