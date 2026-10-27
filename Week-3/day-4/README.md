# Week 3 — Day 4: Vector Databases

BuildLifeandAI | **Where Do Millions of AI Embeddings Go?**

A beginner-friendly **local persistent Chroma vector database** project using Sentence Transformers. No paid API or GPU required.

## Run

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python day4_vector_database.py
```

First run downloads `all-MiniLM-L6-v2`. Install and run once **before filming**. Running the script again updates the same product IDs without duplicates. Chroma saves data under `shopping_db/` inside this project directory.

## What you'll demonstrate

1. Create a persistent Chroma collection configured for **cosine distance**.
2. Encode product descriptions into **384-dimensional vectors**.
3. Upsert five products with product ID, name, category, price, and stock status.
4. Search `crispy rice snack` with metadata filters: **snacks, under $10, in stock**.
5. Display actual returned products and cosine distances; reopen the collection to demonstrate persistence.

**Important:** Chroma returns cosine **distance**, not cosine similarity. For cosine distance, smaller means closer. Show the actual rankings from your run; do not invent scores. Shared descriptions do not prove the model understands regional food names.

## Technical notes

- Uses Chroma's newer `configuration={"hnsw": {"space": "cosine"}}` syntax. If an older Chroma version is installed, upgrade using the requirements file. Existing collections retain their original distance configuration; to change it, create a new collection or remove the demo database folder.
- This is a **small local educational demonstration**. Do not claim it benchmarks 10 million products or proves approximate indexing is faster.
- A vector database stores searchable vectors and metadata; it does not replace order, payment, or inventory transactional databases.
- Re-embed documents if their searchable descriptions change. In a real system, synchronize inventory and price data from the source of truth.
