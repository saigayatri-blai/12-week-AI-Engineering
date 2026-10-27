# Week 3 — Embeddings and Semantic Search

Beginner-friendly, local projects for the BuildLifeandAI 24-Week AI Engineering Roadmap. All projects use free local models; no paid API key or GPU is required.

| Day | Project | Main idea |
| --- | --- | --- |
| 1 | [Embeddings](day-1/) | Turn text into vectors and compare cosine similarity |
| 2 | [Semantic vs keyword search](day-2/) | Compare literal matching with meaning-based retrieval |
| 4 | [Vector databases](day-4/) | Store vectors and metadata in persistent Chroma |
| 5 | [FAISS indexing](day-5/) | Compare exact and IVF approximate vector search |

Day 3 is a concept lesson on distance, cosine similarity, and normalization; it has no standalone code project in this folder.

Days 6 and 7 are pending source verification and are intentionally not published yet.

Each project has its own setup and run commands. The first Sentence Transformers run downloads `sentence-transformers/all-MiniLM-L6-v2`; later runs use the local model cache.

## Practical boundaries

- The datasets are small, synthetic examples for learning.
- Similarity is not calibrated confidence or proof that an answer is correct.
- Measure retrieval on representative labeled queries before selecting an index or shipping a search experience.
- Do not store secrets, customer data, or private business documents in these examples.
