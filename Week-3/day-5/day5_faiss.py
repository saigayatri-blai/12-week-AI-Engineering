"""BuildLifeandAI — Week 3 Day 5: exact vs IVF approximate vector search."""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PRODUCTS = [
    "Comfortable running shoes with soft cushioning",
    "Lightweight sneakers for everyday jogging",
    "Breathable athletic shoes for long runs",
    "Supportive walking shoes for daily use",
    "Formal leather office shoes",
    "Wireless Bluetooth headphones",
]


def show_product_search(model):
    print("\n=== PART 1: REAL PRODUCT EMBEDDINGS + EXACT SEARCH ===")
    vectors = model.encode(PRODUCTS, normalize_embeddings=True).astype("float32")
    dimension = vectors.shape[1]
    print("Embedding dimensions:", dimension)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    query = "comfortable running shoes"
    query_vector = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query_vector, k=3)
    print("Query:", query)
    for rank, (product_id, score) in enumerate(zip(ids[0], scores[0]), 1):
        print(f"{rank}. {PRODUCTS[int(product_id)]} | cosine similarity: {float(score):.3f}")
    return dimension


def show_ivf_comparison(dimension):
    print("\n=== PART 2: 10,000 SYNTHETIC VECTORS + EXACT VS IVF ===")
    rng = np.random.default_rng(42)
    # Synthetic vectors for index mechanics, NOT actual product embeddings.
    vectors = rng.standard_normal((10_000, dimension)).astype("float32")
    faiss.normalize_L2(vectors)
    exact = faiss.IndexFlatIP(dimension)
    exact.add(vectors)
    nlist = 100
    quantizer = faiss.IndexFlatIP(dimension)
    ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)
    query = vectors[0:1].copy()
    k = 5
    _, exact_ids = exact.search(query, k)
    print("Exact top-5 IDs:", exact_ids[0].tolist())
    for nprobe in (1, 5, 20):
        ivf.nprobe = nprobe
        _, approx_ids = ivf.search(query, k)
        recall_at_k = len(set(exact_ids[0]) & set(approx_ids[0])) / k
        print(f"IVF nprobe={nprobe:2d} | top-5 IDs: {approx_ids[0].tolist()} | recall@5: {recall_at_k:.0%}")
    print("NOTE: Recall@5 compares IVF against exact neighbors for this one synthetic query.")
    print("For production, evaluate recall over many representative queries and measure latency.")


def main():
    print("BuildLifeandAI | Week 3 Day 5 | Meta FAISS")
    print("Loading free model (first run may download files)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dimension = show_product_search(model)
    show_ivf_comparison(dimension)


if __name__ == "__main__":
    main()
