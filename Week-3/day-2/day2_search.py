"""Week 3 Day 2 — Keyword vs Semantic Search | BuildLifeandAI."""
from sentence_transformers import SentenceTransformer, util

PRODUCTS = [
    "Chekkinalu",
    "Homemade Mango Pickle",
    "Crispy Rice Flour Snack Rings",
    "Spicy South Indian Murukku",
    "Peanut Chikki",
]
QUERY = "Sakinalu"

def keyword_search(query, products):
    """Intentionally simple substring matching, not full-text/BM25."""
    return [p for p in products if query.casefold() in p.casefold()]

def semantic_search(query, products, model):
    product_vectors = model.encode(products)
    query_vector = model.encode(query)
    scores = util.cos_sim(query_vector, product_vectors)[0]
    return sorted(zip(products, scores.tolist()), key=lambda x: x[1], reverse=True)

def main():
    print("\\n=== WEEK 3 / DAY 2: KEYWORD vs SEMANTIC SEARCH ===")
    print("Query:", QUERY)
    print("\\nCatalog:")
    for p in PRODUCTS:
        print(" -", p)

    print("\\n[1] BASIC KEYWORD SEARCH")
    matches = keyword_search(QUERY, PRODUCTS)
    print("Matches:", matches)
    print("Why? The exact word 'Sakinalu' is absent from every listing.")

    print("\\n[2] SEMANTIC SEARCH — FREE PRETRAINED MODEL")
    print("Loading all-MiniLM-L6-v2 (first run downloads model)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ranked = semantic_search(QUERY, PRODUCTS, model)
    for rank, (product, score) in enumerate(ranked, start=1):
        print(f"{rank}. {score:+.3f} | {product}")
    chekk_rank = next(i for i, (p, _) in enumerate(ranked, 1) if "Chekkinalu" in p)
    print(f"Chekkinalu rank: {chekk_rank}/{len(PRODUCTS)}")
    print("NOTE: Scores are real model outputs; ranking is NOT guaranteed.")
    print("This model may not know the regional-name relationship.")

    print("\\n[3] EXPLICIT SYNONYM FIX")
    synonyms = {"sakinalu": "chekkinalu"}
    normalized = synonyms.get(QUERY.casefold(), QUERY)
    print("Normalized query:", normalized)
    print("Matches:", keyword_search(normalized, PRODUCTS))
    print("\\nTakeaway: Evaluate retrieval; prefer a reliable synonym when known.")

if __name__ == "__main__":
    main()
