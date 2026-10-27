"""BuildLifeandAI | Week 3 Day 1: Embeddings.

Run: python day1_embeddings.py
Requires: pip install -r requirements.txt
The first run downloads a free pretrained model.
"""

from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def similarity(model, left: str, right: str) -> float:
    vectors = model.encode([left, right], convert_to_tensor=True)
    return float(util.cos_sim(vectors[0], vectors[1]).item())


def main() -> None:
    print("\nBUILD LIFE AND AI | WEEK 3, DAY 1: EMBEDDINGS")
    print("Loading free pretrained model (first run may download it)...")
    model = SentenceTransformer(MODEL_NAME)

    descriptions = [
        "Sakinalu - traditional crispy rice flour snack",
        "Chekkinalu - traditional crispy rice flour snack",
        "Smartphone charging cable",
    ]
    vectors = model.encode(descriptions)
    print("\nEXPERIMENT 1: TURN TEXT INTO NUMBERS")
    print("Products:", len(descriptions))
    print("Embedding shape:", vectors.shape)
    print("First 5 numbers of Sakinalu embedding:", vectors[0][:5].round(3))

    print("\nEXPERIMENT 2: COMPARE PRODUCT DESCRIPTIONS")
    description_match = similarity(model, descriptions[0], descriptions[1])
    unrelated_match = similarity(model, descriptions[0], descriptions[2])
    print(f"Sakinalu description vs Chekkinalu description: {description_match:.3f}")
    print(f"Sakinalu description vs smartphone cable:     {unrelated_match:.3f}")

    print("\nEXPERIMENT 3: REMOVE THE SHARED DESCRIPTION")
    names_only = similarity(model, "Sakinalu", "Chekkinalu")
    print(f"Sakinalu vs Chekkinalu (names only):          {names_only:.3f}")

    print("\nTAKEAWAY")
    print("Shared descriptive words can increase similarity.")
    print("Compare with names-only scores before claiming the model knows regional synonyms.")
    print("Scores are model-dependent; verify the actual results on your machine.")


if __name__ == "__main__":
    main()
