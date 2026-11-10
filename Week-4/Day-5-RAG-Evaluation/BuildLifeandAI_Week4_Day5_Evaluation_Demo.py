"""Week 4 Day 5: evaluate a parental-leave retrieval failure and fix.

This uses a three-document fictional corpus so the displayed ranking and
Recall@K values come from actual ChromaDB retrieval, not pasted IDs.
"""
from uuid import uuid4


QUESTION = "Am I eligible for 16 weeks of parental leave?"
EMPLOYEE_TENURE_MONTHS = 2
GOLD_EVIDENCE_IDS = {"parental_leave_benefit", "parental_leave_eligibility"}

DOCUMENTS = [
    {
        "id": "parental_leave_benefit",
        "text": "Parental leave benefit: Eligible employees receive 16 weeks of parental leave.",
        "section": "Benefit",
    },
    {
        "id": "parental_leave_eligibility",
        "text": (
            "Parental leave eligibility requirements: Employees must complete 12 months of "
            "continuous employment before becoming eligible for parental leave."
        ),
        "section": "Eligibility Requirements",
    },
    {
        "id": "remote_work_policy",
        "text": "Employees may work remotely up to three days per week, subject to manager approval.",
        "section": "Remote Work",
    },
]


def get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_collection(encoder=None):
    import chromadb

    encoder = encoder or get_encoder()
    client = chromadb.Client()
    collection = client.create_collection(f"day5_evaluation_{uuid4().hex}")
    texts = [document["text"] for document in DOCUMENTS]
    collection.add(
        ids=[document["id"] for document in DOCUMENTS],
        documents=texts,
        metadatas=[{"section": document["section"]} for document in DOCUMENTS],
        embeddings=encoder.encode(texts).tolist(),
    )
    return collection, encoder


def retrieve(collection, encoder, question: str = QUESTION, top_k: int = 3) -> list[dict]:
    result = collection.query(
        query_embeddings=encoder.encode([question]).tolist(),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "id": item_id,
            "text": text,
            "section": metadata["section"],
            "distance": distance,
        }
        for item_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def grounded_answer(retrieved: list[dict], tenure_months: int = EMPLOYEE_TENURE_MONTHS) -> str:
    retrieved_ids = {item["id"] for item in retrieved}
    if not GOLD_EVIDENCE_IDS.issubset(retrieved_ids):
        return "I don't have enough retrieved evidence to determine eligibility."
    if tenure_months < 12:
        return (
            "No. According to the fictional parental-leave policy, employees must complete "
            f"12 months of continuous employment to qualify. This employee has completed "
            f"{tenure_months} months."
        )
    return "The tenure requirement is met; HR must still confirm all other eligibility requirements."


def main():
    collection, encoder = create_collection()
    retrieved = retrieve(collection, encoder, top_k=3)
    retrieved_ids = [item["id"] for item in retrieved]

    print("QUESTION:", QUESTION)
    print("EMPLOYEE TENURE:", EMPLOYEE_TENURE_MONTHS, "months")
    print("GOLD EVIDENCE IDS:", sorted(GOLD_EVIDENCE_IDS))
    print("\nACTUAL CHROMA RANKING:")
    for rank, item in enumerate(retrieved, start=1):
        print(f"{rank}. {item['id']} — {item['section']}")

    recall_1 = recall_at_k(retrieved_ids, GOLD_EVIDENCE_IDS, 1)
    recall_3 = recall_at_k(retrieved_ids, GOLD_EVIDENCE_IDS, 3)
    print(f"\nRecall@1: {recall_1:.2f} ({int(recall_1 * 2)}/2 gold evidence chunks)")
    print(f"Recall@3: {recall_3:.2f} ({int(recall_3 * 2)}/2 gold evidence chunks)")

    print("\nTOP-1 ANSWER:")
    print(grounded_answer(retrieved[:1]))
    print("\nTOP-3 GROUNDED ANSWER:")
    print(grounded_answer(retrieved))
    print(
        "\nThis is one measured example. Increasing Top-K does not guarantee better results "
        "for every question."
    )


if __name__ == "__main__":
    main()
