"""Calculate actual Recall@3 from Chroma retrieval for the Day 7 examples."""
from pathlib import Path
import argparse

from ingest import DEFAULT_DB_DIR, ingest_documents, get_encoder
from rag import HR_LEADERSHIP, REGULAR_EMPLOYEE, retrieve


EVALUATION_CASES = [
    {
        "question": "How many days per week can employees work remotely?",
        "profile": REGULAR_EMPLOYEE,
        "relevant_ids": {"remote_work"},
    },
    {
        "question": "What is the CEO's annual salary?",
        "profile": HR_LEADERSHIP,
        "relevant_ids": {"executive_compensation"},
    },
    {
        "question": "What does the parental leave policy say?",
        "profile": REGULAR_EMPLOYEE,
        "relevant_ids": {"parental_leave"},
    },
]


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 3) -> float:
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def run_evaluation(collection, encoder) -> list[dict]:
    rows = []
    for case in EVALUATION_CASES:
        results = retrieve(
            case["question"],
            case["profile"],
            collection=collection,
            encoder=encoder,
            top_k=3,
        )
        ids = [item["id"] for item in results]
        rows.append(
            {
                "question": case["question"],
                "profile": case["profile"].display_name,
                "retrieved_ids": ids,
                "relevant_ids": sorted(case["relevant_ids"]),
                "recall_at_3": recall_at_k(ids, case["relevant_ids"], 3),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--reingest", action="store_true")
    args = parser.parse_args()
    encoder = get_encoder()
    collection = ingest_documents(args.db_dir, reset=True, encoder=encoder) if args.reingest else None
    if collection is None:
        from rag import open_collection

        collection = open_collection(args.db_dir)
    rows = run_evaluation(collection, encoder)
    for row in rows:
        print("\nQUESTION:", row["question"])
        print("PROFILE:", row["profile"])
        print("RETRIEVED:", row["retrieved_ids"])
        print("RELEVANT:", row["relevant_ids"])
        print(f"Recall@3: {row['recall_at_3']:.2f}")
    average = sum(row["recall_at_3"] for row in rows) / len(rows)
    print(f"\nMacro-average Recall@3 across {len(rows)} questions: {average:.2f}")
    print("These values were calculated from the displayed Chroma retrieval results.")


if __name__ == "__main__":
    main()
