"""Index the three fictional ABC Technologies HR documents in ChromaDB."""
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_DB_DIR = ROOT / ".chroma"
COLLECTION_NAME = "abc_hr_knowledge"

DOCUMENT_CONFIG = {
    "remote_work.txt": {
        "id": "remote_work",
        "title": "Remote Work Policy",
        "access": "all",
    },
    "parental_leave.txt": {
        "id": "parental_leave",
        "title": "Parental Leave Policy",
        "access": "all",
    },
    "executive_compensation.txt": {
        "id": "executive_compensation",
        "title": "Executive Compensation",
        "access": "hr_leadership",
    },
}


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load the original three-document project corpus and its ACL metadata."""
    documents = []
    for filename, metadata in DOCUMENT_CONFIG.items():
        path = data_dir / filename
        documents.append(
            {
                "id": metadata["id"],
                "text": path.read_text(encoding="utf-8").strip(),
                "metadata": {
                    "source": filename,
                    "title": metadata["title"],
                    "access": metadata["access"],
                },
            }
        )
    return documents


def get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def ingest_documents(
    db_dir: Path = DEFAULT_DB_DIR,
    *,
    data_dir: Path = DATA_DIR,
    reset: bool = True,
    encoder=None,
):
    import chromadb

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    existing_names = {collection.name for collection in client.list_collections()}
    if reset and COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    documents = load_documents(data_dir)
    encoder = encoder or get_encoder()
    texts = [item["text"] for item in documents]
    collection.upsert(
        ids=[item["id"] for item in documents],
        documents=texts,
        metadatas=[item["metadata"] for item in documents],
        embeddings=encoder.encode(texts).tolist(),
    )
    return collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()
    collection = ingest_documents(args.db_dir)
    print(f"Indexed {collection.count()} representative documents in {args.db_dir}")
    print("Business scenario: 10,000 HR documents (illustrative); actual indexed documents: 3")
    print("ACL metadata: remote_work=all, parental_leave=all, executive_compensation=hr_leadership")


if __name__ == "__main__":
    main()
