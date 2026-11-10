"""Day 2: demonstrate a real ingestion/chunking failure and its fix.
Run: python day2_demo.py
Optional vector search: python day2_demo.py --vector
"""
from pathlib import Path
import argparse
ROOT = Path(__file__).parent
POLICY = ROOT / "sample_docs" / "maternity_policy.txt"

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_pdf(path: Path) -> str:
    import pymupdf
    with pymupdf.open(path) as pdf:
        return "\n".join(page.get_text() for page in pdf)

def fixed_size_chunks(text: str, size: int = 90) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]

def section_chunks(text: str) -> list[str]:
    """Keep eligibility, benefit and decision together for this short policy.
    In a real system use a document-aware parser for headings/tables.
    """
    sections = [s.strip() for s in text.split('SECTION:') if s.strip()]
    return ["ABC Technologies — Maternity Leave Policy\n" + '\n'.join(sections)]

def has_complete_evidence(chunk: str) -> bool:
    s = chunk.lower()
    return all(x in s for x in ('16 weeks', 'given birth', '12 months'))

def vector_search(chunks: list[str], query: str):
    import chromadb
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    db = chromadb.Client()
    collection = db.create_collection('day2_chunks')
    collection.add(ids=[str(i) for i in range(len(chunks))], documents=chunks,
                   embeddings=encoder.encode(chunks).tolist())
    return collection.query(query_embeddings=encoder.encode([query]).tolist(),
                            n_results=min(2, len(chunks)))['documents'][0]

def main():
    text = load_text(POLICY)
    bad = fixed_size_chunks(text)
    good = section_chunks(text)
    print('SOURCE:', POLICY.name)
    print('Naive fixed-size chunks:', len(bad))
    print('Fixed-size chunks with COMPLETE eligibility evidence:', sum(map(has_complete_evidence,bad)))
    print('Structure-preserving chunks:', len(good))
    print('Structure-preserving chunks with COMPLETE evidence:', sum(map(has_complete_evidence,good)))
    print('\nBEFORE (incomplete evidence):', next(c for c in bad if '16 weeks' in c))
    print('\nAFTER (complete evidence):\n',good[0])
    if argparse.ArgumentParser().parse_known_args()[0] and '--vector' in __import__('sys').argv:
        print('\nChromaDB retrieval:',vector_search(good,'Am I eligible for maternity leave?'))
    print('\nIMPORTANT: A policy chunk alone cannot establish an individual employee\'s eligibility.')
if __name__ == '__main__': main()
