# Week 4 — Retrieval-Augmented Generation (RAG)

Week 4 follows one fictional business scenario: an internal HR knowledge assistant for **ABC Technologies**. The course narrative references 10,000 HR documents, while each runnable demo intentionally indexes only a few representative synthetic documents.

## Project index

| Day | Topic | Project |
| --- | --- | --- |
| 1 | RAG fundamentals and authorization | [Day 1 — RAG Authorization](./Day-1-RAG-Authorization/) |
| 2 | Document loading and chunking | [Day 2 — Document Loading & Chunking](./Day-2-Document-Loading-Chunking/) |
| 3 | Choosing a vector database | Conceptual lesson comparing Pinecone, Weaviate, Milvus, and OpenSearch; no separate code project. |
| 4 | Complete retrieval-to-generation pipeline | [Day 4 — Complete RAG Pipeline](./Day-4-Complete-RAG-Pipeline/) |
| 5 | Evaluation and optimization | [Day 5 — RAG Evaluation](./Day-5-RAG-Evaluation/) |
| 6 | No-code revision | Conceptual/no-code revision; no Python project. |
| 7 | End-to-end HR knowledge assistant | [Day 7 — HR Knowledge Assistant](./Day-7-HR-Knowledge-Assistant/) |

## Environment setup

Use Python 3.10 or newer. Each project is self-contained and has its own requirements file.

```bash
git clone https://github.com/saigayatri-blai/12-week-AI-Engineering.git
cd 12-week-AI-Engineering/Week-4

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install the requirements for the project being recorded:

```bash
pip install -r Day-1-RAG-Authorization/requirements.txt
# or
pip install -r Day-2-Document-Loading-Chunking/requirements.txt
# or
pip install -r Day-4-Complete-RAG-Pipeline/requirements.txt
# or
pip install -r Day-5-RAG-Evaluation/requirements.txt
# or
pip install -r Day-7-HR-Knowledge-Assistant/requirements.txt
```

The first SentenceTransformers run may download `all-MiniLM-L6-v2`. ChromaDB runs locally. Never commit or display API credentials.

## Quick verification

```bash
python -m unittest discover -s Day-1-RAG-Authorization -p 'test_*.py' -v
python -m unittest discover -s Day-2-Document-Loading-Chunking -p 'test_*.py' -v
python -m unittest discover -s Day-4-Complete-RAG-Pipeline -p 'test_*.py' -v
python -m unittest discover -s Day-5-RAG-Evaluation -p 'test_*.py' -v
python -m unittest discover -s Day-7-HR-Knowledge-Assistant -p 'test_*.py' -v
```

## Run commands

```bash
python Day-1-RAG-Authorization/day1_rag_demo.py
python Day-2-Document-Loading-Chunking/day2_demo.py --vector
python Day-4-Complete-RAG-Pipeline/day4_demo.py --vector
python Day-5-RAG-Evaluation/BuildLifeandAI_Week4_Day5_Evaluation_Demo.py

cd Day-7-HR-Knowledge-Assistant
python ingest.py
python evaluate.py
streamlit run app.py
```

Day 1, Day 2, Day 4, Day 5, and Day 7 were verified with local ChromaDB and SentenceTransformers. Day 7's deterministic mode is a no-key rehearsal path, not an LLM. Live OpenAI generation is optional, may incur API cost, and has not been executed as part of repository verification.

## Security and data boundary

All policies, users, and compensation figures are fictional. Authorization filters are applied during retrieval, before restricted text can enter the model context. Demo profile selections are not authentication; a production application must obtain identity, tenant, role, and entitlements from trusted systems.
