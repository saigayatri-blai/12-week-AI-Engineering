# Day 7 — ABC Technologies HR Knowledge Assistant

This implements the original Sunday project design: three fictional HR documents, SentenceTransformers embeddings, ChromaDB with source and ACL metadata, authorization-filtered retrieval, OpenAI generation, source display, and retrieval evaluation.

The business story references 10,000 HR documents. This demo actually indexes exactly three representative files:

- `remote_work.txt` — available to all employees
- `parental_leave.txt` — available to all employees
- `executive_compensation.txt` — restricted to HR leadership

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
```

## Run the application

```bash
streamlit run app.py
```

The default deterministic mode is a no-key rehearsal path and is not an LLM. To use actual generation, select **Live OpenAI** after setting `OPENAI_API_KEY`. Do not show a real key while recording; API calls may incur cost.

## Required recorded demonstrations

1. Select **Regular Employee** and ask `What is the CEO's annual salary?` The assistant abstains and the restricted source is absent.
2. Select **HR Leadership** and ask the same question. The answer includes `$534,000` and the executive-compensation source.
3. Select **Regular Employee** and ask `How many days per week can employees work remotely?` The answer says `up to three days per week, subject to manager approval.`

Roles are simulated for teaching. The application enforces the `access` metadata filter inside the Chroma query, before retrieved text is assembled into model context. A production system must derive roles from trusted authentication and authorization services.

## Evaluate retrieval

```bash
python evaluate.py
```

`evaluate.py` prints each real Chroma ranking and calculates Recall@3 from those returned IDs. It does not insert a scripted perfect score.

## Tests

```bash
python -m unittest test_day7.py -v
```

The test suite exercises real ChromaDB and SentenceTransformers retrieval, both authorization profiles, the required remote-work answer, actual Recall@3 calculation, and the default Streamlit interaction. Live OpenAI generation is intentionally not called by tests.
