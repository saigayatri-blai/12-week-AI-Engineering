# BuildLifeandAI — Week 4 Day 1 RAG Demo

## Runtime target
Python 3.10+; first run downloads the SentenceTransformers `all-MiniLM-L6-v2` model (~90 MB). Internet access is needed for package/model installation. Optional OpenAI generation requires an API key and may incur cost.

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python day1_rag_demo.py
```

For actual LLM generation (optional):

```bash
export OPENAI_API_KEY='YOUR_KEY'  # Never put a real key in recorded footage
python day1_rag_demo.py
```

## Demo order (~3 minutes of screen time)
1. **Normal question:** Work-from-home policy retrieved.
2. **Security bug:** No filter, salary `$534,000` exposed.
3. **Fix:** `where={"access": {"$in": ["all"]}}` excludes restricted HR document for employee.
4. **Authorized HR leader:** Same question, salary document permitted.
5. Optional: OpenAI generation with authorized context only.

All documents are fictional. Three short sample documents represent a company with **10,000 HR documents**; the code does **not** index 10,000 documents. Each short document is treated as a single chunk; Day 2 covers actual chunking.

## Important limitations
- Offline `grounded_answer()` is **deterministic answer formatting, not LLM generation**. Set `OPENAI_API_KEY` to show real LLM generation.
- `Principal` values are hardcoded for teaching. In production, role/entitlements must come from a trusted identity service; **never** accept arbitrary user-provided role strings.
- Metadata filters are one defense, not a complete authorization design. Enforce permissions consistently on ingestion, updates, retrieval, and logs, and use tenant isolation where needed.
- The `grounded_answer` function is deliberately specific to the three teaching questions. Production grounding requires more robust relevance/evidence checks and evaluation.
- Avoid filming any actual confidential data or API credentials.

## Verification status
Python compilation and deterministic mocked-retrieval tests were run in the authoring environment. **Real ChromaDB + SentenceTransformers execution and live OpenAI calls could not be tested here** because those packages are unavailable and the environment cannot reach the package index. Run the script once on the recording machine before filming.
