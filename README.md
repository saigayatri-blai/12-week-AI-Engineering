# BuildLifeandAI | Week 2: AI Resume Analyzer

**Learning project, not an automated hiring or candidate-ranking system.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env .env  # Windows: copy .env .env
# Edit .env to add your own API key
python app.py sample_resume.txt sample_job.txt
```

Requires a valid API key and access to the configured model. API usage may incur charges. Do not commit `.env` or upload real resumes without permission.

## Concepts covered
- Day 1: clear task and output instructions in `SYSTEM` and `extract_skills()`.
- Day 2: zero-shot baseline (remove example messages) versus few-shot demonstration (`EXAMPLE_USER` / `EXAMPLE_ASSISTANT`).
- Day 3: ask for a concise *decision explanation* in a follow-up; do not ask for private chain-of-thought.
- Day 4: system message, untrusted document boundaries.
- Day 5: JSON Schema with strict structured output.
- Day 6: model function calling to a deterministic Python `compare_skills()` tool, followed by a tool response.
- Day 7: fork, clone, test, and extend this app.

## Experiments
1. Compare zero-shot and few-shot extraction on five sample resumes.
2. Change one requirement and check whether missing skills update.
3. Add a test for case-insensitive matching and skill aliases (e.g., `PyTorch` vs `pytorch`).
4. Add a `tests/` folder with assertions and submit an issue or PR.

## Known limitations
- Exact-string comparison misses synonyms; use normalization or a curated alias map.
- The LLM can omit or misclassify skills. Human review is essential.
- Schema compliance is not factual accuracy.
- This is educational and must not be used as the sole basis for employment decisions.

## Community challenge
Clone the repository to run locally; fork it to add your version; leave a constructive comment/issue explaining what worked, what failed, or what you improved. **Repository URL: [ADD YOUR GITHUB REPO LINK].**
