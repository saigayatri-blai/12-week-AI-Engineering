# Day 2 — Document Loading & Chunking

**Recording story:** ABC Technologies has 10,000 HR documents. We demonstrate using one representative fictional policy, not 10,000 indexed files.

## Run
```bash
python day2_demo.py
python -m unittest test_day2.py -v
```
No third-party packages are needed for the default deterministic demo. Optional:
```bash
pip install -r requirements.txt
python day2_demo.py --vector
```
`--vector` downloads a SentenceTransformers model on first run and performs ChromaDB retrieval. Use a local Python environment with network access for installation and model download.

## On-screen sequence (~2 min)
1. Show original maternity policy (benefit + eligibility + decision).
2. Run naive 90-character splitting. Highlight that no single chunk contains all three critical facts.
3. Run structure-preserving split. Highlight that the complete policy stays together.
4. Explain: a correct policy is not enough to determine an individual employee's eligibility; employee attributes are required.

**Note:** This is a deliberately simplified policy, and the fixed-size split is intentionally poor to demonstrate the failure. It does not imply that structure-aware chunking always fits every policy in one chunk. Do not claim the model necessarily hallucinates; the demo proves evidence fragmentation.
