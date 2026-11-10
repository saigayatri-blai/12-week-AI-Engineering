# Day 5 — RAG Evaluation & Optimization

This source-aligned demo reproduces the recorded fictional scenario:

- an employee has two months of tenure;
- the question asks about 16 weeks of parental leave;
- one gold chunk describes the benefit;
- a second gold chunk contains the 12-month eligibility requirement;
- Recall@1 and Recall@3 are calculated from the actual ChromaDB ranking.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python BuildLifeandAI_Week4_Day5_Evaluation_Demo.py
python -m unittest test_day5.py -v
```

The verified local run produced Recall@1 = 0.50 and Recall@3 = 1.00, followed by the grounded answer that the fictional two-month employee has not met the 12-month tenure requirement. These values are calculated from displayed retrieval IDs for this one example; they are not a benchmark or a claim that increasing Top-K always improves quality.
