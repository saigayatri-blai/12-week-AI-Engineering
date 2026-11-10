# Day 4 — Complete RAG Pipeline

**Recording story:** ABC Technologies has 10,000 HR documents. The demo uses four fictional representative policies to make the retrieval and generation stages easy to inspect.

## Run the deterministic offline walkthrough
```bash
python day4_demo.py
python -m unittest test_day4.py -v
```
This path needs no external packages or API key. It is **a scripted retrieval/generation simulation**, not an actual vector RAG system.

## Run actual vector retrieval and LLM generation
```bash
pip install -r requirements.txt
python day4_demo.py --vector
export OPENAI_API_KEY='your_key_here'
python day4_demo.py --vector --llm
```
Do not show or commit API keys. The first model download requires internet access. OpenAI calls may incur charges.

## Recording sequence (~3 minutes)
1. Show the employee's combined question.
2. Show all four policy records, including the superseded US policy and Canadian policy.
3. Run `--vector`: show retrieved US, active source IDs and context with citations.
4. Run `--vector --llm`: show answer and uncertainty about benefits.
5. Explain that source IDs are not proof that generated claims are supported. Citation verification is Friday's evaluation topic.

**Production boundaries:** This demo has a trusted-country parameter and a simplified `access=all` retrieval filter; real systems must derive user identity, country, tenant, and permissions from trusted services. For more complex roles, the filter must be expanded. If two active authoritative policies disagree, the application should surface that conflict for review rather than select one arbitrarily. Retrieval ranking and LLM wording may vary.
