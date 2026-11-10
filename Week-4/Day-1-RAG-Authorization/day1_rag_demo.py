"""BuildLifeandAI Week 4 Day 1: simple RAG + authorization demonstration.
Run: pip install -r requirements.txt && python day1_rag_demo.py
Optional LLM: set OPENAI_API_KEY to enable generated answers.
All company data and users are fictional.
"""
import os
from dataclasses import dataclass

DOCUMENTS = [
    {"id": "hr_remote", "text": "ABC Technologies employees may work remotely up to three days per week, subject to manager approval.", "access": "all"},
    {"id": "hr_leave", "text": "ABC Technologies employees receive 20 days of annual leave per year.", "access": "all"},
    {"id": "hr_ceo", "text": "ABC Technologies CEO's annual base salary is $534,000.", "access": "hr_leadership"},
]

@dataclass(frozen=True)
class Principal:
    name: str
    role: str  # Demo identity: a real app must derive this from trusted auth.

EMPLOYEE = Principal("New employee", "employee")
HR_LEADER = Principal("Authorized HR leader", "hr_leadership")


def create_collection():
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()  # in-memory: fresh collection each run
    collection = client.create_collection("abc_hr_day1")
    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[{"access": doc["access"]} for doc in DOCUMENTS],
        embeddings=model.encode([doc["text"] for doc in DOCUMENTS]).tolist(),
    )
    return collection, model


def search(collection, model, question, principal=None):
    """Filter before retrieval; principal=None is intentionally insecure."""
    options = {}
    if principal is not None:
        if principal.role not in {"employee", "hr_leadership"}:
            raise ValueError("Unknown role; deny by default")
        allowed = ["all", "hr_leadership"] if principal.role == "hr_leadership" else ["all"]
        options["where"] = {"access": {"$in": allowed}}
    result = collection.query(
        query_embeddings=model.encode([question]).tolist(),
        n_results=3 if principal is None or principal.role == "hr_leadership" else 2,
        **options,
    )
    return list(zip(result["ids"][0], result["documents"][0]))


def grounded_answer(question, retrieved):
    """Offline demo: deterministic answer selection, NOT an LLM."""
    q = question.lower()
    ids = {doc_id: text for doc_id, text in retrieved}
    if "ceo" in q and ("salary" in q or "compensation" in q):
        return ids.get("hr_ceo", "I couldn't find that information in the documents available to you.")
    if "remote" in q or "work from home" in q:
        return ids.get("hr_remote", "No authorized supporting document found.")
    if "annual leave" in q:
        return ids.get("hr_leave", "No authorized supporting document found.")
    return "No verified answer available from the retrieved documents."


def optional_llm_answer(question, retrieved):
    """Optional actual generation. Uses ONLY authorized retrieved context."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    from openai import OpenAI
    context = "\n".join(f"[{i}] {t}" for i, t in retrieved)
    response = OpenAI().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        instructions=("Use only supplied context. If it does not answer the question, "
                      "say 'I couldn't find that information in the documents available to you.' "
                      "Do not guess or reveal unavailable information."),
        input=f"Authorized context:\n{context}\n\nQuestion: {question}",
    )
    return response.output_text


def main():
    collection, model = create_collection()
    salary_question = "What is the CEO's annual salary?"
    print("\n1. NORMAL HR QUESTION")
    normal = search(collection, model, "Can I work from home?", EMPLOYEE)
    print(grounded_answer("Can I work from home?", normal))

    print("\n2. SECURITY BUG: NO AUTHORIZATION FILTER")
    insecure = search(collection, model, salary_question)
    print(grounded_answer(salary_question, insecure))

    print("\n3. FIX: REGULAR EMPLOYEE")
    employee_docs = search(collection, model, salary_question, EMPLOYEE)
    assert "hr_ceo" not in [doc_id for doc_id, _ in employee_docs]
    print(grounded_answer(salary_question, employee_docs))

    print("\n4. FIX: AUTHORIZED HR LEADER")
    leader_docs = search(collection, model, salary_question, HR_LEADER)
    assert "hr_ceo" in [doc_id for doc_id, _ in leader_docs]
    print(grounded_answer(salary_question, leader_docs))

    if os.getenv("OPENAI_API_KEY"):
        print("\n5. OPTIONAL LLM GENERATION (authorized context only)")
        print("Employee:", optional_llm_answer(salary_question, employee_docs))
        print("HR leader:", optional_llm_answer(salary_question, leader_docs))
    print("\nPASS: Restricted CEO document was excluded for the employee.")


if __name__ == "__main__":
    main()
