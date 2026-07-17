"""
Entry point. Ties knowledge_base -> vector_store -> gate -> LLM together.
"""
from knowledge_base import DOCUMENTS
from vector_store import vec_store,retrieve
from gate import check_confidence
from generate import generate_ans, refuse

def answer_question(question,store):
    # retrieving most relevant chunks
    retrieved=retrieve(question,store,k=3)

    # gate: are they relevant enough?
    verdict = check_confidence(retrieved)

    # acting on verdict

    if verdict["passed"]:
        context_chunks = [r["text"] for r in retrieved]
        return generate_ans(question,context_chunks)
    else:
        return refuse(verdict["top_score"],verdict["threshold"])
    
    
    
if __name__ == "__main__":
    print("Building knowledge base...")
    store = vec_store(DOCUMENTS)
    print("Ready. Ask a question (or type 'quit').\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit","exit","q"}:
            break
        if not question:
            continue
        print(f"\nBot: {answer_question(question,store)}\n")