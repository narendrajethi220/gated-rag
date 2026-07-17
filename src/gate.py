"""
The confidence gate: decides whether retrieved context is relevant
enough to answer, based on a similarity threshold.
"""

CONFIDENCE_THRESHOLD = 0.6 # starting guess

def check_confidence(retrieved,threshold:float=CONFIDENCE_THRESHOLD):
    top_score = retrieved[0]["score"]
    passed = top_score>=threshold

    return {
        "passed":passed,
        "top_score":top_score,
        "threshold":threshold
    }

if __name__ == "__main__": # safely importing fucntion without accidently triggering the script behaviour
    from vector_store import vec_store,retrieve

    from knowledge_base import DOCUMENTS

    print("Building store...")
    store = vec_store(DOCUMENTS)

    for question in [
        "How much does the Pro plan cost?", # answerable
        "What is your refund policy?", # NOT in the knowledge base
    ]:
        retrieved=retrieve(question,store,k=3)
        result = check_confidence(retrieved)

        print(f"Q: {question}")
        print(f"top score: {result['top_score']:.3f}"
              f"(threshold {result['threshold']})"
              )
        print(f"gate: {'OPEN -> answer' if result['passed'] else 'SHUT -> refuse'}")
        print(f"best chunk: {retrieved[0]['text'][:60]}...")

