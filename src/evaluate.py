from knowledge_base import DOCUMENTS
from vector_store import vec_store, retrieve

# (question, should_be_answerable)
TEST_QUESTIONS = [
    ("How much is the Pro plan?", True),
    ("What are the data center locations?", True),
    ("How do I reset my password?", True),
    ("Is there a free trial?", True),
    ("What encryption do you use?", True),
    ("How many team members can I invite?", True),

    ("What is your refund policy?", False),
    ("What's the weather today?", False),
    ("Who is the CEO?", False),
    ("Can I pay with cryptocurrency?", False),
    ("Do you offer phone cases?", False),
]

def main():
    store = vec_store(DOCUMENTS)

    print(f"{'score':>6} {'label':>12} question")
    print("-"*55)

    answerable_scores=[]
    unanswerable_scores=[]

    for question, is_answerable in TEST_QUESTIONS:
        top = retrieve(question,store,k=1)[0]["score"]
        label = "answerable" if is_answerable else "unanswerable"
        print(f"{top:>6.3f} {label:>12} {question}")

        (answerable_scores if is_answerable else unanswerable_scores).append(top)

    print("-"*55)
    print(f"answerable  :min={min(answerable_scores):.3f} "
          f"max={max(answerable_scores):.3f}")
    print(f"unanswerable : min={min(unanswerable_scores):.3f}  "
          f"max={max(unanswerable_scores):.3f}")   
    
    # ideal threshold sits in the GAP between the two groups.

    gap_low  = max(unanswerable_scores)
    gap_high = min(answerable_scores)
    print(f"\nSeparation gap: {gap_low:.3f} (worst anwerable-miss) "
          f"-> {gap_high:.3f} (worst answerable-hit)")

    if gap_high > gap_low:
        suggested = (gap_low+ gap_high)/2
        print(f"Clean separation. Suggested threshold = {suggested:.3f}")

    else:
        print("Overlab! No single threshold separates them perfectly."
              "Pick the value that minimizes mistakes.")

if __name__ == "__main__":
    main()