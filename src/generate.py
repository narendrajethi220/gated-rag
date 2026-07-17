import ollama

LLM_MODEL = "llama3.2:3b"

def build_prompt(question,context_chunks):

    context="\n\n".join(chunk for chunk in context_chunks)
    
    return f"""You are a support assistant for Nimbus Cloud.
Answer the user's question using ONLY the context below.
Speak about Nimbus Cloud in the third person (e.g. "Nimbus Cloud offers...").
If the context does not contain the answer, say you don't have that information.

    Context:{context}

    Question: {question}

    Answer:"""

def generate_ans(question,context_chunks):
    prompt = build_prompt(question,context_chunks)
    res=ollama.chat(
        model=LLM_MODEL,
        messages=[{"role":"user","content":prompt}],
    )
    return res["message"]["content"]

def refuse(top_score,threshold):
    return(
        "I don't have enough information in my knowledge base to answer "
        f"that confidently. (best match {top_score:.2f} < required "
        f"{threshold:.2f})"
    )