"""
Embedding + retrieval. Turns text into vectors and finds the most
relevant chunks for a question using cosine similarity.
"""

import ollama
import numpy as np
from knowledge_base import DOCUMENTS

EMBED_MODEL = "nomic-embed-text"


def embed(text):
    res=ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return np.array(res["embedding"])

def vec_store(documents):
    store=[]
    for doc in documents:
        store.append({"text":doc,"vector":embed(doc)})
    return store

def cosine_sim(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
# dividing the dot prod by the two len cancel out magnitude and leaves pure direction - a num from -1 to 1 where 1 means identical meaning.
# we care about the direction, because two chunks about the same topic point the same way even if one is longer than the other

def retrieve(question,store,k:int=3):
    q_vec=embed(question)

    scored=[]

    for record in store:
        score = cosine_sim(q_vec,record["vector"]) # comparing against every stored chunk

        scored.append({"text":record["text"],"score":score})

    scored.sort(key=lambda r:r["score"],reverse=True) # storing all chunks from most to least similar
    
    return scored[:k]


