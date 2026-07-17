# Architecture — Gated RAG

How a question flows through the system, and what each component does.

---

## 1. The full data flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          STARTUP (once)                              │
│                                                                       │
│   knowledge_base.py           vector_store.py                        │
│   ┌───────────────┐           ┌──────────────────────┐               │
│   │ 16 text chunks│ ────────▶ │ embed each chunk with │               │
│   │ (DOCUMENTS)   │           │ nomic-embed-text      │               │
│   └───────────────┘           │ → 16 vectors in memory│               │
│                               └──────────────────────┘               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       PER QUESTION (every turn)                      │
│                                                                       │
│   User question                                                       │
│        │                                                              │
│        ▼                                                              │
│   ┌─────────────────────┐                                            │
│   │ embed the question   │  vector_store.embed()                     │
│   │ → 1 vector (768-dim) │                                           │
│   └──────────┬──────────┘                                            │
│              ▼                                                        │
│   ┌───────────────────────────────┐                                 │
│   │ RETRIEVE                       │  vector_store.retrieve()        │
│   │ cosine-similarity vs all 16    │                                 │
│   │ chunks → sort → keep top-3     │                                 │
│   └──────────┬────────────────────┘                                 │
│              ▼                                                        │
│   ┌───────────────────────────────┐                                 │
│   │ CONFIDENCE GATE                │  gate.check_confidence()        │
│   │ is top score ≥ 0.60 ?          │                                 │
│   └───────┬───────────────┬────────┘                                 │
│           │ YES           │ NO                                        │
│           ▼               ▼                                           │
│   ┌───────────────┐   ┌────────────────────────┐                    │
│   │ GENERATE       │   │ REFUSE                  │                    │
│   │ llama3.2:3b    │   │ "I don't have enough    │                    │
│   │ answers using  │   │  information..."        │                    │
│   │ the top chunks │   │ (+ show the score)      │                    │
│   │ generate.py    │   │ generate.refuse()       │                    │
│   └───────────────┘   └────────────────────────┘                    │
│                                                                       │
│   orchestrated by main.answer_question()                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components

| File | Role | Pipeline stage |
|---|---|---|
| `knowledge_base.py` | The documents the system knows | data |
| `vector_store.py` | Embed text, compute cosine similarity, retrieve top-k | **perceive** |
| `gate.py` | Compare top score to threshold → open or shut | **judge** |
| `generate.py` | LLM answer (gate open) or refusal (gate shut) | **act** |
| `evaluate.py` | Score answerable vs unanswerable questions to set the threshold | tuning |
| `main.py` | Orchestrate the flow, run the chat loop | orchestration |

---

## 3. The mental model

The system is a small **perceive → judge → act** loop:

- **Perceive** — retrieval turns the question into a vector and finds the
  closest chunks. Pure similarity math; no LLM.
- **Judge** — the gate inspects the best similarity score and decides whether
  the retrieved context is trustworthy. This is the "agentic" step: the system
  reasons about its own state.
- **Act** — either the LLM answers *grounded in the retrieved chunks*, or the
  system refuses honestly.

This same loop is the skeleton behind more advanced agents (routers,
self-correcting agents). Gated RAG is the simplest useful instance of it.

---

## 4. Two layers of hallucination defense

1. **The gate** blocks generation entirely when retrieval is weak
   (score < threshold).
2. **The grounding prompt** constrains the LLM to answer *only* from the
   retrieved context even when the gate opens.

Defense in depth: the gate stops the obvious failures, the prompt catches the
subtle ones.

---

## 5. Where a real system would differ (extension points)

- **Vector database** — we store 16 vectors in a Python list. Production uses
  FAISS / Chroma / Pinecone to search millions of vectors in milliseconds. The
  *logic* is identical; only the storage and search index change.
- **Chunking** — we hand-wrote clean one-idea chunks. Real documents need a
  chunking step (fixed-size or semantic) before embedding.
- **Threshold** — a single global threshold works here. Larger systems tune it
  per domain or use a small classifier instead of a fixed cutoff.
- **Re-ranking** — production RAG often adds a re-ranker after retrieval for
  sharper top-k ordering.