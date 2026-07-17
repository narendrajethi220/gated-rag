# Gated RAG — Agentic RAG with a Confidence Gate

> A retrieval system that knows when it doesn't know.

Most RAG systems hallucinate because they never ask themselves whether they
actually found the answer. **Gated RAG** adds a *confidence gate*: before it
lets the LLM answer, it checks whether the retrieved context is relevant
enough. If it isn't, the system refuses instead of making something up.

Runs **fully local** with [Ollama](https://ollama.com) — no API keys, no data
leaves your machine.

---

## The core idea

```
Question → Embed → Retrieve top-k → [ CONFIDENCE GATE ] → Answer  (if relevant)
                                            └──────────→ Refuse  (if not)
```

The gate is driven by **cosine similarity**. If the best retrieved chunk
scores above a threshold, we trust the context and generate an answer.
If not, we say "I don't have enough information."

See `docs/architecture.md` for the full diagram and data flow.

---

## Why it works — evaluation

Across a labeled test set of answerable vs. unanswerable questions, the two
groups separate cleanly by similarity score:

| Group | Score range |
|---|---|
| Answerable (in knowledge base) | 0.650 – 0.833 |
| Unanswerable (not in knowledge base) | 0.435 – 0.476 |

That's a **0.17 gap** with no overlap. The threshold lives inside that gap.
The data suggested ≈0.563; the project uses **0.60** to bias slightly toward
*safe refusals* — a false "I don't know" is cheaper than a confident wrong
answer.

---

## Tech stack

| Piece | Choice |
|---|---|
| LLM | `llama3.2:3b` (via Ollama) |
| Embeddings | `nomic-embed-text` (via Ollama) |
| Math / retrieval | `numpy` (hand-written, no vector-DB library) |

Retrieval and the gate are hand-written on purpose, so every mechanism is
visible instead of hidden inside a framework.

---

## Project structure

```
gated-rag/
├── README.md              ← you are here
├── requirements.txt       ← Python dependencies
├── .gitignore
├── src/                   ← all source code
│   ├── knowledge_base.py  ← the documents (our "knowledge")
│   ├── vector_store.py    ← embedding + cosine similarity + retrieval
│   ├── gate.py            ← the confidence gate logic
│   ├── generate.py        ← LLM answer + refusal branches
│   ├── evaluate.py        ← threshold tuning: scores answerable vs not
│   └── main.py            ← ties it all together, run this
├── data/                  ← room for real documents later
└── docs/                  ← architecture diagram + step-by-step build notes
```

---

## Quick start

```bash
# 1. Install Ollama models (one time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run the interactive assistant
python src/main.py

# 4. (Optional) reproduce the threshold evaluation
python src/evaluate.py
```

---

## Try these

- `How much is the Pro plan?` → answers from context
- `Where are your data centers?` → answers from context
- `What is your refund policy?` → **refuses** (not in the knowledge base)
- `What's the weather today?` → **refuses** (off-topic)

The last two are the point: it declines instead of inventing an answer.

---

## How it's built

Built step by step as a learning project. See `docs/` for the reasoning
behind each part — concept, embeddings, retrieval, the gate, generation, and
threshold tuning.