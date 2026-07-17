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

---

## Tech stack

| Piece | Choice |
|---|---|
| LLM | `llama3.2:3b` (via Ollama) |
| Embeddings | `nomic-embed-text` (via Ollama) |
| Math / retrieval | `numpy` (written by hand, no vector-DB library) |

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
│   └── main.py            ← ties it all together, run this
├── data/                  ← (room for real documents later)
└── docs/                  ← notes, architecture diagram, step-by-step READMEs
```

---

## Quick start

```bash
# 1. Install Ollama models (one time)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run
python src/main.py
```

---

## Status

Built step by step as a learning project. See `docs/` for the reasoning
behind each part.