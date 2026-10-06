# finrag — Financial Document RAG Pipeline

A production-style **Retrieval-Augmented Generation** pipeline over financial documents:
semantic chunking, dense/sparse embeddings, a FAISS vector index, a FastAPI query
service, and an offline evaluation harness for retrieval quality and answer
faithfulness.

Built to mirror real-world reg-tech RAG systems (SEC filings, compliance docs,
research notes) — the same architecture pattern used for regulatory information
retrieval in banking.

## Architecture

```
data/corpus/*.txt
        │  RecursiveCharacterTextSplitter (600 chars, 80 overlap)
        ▼
   Embedder ──► TF-IDF (default, zero downloads) or
   │            sentence-transformers/all-MiniLM-L6-v2 (optional)
        ▼
   FAISS IndexFlatIP (cosine via L2-normalized vectors)
        ▼
   ┌─────────────┐     ┌──────────────┐
   │  demo.py    │     │  api.py      │  FastAPI: /query /stats /health
   └─────────────┘     └──────────────┘
        │
   eval.py ──► recall@k, MRR, faithfulness (claim grounding)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Run the end-to-end demo (builds index, answers sample questions)
python demo.py

# 2. Serve the query API
uvicorn api:app --reload
# curl "http://localhost:8000/query?q=What+was+the+revenue+growth%3F&k=3"

# 3. Run the evaluation harness
python eval.py
```

To use transformer embeddings instead of TF-IDF:

```bash
pip install sentence-transformers
python demo.py --embedder sbert
```

## How it works

- **Chunking** — `RecursiveCharacterTextSplitter` keeps tables and paragraphs
  intact where possible; every chunk carries `{doc_id, title, chunk_id}` metadata.
- **Embeddings** — `src/embeddings.py` exposes one `Embedder` interface.
  Default is TF-IDF (runs anywhere, no model download); pass `sbert` for
  `all-MiniLM-L6-v2` dense embeddings.
- **Retrieval** — FAISS `IndexFlatIP` over L2-normalized vectors = cosine
  similarity. `src/retriever.py` loads the persisted index + metadata.
- **Answering** — extractive: the best chunk is returned with its source
  citation. Swap in any LLM in `api.py::synthesize()` for abstractive answers.
- **Evaluation** — `eval.py` reports **recall@k**, **MRR** on the labeled
  `eval/questions.jsonl` set, and a **faithfulness** score: each sentence of the
  answer must be lexically grounded in a retrieved chunk.

## Sample output

```
Q: What drove the increase in net interest income?
→ [fincorp_10k_2024.txt :: chunk 3 | score 0.83]
  "Net interest income grew 14.2% year-over-year, driven by a 38bp
   expansion in net interest margin as deposit costs repriced slower
   than the loan book..."
faithfulness: 1.00 | recall@3 (eval set): 0.92
```

## Notes

- The corpus under `data/corpus/` is **synthetic sample data** written for this
  demo — no real company filings are included.
- Designed to run fully offline with the default TF-IDF backend.

## License

MIT — see [LICENSE](LICENSE).
