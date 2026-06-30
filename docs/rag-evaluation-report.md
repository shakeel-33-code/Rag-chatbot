# RAG Evaluation Report

## Selected ML PDF

Document: `Dive into Deep Learning`

Official PDF: https://d2l.ai/d2l-en.pdf

Local path: `eval/source_docs/d2l-en.pdf`

Size/page count observed locally: 44,685,994 bytes; 1,151 PDF pages.

License/open-access assumption: the D2L GitHub repository describes the project as open source and makes the book available under the Creative Commons Attribution-ShareAlike 4.0 International License. This makes it suitable for a reproducible RAG evaluation dataset.

Why suitable:

- It is a public ML/deep learning educational book.
- It covers definitions, comparisons, algorithms, and reasoning-heavy concepts.
- The selected pages include regression, classification, MLPs, CNNs, attention, Transformers, BERT, optimization, and Gaussian processes.

## Golden Dataset Summary

Generated files:

- `eval/golden_dataset/ml_pdf_golden_dataset.jsonl`
- `eval/golden_dataset/ml_pdf_golden_dataset.md`

Dataset size: 24 questions.

Question types:

- factual
- definition
- comparison
- conceptual
- reasoning

Difficulty mix:

- easy
- medium
- hard

The dataset was written from verified PDF page text. No PDF content was fabricated.

## Current Ingestion, Chunking, and Retrieval Behavior

Verified current behavior:

- `/upload` accepts only PDF files.
- PDF text is extracted with `pdfplumber`.
- Page text is concatenated into a single string before chunking, so chunk metadata does not preserve page numbers.
- Chunking uses `chonkie.TokenChunker` with `CHUNK_SIZE=400` and `CHUNK_OVERLAP=50`.
- Chunk metadata currently stores only `source`.
- ChromaDB stores vectors persistently.
- Retrieval is dense-only Chroma vector search with `TOP_K=3`.
- Retrieval does not apply `where` metadata filtering.
- No user isolation exists in the current endpoint or retriever.
- `/chat` returns `answer` and `context_used`, where context is a concatenated string rather than structured chunks.

## Chat Endpoint Tested

Endpoint: `POST /chat`

Request shape:

```json
{
  "question": "...",
  "history": []
}
```

Response shape observed:

```json
{
  "answer": "...",
  "context_used": "chunk 1\n\n---\n\nchunk 2..."
}
```

Evaluation output:

- `eval/results/chat_endpoint_eval_results.jsonl`

Important setup note:

- I did not index the full 1,151-page PDF through `/upload`, because that would require thousands of remote embedding calls and would modify the existing Chroma store.
- Instead, I seeded an isolated eval Chroma store at `eval/chroma_db` with page-level D2L chunks from `eval/fixtures/d2l_eval_chunks.jsonl`.
- The unchanged FastAPI `/chat` endpoint was then started with `CHROMA_DB_PATH=eval/chroma_db` and `CHROMA_COLLECTION_NAME=rag_eval_d2l`.
- This keeps the evaluation reproducible and avoids contaminating the existing `backend/chroma_db_recovered` collection.

Commands attempted:

```powershell
Invoke-WebRequest -UseBasicParsing https://d2l.ai/d2l-en.pdf -OutFile eval\source_docs\d2l-en.pdf
```

Initial result: failed in sandbox with remote connection error.

Resolution: reran with elevated network permission; download succeeded.

```powershell
$env:CHROMA_DB_PATH='D:\AI Builders Community\RAG\rag-project\eval\chroma_db'
$env:CHROMA_COLLECTION_NAME='rag_eval_d2l'
.\.venv\Scripts\python.exe -c "<seed eval Chroma collection>"
```

Initial result: failed in sandbox with `[WinError 10013]` socket permission while calling Hugging Face embeddings.

Resolution: reran with elevated network permission; eval collection seeded with 12 chunks.

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health -> 200 {"status":"ok"}
```

## Evaluation Results

Total questions sent to `/chat`: 24

Endpoint-level statuses:

| Status | Count |
|---|---:|
| pass | 14 |
| partial | 2 |
| fail | 0 |
| error | 8 |

Error rate: 33.33%

Successful or partial responses: 16/24.

Average latency, all requests: 1,257.75 ms.

Average latency, non-error requests: 1,560.57 ms.

P95 latency, non-error requests: 2,190.51 ms.

Failure blocker:

Questions `d2l_q017` through `d2l_q024` returned HTTP 500. The error body was:

```text
Error code: 402 - You have depleted your monthly included credits. Purchase pre-paid credits to continue using Inference Providers.
```

This came from the configured Hugging Face Inference Provider during LLM generation. The results file records these as `error`; no answers were fabricated.

## Retrieval Metrics

The current `/chat` endpoint does not expose chunk ids, page numbers, distances, or scores. For this report, retrieval metrics were computed by matching returned `context_used` chunks against the seeded eval fixture content. This is possible for the isolated evaluation fixture but is not a production substitute for structured retrieval metadata.

For all 24 endpoint calls:

| Metric | Value |
|---|---:|
| Hit Rate | 0.6667 |
| Recall@3 | 0.6667 |
| MRR | 0.6667 |
| Context precision | 0.2222 |
| Context recall | 0.6667 |

For the 16 non-error endpoint calls:

| Metric | Value |
|---|---:|
| Hit Rate | 1.0000 |
| Recall@3 | 1.0000 |
| MRR | 1.0000 |
| Context precision | 0.3333 |
| Context recall | 1.0000 |

Interpretation:

- Retrieval was strong on successful requests: the expected chunk appeared at rank 1 for all non-error requests.
- Context precision is low because `TOP_K=3` returns three chunks and only one expected chunk was usually required by the golden answer.
- The all-question retrieval metrics are depressed by LLM provider errors because `/chat` returns no context on HTTP 500.

## Generation Metrics

RAGAS and DeepEval were not installed in the local venv during this audit, so no RAGAS/DeepEval judge run was performed.

Lightweight deterministic checks used:

- Expected keyword coverage in the answer.
- Endpoint status.
- Presence of returned context.
- Exact fixture-context match for retrieval metrics.

Generation summary:

- 14 answers passed deterministic keyword coverage.
- 2 answers were partial.
- 8 answers could not be generated because the provider returned credit exhaustion.

Approximate generation quality for completed answers:

| Dimension | Observation |
|---|---|
| Faithfulness | Completed answers were generally grounded in returned D2L context. |
| Answer relevance | Completed answers answered the asked question directly. |
| Groundedness | Strong for completed answers because expected context was retrieved at rank 1. |
| Hallucination cases | No clear hallucination was observed in the 16 completed responses by deterministic inspection. |

Optional future LLM-as-judge design:

- Use a configured evaluator LLM separate from the production answer model.
- Judge answer faithfulness against retrieved contexts.
- Judge answer relevance against the question.
- Judge answer correctness against `expected_answer`.
- Keep deterministic retrieval metrics as the CI gate and use LLM judging as an additional report, not the only gate.

## Retrieval Quality Observations

- Dense retrieval worked well on the seeded D2L chunks.
- Keyword-sensitive questions would still be risky on large real documents because there is no BM25 branch.
- Current chunk metadata is insufficient for normal production citation, filtering, and evaluation.
- The endpoint loses structured retrieval details by returning only a context string.
- The current retriever computes structured IDs/scores internally for tracing, but does not return them to `/chat`.

## Answer Quality Observations

- The LLM produced concise, grounded answers when provider credits were available.
- The prompt includes retrieved context in the system message and instructs the assistant to use only that context for document questions.
- There is no citation requirement, so answers do not cite page numbers.
- The endpoint does not expose whether the answer used every retrieved chunk.

## Failure Cases

System failures:

- `d2l_q017` to `d2l_q024` failed with HTTP 500 due to Hugging Face Inference Provider credit exhaustion.
- Phoenix tracing attempted to export spans to `localhost:6006` even though no collector was running, causing noisy exporter failures in server logs. The `.env` value loaded by `config.py` overrode the attempted shell override.

Retrieval/generation partials:

- `d2l_q005` and `d2l_q015` were marked partial by deterministic keyword matching. The answers were not necessarily wrong; they did not contain enough expected keywords for the simple pass threshold.

Missing context cases:

- Error responses did not include context, even though retrieval may have completed before LLM failure. This prevents post-failure retrieval diagnosis from the endpoint response alone.

Hallucination cases:

- No clear hallucination was identified in completed responses using deterministic review.
- Full hallucination scoring requires a judge model or human review.

## Recommended Improvements

Highest priority:

1. Add mandatory `user_id` metadata and enforce `where={"user_id": current_user_id}` in retrieval.
2. Return structured retrieval metadata from `/chat` in eval/debug mode: `chunk_id`, `document_id`, `page_number`, `score`, `metadata`.
3. Preserve page-level provenance during PDF ingestion.
4. Add hybrid retrieval: Chroma dense search plus service-layer BM25 first, then consider Qdrant native hybrid retrieval later.
5. Add RRF fusion and cross-encoder reranking after hybrid retrieval.
6. Add deterministic eval runner that does not require RAGAS/DeepEval to be installed.
7. Add provider quota/error handling so one upstream 402/429 does not look like an application bug.

Next implementation step:

Implement Phase 2 first: metadata schema plus mandatory user filtering. Hybrid retrieval should not be built on top of a retriever that can leak data across users.

## NVIDIA Revalidation - June 23, 2026

Provider re-tested: `nvidia`

Model re-tested: `meta/llama-3.1-8b-instruct`

Vector store used for this revalidation: isolated Chroma eval store at `eval/chroma_db_nvidia_single` with collection `rag_eval_d2l_nvidia_single`

Question re-tested through the actual FastAPI `/chat` endpoint:

- `d2l_q001`
- "In D2L's linear regression section, how are predictions for an entire dataset expressed using the design matrix?"

Artifacts generated:

- `eval/results/nvidia_single_chat_eval_result.jsonl`
- `eval/results/nvidia_single_eval_report.json`
- `eval/results/nvidia_single_eval_report.md`
- `eval/results/nvidia_single_report_card.yaml`

Observed result:

- HTTP status: `200`
- Endpoint latency: `2961.25 ms`
- Overall RAG score: `0.7082`
- Pass: `false`

Metric outcome:

- Passed: `context_recall`, `context_entity_recall`, `faithfulness`, `groundedness_score`, `hallucination_rate`
- Failed: `context_precision`, `context_relevancy`, `answer_relevancy`, `answer_correctness`, `answer_similarity`, `retrieval_f1`, `overall_rag_score`

Interpretation:

- Retrieval found the right primary chunk at rank 1, but `TOP_K=3` added two irrelevant chunks, which drove `context_precision` and `retrieval_f1` down.
- The NVIDIA answer was grounded and factually aligned, but it was too compressed relative to the golden reference answer, which caused low deterministic `answer_correctness` and `answer_similarity`.
- This is a quality failure, not an availability failure. The endpoint worked; it did not clear the repo's current thresholds.
