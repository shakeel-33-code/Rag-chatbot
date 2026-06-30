# RAG Retrieval Upgrade Plan

## Phase 0: Codebase Audit

### Current architecture summary

Verified from code:

| Area | Current implementation |
|---|---|
| Backend framework | FastAPI in `backend/main.py` |
| Frontend | Static single-file HTML in `frontend/index.html` |
| Upload endpoint | `POST /upload` |
| Chat endpoint | `POST /chat` |
| Health endpoint | `GET /health` |
| Supported ingestion type | PDF only. `/upload` rejects non-`.pdf` filenames. |
| PDF parsing | `pdfplumber.open(io.BytesIO(file_bytes))`; page text is concatenated into one string. |
| Chunking | `chonkie.TokenChunker(chunk_size=400, chunk_overlap=50)` from `backend/config.py`. |
| Embedding model | `EMBED_MODEL`, default `BAAI/bge-large-en-v1.5`. |
| Embedding provider | OpenAI-compatible embeddings only when embedding/OpenAI config indicates it; otherwise Hugging Face `InferenceClient.feature_extraction`. |
| Vector database | ChromaDB `PersistentClient`; default paths `backend/chroma_db` or `backend/chroma_db_recovered`. |
| Collection naming | `rag_docs_<normalized_embed_model>` unless `CHROMA_COLLECTION_NAME` is set. |
| Retrieval method | Dense vector search only via `collection.query(query_embeddings=[...], n_results=top_k, include=["documents","metadatas","distances"])`. |
| Context assembly | Retrieved chunks are concatenated with `\n\n---\n\n` until `MAX_CTX_TOKENS=1800` estimated tokens. |
| Metadata schema | Ingested chunks store only `{"source": filename}`. Retrieval adds transient rank, distance, chars, token estimate, and included flag for tracing only. |
| User/document filtering | None. No auth, no `user_id`, no `document_id`, no Chroma `where` filter in current retrieval. |
| LLM flow | `/chat` strips the query, retrieves context, builds a system message containing the context, appends last 6 history messages, calls OpenAI-compatible chat completions, strips the response. |
| LLM model | `LLM_MODEL`, default `Qwen/Qwen2.5-7B-Instruct`, with configured fallback models. |
| Evaluation | RAGAS-oriented code exists under `backend/evaluation`, plus CLI wrappers `evaluation.py` and `evaluation_single.py`. Local venv did not have `ragas` or `deepeval` installed during this audit. |
| Tests | No project test files found outside `.venv`. Existing evaluation scripts are not unit/integration tests. |
| Env/config | `.env.example`, `requirements.txt`, runtime settings in `backend/config.py`, live `/settings` endpoint. |
| Docker | No Dockerfile or compose file found. |
| Observability | Phoenix/OpenInference tracing in `backend/observability.py`; `.env` controls collector endpoint and content capture. |

### Current limitations

- Retrieval is dense-only; there is no sparse keyword/BM25 search.
- Chroma is used only as a vector store; current code does not implement BM25 or hybrid fusion.
- No mandatory user isolation. This is the highest-risk production gap.
- Metadata is too thin for document governance, filtering, page citations, deletion, or audit.
- PDF ingestion loses page-level provenance by concatenating all pages before chunking.
- `/chat` returns `context_used` as a string, not structured chunks with ids, page numbers, scores, or metadata.
- No document model, ingestion job model, file checksum, parser version, chunk version, or embedding version.
- No reranking, query rewriting, parent-child retrieval, or citation formatting.
- No API authentication/authorization.
- No automated test suite for ingestion, retrieval, filtering, or endpoint contracts.
- Observability can try to export to Phoenix when no collector is running, producing noisy connection failures.

## Target Architecture

```text
User Question
 -> Optional Intent Agent / Query Rewriter
 -> Qdrant Dense Vector Search
 -> Qdrant SPLADE Sparse Vector Search
 -> Metadata Filtering
 -> Fusion Ranking
 -> Cross-Encoder Reranking
 -> Final Context
 -> LLM Answer
```

Mandatory security invariant:

```sql
WHERE metadata.user_id = current_user_id
```

In Qdrant, this must be enforced as a payload filter on `payload.user_id`. Every retriever path must apply user-level filtering before final context assembly. No user should retrieve another user's documents.

## Proposed Module Structure

```text
backend/
  api/
    routes_upload.py
    routes_chat.py
    routes_evaluation.py
  ingestion/
    loaders.py
    normalize.py
    chunkers.py
    pipeline.py
    metadata.py
  retrieval/
    dense.py
    sparse.py
    hybrid.py
    fusion.py
    rerank.py
    filters.py
    schemas.py
  generation/
    prompts.py
    llm.py
  evaluation/
    datasets.py
    runner.py
    metrics.py
    report.py
  storage/
    vector_store.py
    document_store.py
    qdrant_store.py
  config.py
```

Keep the current files as compatibility wrappers during migration, then move logic behind stable service interfaces.

## Multi-Format Ingestion Design

Target formats:

- PDF: `pdfplumber` or `pymupdf` for text, page numbers, tables where possible.
- DOCX: `python-docx`.
- XLSX: `openpyxl` or `pandas`.
- CSV: Python `csv` or `pandas`.
- TXT/MD: native text reader with encoding detection.

Normalize all loaders into a common `DocumentUnit`:

```python
{
    "text": "...",
    "source_file_name": "...",
    "file_type": "pdf|docx|xlsx|csv|txt",
    "page_number": 12,
    "sheet_name": null,
    "row_range": null,
    "section_title": "...",
    "metadata": {...}
}
```

## Metadata Schema and User Isolation

Minimum stored chunk metadata:

```json
{
  "user_id": "required",
  "tenant_id": "optional",
  "document_id": "required",
  "source": "filename.pdf",
  "file_type": "pdf",
  "page_number": 12,
  "section_title": "optional",
  "chunk_id": "required",
  "parent_chunk_id": "optional",
  "chunk_index": 42,
  "chunk_type": "child|parent",
  "parser": "pdfplumber",
  "chunker": "token|semantic|hierarchical",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "created_at": "iso8601"
}
```

Retrieval must require a filter object:

```python
where = {"user_id": current_user_id}
```

For the Qdrant target, use a payload filter on every dense, sparse, hybrid, and reranked retrieval path. During the temporary Chroma compatibility phase, keep the equivalent `where={"user_id": current_user_id}` behavior if Chroma remains enabled. Add tests that seed two users' documents and assert zero cross-user leakage.

## Chunking Strategy Design

Use layered chunking:

- Atomic text units: page, paragraph, table row/block, list item.
- Parent chunks: larger semantic sections for answer context.
- Child chunks: smaller retrievable units optimized for similarity and keyword matching.
- Store parent-child links: retrieve child chunks, expand to parent context.
- Preserve page and section metadata on every chunk.

Initial pragmatic path:

1. Keep token chunking as baseline.
2. Add page-aware chunking for PDFs.
3. Add semantic boundary detection using headings, paragraph breaks, and token budget.
4. Add parent-child retrieval with child embeddings and parent context expansion.

## Hybrid Retrieval Design

Decision: switch the target production retrieval store to Qdrant and implement native hybrid retrieval with dense embeddings plus SPLADE sparse vectors.

Chosen strategy:

```text
Option C: Qdrant Dense + SPLADE Sparse Vectors
```

Why this is the target:

- Qdrant is better long-term retrieval infrastructure than the current embedded Chroma setup.
- Qdrant can store dense vectors, sparse vectors, and metadata payloads in the same retrieval system.
- This avoids maintaining Chroma plus a separate BM25 side index as two independent retrieval stores.
- It gives a cleaner path to native hybrid retrieval, RRF-style fusion, payload filtering, reranking, and operational scaling.

Tradeoff:

- Higher migration cost.
- Additional operational dependency.
- Requires re-ingestion and payload schema hardening before production use.

Dense search:

- Use Qdrant dense vector search with the existing embedding model as the first migration milestone.
- Store each chunk as a Qdrant point with a dense vector, sparse vector, and payload.
- Return structured `RetrievedChunk` objects with id, content, metadata payload, dense score, sparse score, fusion score, and optional rerank score.

Sparse search:

- Current code has no BM25 dependency and no BM25 index.
- `rank_bm25` is not installed in the local venv.
- Chroma is not used in this codebase for native BM25 and should not be assumed to provide production BM25.
- The target sparse representation is SPLADE sparse vectors stored in Qdrant, not a separate `rank-bm25` index.
- SPLADE gives neural sparse retrieval with exact-term sensitivity and learned term expansion.

Rejected or deferred options:

1. Keep Chroma for vectors and add a service-layer BM25 index using `rank-bm25`.
   - Lower migration cost, but creates two indexes to keep consistent.
   - Deferred because the target is native hybrid retrieval in Qdrant.
2. Keep Chroma and add separate keyword search with SQLite FTS5 or OpenSearch.
   - Stronger keyword filtering/search than in-memory BM25.
   - More moving parts and more synchronization work.
   - Deferred unless Qdrant sparse retrieval underperforms.

Selected option:

3. Switch to Qdrant for native hybrid retrieval with dense vectors plus SPLADE sparse vectors.
   - Better long-term retrieval infrastructure.
   - Higher migration cost and operational dependency.
   - Preferred production direction.

Qdrant point design:

```json
{
  "id": "chunk_id",
  "vector": {
    "dense": [0.01, 0.02],
    "sparse": {
      "indices": [123, 456],
      "values": [0.8, 0.4]
    }
  },
  "payload": {
    "user_id": "required",
    "tenant_id": "optional",
    "document_id": "required",
    "chunk_id": "required",
    "parent_chunk_id": "optional",
    "source": "filename.pdf",
    "file_type": "pdf",
    "page_number": 12,
    "section_title": "optional",
    "chunk_index": 42,
    "chunk_type": "child",
    "content": "chunk text",
    "dense_model": "BAAI/bge-large-en-v1.5",
    "sparse_model": "splade-model-name",
    "parser": "pdfplumber",
    "chunker": "semantic|hierarchical|token",
    "created_at": "iso8601"
  }
}
```

Required Qdrant payload filter:

```json
{
  "must": [
    {
      "key": "user_id",
      "match": {
        "value": "current_user_id"
      }
    }
  ]
}
```

Recommended migration path:

1. Add a vector-store interface while keeping current Chroma behavior available.
2. Add Qdrant dense-only retrieval behind the interface.
3. Re-ingest source documents into Qdrant with the new metadata schema.
4. Enforce Qdrant payload filtering on `user_id` for every dense search.
5. Add SPLADE sparse vector generation during ingestion.
6. Store both dense and sparse vectors in Qdrant.
7. Add Qdrant hybrid retrieval using dense and sparse prefetches.
8. Fuse candidates using RRF.
9. Add cross-encoder reranking.
10. Compare Chroma dense baseline vs Qdrant dense vs Qdrant hybrid + rerank on the golden dataset.

Migration rule:

Do not migrate old weak metadata blindly. The current Chroma chunks only store `source`, which is not enough for user isolation, page citations, deletion, document filtering, or audit. Re-ingest documents with the new metadata schema instead of bulk-copying weak Chroma records into Qdrant.

Target configuration:

```env
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_chunks
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_DISTANCE=Cosine
QDRANT_DENSE_MODEL=BAAI/bge-large-en-v1.5
QDRANT_SPARSE_MODEL=splade-model-name
```

The exact SPLADE model should be selected during implementation based on latency, license, model size, and benchmark quality. Record the chosen model in every Qdrant point payload as `sparse_model`.

## Fusion Ranking Design

Normalize dense and sparse candidate lists into:

```python
{
  "chunk_id": "...",
  "dense_rank": 3,
  "sparse_rank": 1,
  "dense_score": 0.73,
  "sparse_score": 8.1,
  "metadata": {...}
}
```

Start with Reciprocal Rank Fusion:

```text
rrf_score = sum(1 / (k + rank_i))
```

Use `k=60` as a conventional default, then tune on the golden dataset. Weighted score fusion can follow after score calibration.

## Reranking Design

Add cross-encoder reranking after fusion:

- Candidate count before rerank: 20-50.
- Final context count: 3-8, bounded by token budget.
- Candidate model options: `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, or a hosted reranker.
- Store `rerank_score` in retrieval results and endpoint debug/eval outputs.

Reranking should run after mandatory user filtering and before context expansion to keep latency controlled.

## Intent Agent / Query Rewriter Design

Only add after hybrid retrieval and reranking have been benchmarked.

Inputs:

- Raw question.
- Optional chat history.
- User/document scope.

Outputs:

```json
{
  "retrieval_query": "...",
  "filters": {"document_id": "..."},
  "intent": "definition|comparison|summary|factual|reasoning",
  "needs_clarification": false
}
```

Example:

```text
Raw question:
Can you tell me what this document says about chunking methods?

Retrieval query:
chunking methods semantic chunking hierarchical chunking fixed-size chunking document splitting strategies
```

Guardrails:

- The rewriter may expand retrieval terms but must not remove `user_id` filters.
- Log raw and rewritten query for eval.
- Fall back to raw query if rewrite fails.

## Evaluation and Benchmarking Design

Keep two layers:

- Deterministic CI checks: hit rate, Recall@k, MRR, context precision/recall, latency, error rate, keyword coverage.
- Optional LLM-as-judge: faithfulness, answer relevance, groundedness, hallucination classification.

Evaluation artifacts:

```text
eval/golden_dataset/
eval/results/
eval/fixtures/
```

For every retrieval run, save:

- question id
- expected answer
- actual answer
- retrieved chunk ids
- scores
- page numbers
- metadata
- latency
- status/error

## Migration Plan and Phase Roadmap

### Phase 0: Codebase audit

Goal: Establish verified baseline.

Files affected: docs only.

New files needed:

- `docs/rag-retrieval-upgrade-plan.md`
- `docs/rag-evaluation-report.md`
- `eval/golden_dataset/*`
- `eval/results/*`

Implementation steps:

- Inspect FastAPI routes, ingestion, retrieval, LLM, config, evaluation code.
- Generate golden dataset.
- Run endpoint evaluation if provider config allows.

Test cases:

- JSONL dataset parses.
- `/health` and `/chat` endpoint contract verified.

Acceptance criteria:

- Current architecture documented from code.
- Evaluation artifacts saved.
- Endpoint failures documented without fabricated results.

### Phase 1: Ingestion normalization

Goal: Support PDF, DOCX, XLSX, CSV, TXT through a common normalized document unit.

Files affected:

- `backend/ingest.py`
- `backend/main.py`

New files needed:

- `backend/ingestion/loaders.py`
- `backend/ingestion/normalize.py`
- `backend/ingestion/pipeline.py`

Implementation steps:

- Add loader registry by MIME type and extension.
- Preserve page/sheet/row provenance.
- Return structured units before chunking.
- Keep `/upload` backward compatible.

Test cases:

- Upload each supported file type.
- Reject unsupported types.
- Verify page/sheet metadata survives ingestion.

Acceptance criteria:

- All target file types produce normalized units with provenance.
- Existing PDF upload still works.

### Phase 2: Metadata schema + user filtering

Goal: Enforce user isolation everywhere.

Files affected:

- `backend/main.py`
- `backend/ingest.py`
- `backend/retriever.py`

New files needed:

- `backend/retrieval/filters.py`
- `backend/ingestion/metadata.py`

Implementation steps:

- Add auth/current-user dependency or explicit temporary user parameter for local dev.
- Require `user_id` at ingestion.
- Store `user_id`, `document_id`, `chunk_id`, `page_number`, `file_type`.
- Add Qdrant payload filters equivalent to `payload.user_id = current_user_id` to every vector query.

Test cases:

- User A cannot retrieve User B chunks.
- Missing user id fails closed.
- Document-specific filter only returns that document.

Acceptance criteria:

- Every target Qdrant retrieval path applies `payload.user_id = current_user_id`.
- Cross-user leakage test passes.

### Phase 3: Chunking upgrade

Goal: Improve retrieval quality for large documents.

Files affected:

- `backend/ingest.py`

New files needed:

- `backend/ingestion/chunkers.py`
- `backend/retrieval/context_expansion.py`

Implementation steps:

- Add page-aware chunking.
- Add semantic split by headings/paragraphs.
- Add parent-child chunk records.
- Store chunk lineage and token counts.

Test cases:

- Chunk boundaries preserve section/page metadata.
- Child chunks point to valid parent chunks.
- Context expansion returns parent text for child hits.

Acceptance criteria:

- Page citations are available for every retrieved chunk.
- Long-document retrieval improves or holds Recall@k against baseline.

### Phase 4: Qdrant hybrid retrieval

Goal: Switch the target retrieval store to Qdrant and combine dense semantic search with SPLADE sparse vector search.

Files affected:

- `backend/retriever.py`
- `backend/ingest.py`
- `backend/config.py`
- `requirements.txt`

New files needed:

- `backend/retrieval/dense.py`
- `backend/retrieval/sparse.py`
- `backend/retrieval/hybrid.py`
- `backend/retrieval/fusion.py`
- `backend/storage/qdrant_store.py`
- `backend/storage/vector_store.py`
- `scripts/reingest_to_qdrant.py`

Implementation steps:

- Add a vector-store interface so Chroma and Qdrant can coexist during migration.
- Add Qdrant collection creation with named dense and sparse vectors.
- Add Qdrant dense-only retrieval first to establish parity with the current Chroma baseline.
- Re-ingest source documents into Qdrant with the new metadata schema. Do not bulk-copy old weak Chroma metadata.
- Generate dense embeddings using the configured embedding model.
- Generate SPLADE sparse vectors during ingestion and store them as Qdrant sparse vectors.
- Apply Qdrant payload filters, especially `user_id`, to dense and sparse retrieval paths.
- Add hybrid retrieval with dense and sparse candidate prefetches.
- Fuse dense and sparse candidates using RRF.
- Return structured candidates with dense rank, sparse rank, dense score, sparse score, fusion score, and payload metadata.

Test cases:

- Exact keyword questions retrieve exact chunks.
- Semantic paraphrases still retrieve dense hits.
- Hybrid beats dense-only on keyword-sensitive questions.
- User A cannot retrieve User B chunks through dense, sparse, or hybrid retrieval.
- Re-ingested Qdrant points contain required payload fields: `user_id`, `document_id`, `chunk_id`, `page_number`, `content`, `dense_model`, `sparse_model`.
- Old Chroma records with only `source` metadata are rejected by the migration/re-ingestion validation.

Acceptance criteria:

- Hybrid Recall@k exceeds dense-only baseline on golden dataset.
- User filter applies to dense, sparse, and fused Qdrant branches.
- Qdrant hybrid retrieval is the selected production target.
- Chroma remains only as a temporary compatibility fallback until Qdrant evaluation passes.

### Phase 5: Reranking

Goal: Improve final context precision.

Files affected:

- `backend/retriever.py`

New files needed:

- `backend/retrieval/rerank.py`

Implementation steps:

- Add cross-encoder reranker interface.
- Rerank top fused candidates.
- Return `rerank_score`.
- Add latency budget and fallback.

Test cases:

- Reranker preserves user filters.
- Reranker improves context precision on benchmark.
- Timeout falls back to fused ranking.

Acceptance criteria:

- Context precision improves without unacceptable latency.
- `/chat` exposes structured retrieval metadata in eval/debug mode.

### Phase 6: Evaluation benchmark

Goal: Make RAG quality measurable and repeatable.

Files affected:

- `backend/evaluation/*`

New files needed:

- `backend/evaluation/datasets.py`
- `backend/evaluation/runner.py`
- `eval/results/*.jsonl`

Implementation steps:

- Add JSONL golden dataset loader.
- Evaluate endpoint and retriever separately.
- Save per-question results.
- Add deterministic metrics; use RAGAS/DeepEval only when installed/configured.

Test cases:

- Dataset schema validation.
- Endpoint evaluation handles 500s without aborting.
- Metrics are reproducible.

Acceptance criteria:

- CI can run a small deterministic RAG benchmark.
- Reports include hit rate, Recall@k, MRR, context precision/recall, latency, error rate.

### Phase 7: Intent Agent / Query Rewriter

Goal: Improve weak retrieval queries after hybrid search is established.

Files affected:

- `backend/main.py`
- `backend/retriever.py`

New files needed:

- `backend/retrieval/query_rewrite.py`

Implementation steps:

- Add rewrite interface.
- Log raw and rewritten query.
- Evaluate raw vs rewritten retrieval.
- Add fallback to raw query.

Test cases:

- Rewriter expands vague questions.
- Rewriter does not remove security filters.
- Retrieval improves for weak natural-language questions.

Acceptance criteria:

- Query rewriting improves benchmark retrieval without increasing hallucinations.

### Phase 8: Production hardening

Goal: Prepare for real multi-user production use.

Files affected:

- API routes, config, ingestion, retrieval, evaluation, deployment files.

New files needed:

- `Dockerfile`
- `docker-compose.yml`
- CI workflow
- migration scripts

Implementation steps:

- Add auth and authorization.
- Add async ingestion jobs and status endpoints.
- Add document deletion and reindexing.
- Add rate limits and request size limits.
- Add structured logs and tracing fallback.
- Add Docker and deployment docs.
- Add Qdrant backup/restore and collection migration runbooks.

Test cases:

- Load tests for concurrent chat/retrieval.
- Ingestion retry/idempotency tests.
- Deletion removes Qdrant dense/sparse vector points and document records.
- Observability disabled mode produces no collector noise.

Acceptance criteria:

- Production deployment can run from documented config.
- Security, retrieval, ingestion, and evaluation checks pass in CI.

## Risks and Tradeoffs

- Qdrant dense + SPLADE sparse retrieval is cleaner long-term but requires data migration, re-ingestion, SPLADE sparse-vector generation, and Qdrant operations.
- Chroma plus service-layer BM25 is now a fallback option only; it is faster to adopt but creates two indexes to keep consistent.
- SPLADE sparse vectors improve lexical matching and term expansion but add ingestion cost and model dependency.
- Do not migrate old weak metadata blindly. Re-ingest with the new metadata schema so Qdrant payload filters, page citations, deletion, and audit all work correctly.
- Cross-encoder reranking improves precision but adds latency and another model dependency.
- Intent rewriting can improve retrieval but may over-expand or drift; evaluate only after hybrid baseline.
- Full-document ingestion of large PDFs can be slow and expensive with remote embeddings; add ingestion jobs, batching, and cost controls.
- Current Phoenix config can emit connection errors when no collector is running; production should fail quiet when disabled/unavailable.

## Testing Strategy

- Unit tests for loaders, metadata creation, chunking, filter construction, fusion, reranking.
- Integration tests with isolated Qdrant collections using dense and SPLADE sparse vectors.
- Security tests with two users and overlapping document text.
- Contract tests for `/upload`, `/chat`, `/settings`, `/evaluation/run`.
- Migration/re-ingestion tests that reject chunks missing required Qdrant payload fields.
- Golden dataset regression tests for retrieval and generation.
- Latency and error-rate thresholds in evaluation reports.

## Acceptance Criteria for the Upgrade

- Multi-format ingestion works for PDF, DOCX, XLSX, CSV, TXT.
- Every Qdrant point stores `user_id`, `document_id`, `chunk_id`, provenance, `dense_model`, and `sparse_model`.
- Every retrieval path enforces `payload.user_id = current_user_id` in Qdrant.
- Qdrant dense + SPLADE sparse hybrid retrieval improves benchmark Recall@k over dense-only baseline.
- Reranking improves context precision without violating latency SLOs.
- `/chat` can return structured retrieval metadata for evaluation/debug use.
- Automated evaluation produces reproducible JSONL and Markdown reports.
- No cross-user retrieval is possible in tests.
