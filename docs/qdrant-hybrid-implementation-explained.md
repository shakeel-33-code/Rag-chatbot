# Qdrant Hybrid Retrieval Implementation Explained

This document explains the Qdrant hybrid retrieval implementation added to the RAG project. It is written for maintainers who need to understand what changed, why it changed, how the code works, and what still needs to happen before full production rollout.

The implementation is intentionally incremental:

- The current ChromaDB path remains the default.
- Qdrant is enabled only when `VECTOR_DB=qdrant`.
- Qdrant mode requires stronger metadata and a `user_id` filter.
- Dense + SPLADE sparse retrieval is supported structurally.
- The code rejects weak legacy metadata instead of silently migrating it.

## Executive Summary

The original RAG system used this retrieval flow:

```text
PDF upload
 -> pdfplumber text extraction
 -> token chunks
 -> ChromaDB vector add

Question
 -> embedding
 -> ChromaDB dense vector search
 -> concatenate top-k chunks
 -> LLM
```

The new target Qdrant flow is:

```text
PDF upload
 -> page-aware text extraction
 -> token chunks with page metadata
 -> dense embeddings
 -> SPLADE sparse vectors
 -> Qdrant points with dense vector + sparse vector + payload

Question
 -> dense query embedding
 -> SPLADE sparse query vector
 -> Qdrant dense search with user_id payload filter
 -> Qdrant sparse search with user_id payload filter
 -> RRF fusion
 -> context assembly
 -> LLM
```

The most important production rule is:

```text
No Qdrant retrieval is allowed without user_id filtering.
```

That is why Qdrant ingestion/retrieval fails closed when `user_id` is missing.

## Files Changed

Main application files:

- `backend/config.py`
- `backend/ingest.py`
- `backend/retriever.py`
- `backend/main.py`
- `.env.example`
- `requirements.txt`

New retrieval/storage files:

- `backend/retrieval/schemas.py`
- `backend/retrieval/fusion.py`
- `backend/retrieval/sparse.py`
- `backend/storage/qdrant_store.py`

New tests:

- `tests/test_qdrant_planning_helpers.py`

Related planning document:

- `docs/rag-retrieval-upgrade-plan.md`

## Why Qdrant Was Added This Way

The implementation avoids a risky hard cutover. The current Chroma path is still useful as a working baseline and as a fallback while Qdrant infrastructure, sparse models, and re-ingestion are prepared.

The implementation adds Qdrant behind a config flag:

```env
VECTOR_DB=chroma
```

or:

```env
VECTOR_DB=qdrant
```

This makes the migration reversible during rollout.

The implementation also enforces strong metadata in Qdrant mode. The old Chroma chunks only stored:

```json
{"source": "filename.pdf"}
```

That is not enough for production because it cannot safely support:

- user isolation
- document filtering
- page citations
- deletion by document
- audit trails
- re-indexing
- evaluation with page-level provenance

So the Qdrant code requires metadata such as:

```json
{
  "user_id": "user-1",
  "document_id": "doc-1",
  "source": "file.pdf",
  "file_type": "pdf",
  "page_number": 1
}
```

The rule is:

```text
Do not migrate old weak metadata blindly.
Re-ingest source documents with the new metadata schema.
```

## Configuration Implementation

File: `backend/config.py`

### `VECTOR_DB`

```python
VECTOR_DB = os.getenv("VECTOR_DB", "chroma").strip().lower()
```

Meaning:

- Reads the desired vector database from the environment.
- Defaults to `chroma` so existing behavior does not break.
- Normalizes the value with `strip().lower()` so values like ` QDRANT ` still work.

Why:

- A production migration should be switchable by config.
- The existing Chroma path should remain available until Qdrant is fully validated.

### Qdrant connection settings

```python
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip() or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks").strip()
```

Meaning:

- `QDRANT_URL` points to a local or remote Qdrant service.
- `QDRANT_API_KEY` is optional for local development but needed for secured cloud deployments.
- `QDRANT_COLLECTION` is the collection where chunks are stored.

Why:

- Qdrant is not embedded like local Chroma; it is a service.
- These settings make the service endpoint configurable per environment.

### Named vector settings

```python
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense").strip()
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse").strip()
```

Meaning:

- Qdrant supports named vectors.
- This implementation stores dense embeddings under `dense`.
- It stores SPLADE sparse vectors under `sparse`.

Why:

- Hybrid retrieval needs both dense and sparse vectors for the same chunk.
- Named vectors make the dense and sparse search paths explicit.

### Vector index settings

```python
QDRANT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine").strip()
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
```

Meaning:

- Dense vector distance defaults to cosine.
- Dense vector size defaults to `1024`, matching the common dimension for `BAAI/bge-large-en-v1.5`.

Why:

- Qdrant collection creation requires a fixed dense vector size.
- The vector size must match the embedding model output.

Important:

If you change `EMBED_MODEL`, verify its embedding dimension and update `QDRANT_VECTOR_SIZE`.

### Sparse model and fusion settings

```python
QDRANT_SPARSE_MODEL = os.getenv("QDRANT_SPARSE_MODEL", "").strip()
QDRANT_HYBRID_RRF_K = int(os.getenv("QDRANT_HYBRID_RRF_K", "60"))
QDRANT_DENSE_PREFETCH_LIMIT = int(os.getenv("QDRANT_DENSE_PREFETCH_LIMIT", "20"))
QDRANT_SPARSE_PREFETCH_LIMIT = int(os.getenv("QDRANT_SPARSE_PREFETCH_LIMIT", "20"))
```

Meaning:

- `QDRANT_SPARSE_MODEL` names the SPLADE/fastembed sparse model.
- `QDRANT_HYBRID_RRF_K` controls Reciprocal Rank Fusion smoothing.
- Dense and sparse prefetch limits control how many candidates each branch retrieves before fusion.

Why:

- Hybrid retrieval should retrieve more candidates than the final `top_k`.
- Fusion can then choose the strongest final contexts from both branches.

### Default user id

```python
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "").strip()
```

Meaning:

- Optional development fallback user id.

Why:

- Production should use real authentication.
- Local development sometimes needs a temporary user id before auth is implemented.

Important:

Do not use one shared `DEFAULT_USER_ID` in production multi-user deployments.

### Qdrant switch helper

```python
def use_qdrant() -> bool:
    return VECTOR_DB == "qdrant"
```

Meaning:

- Centralizes the decision of whether the app uses Qdrant.

Why:

- Keeps branching logic clean in ingestion and retrieval.

## Data Structures

File: `backend/retrieval/schemas.py`

### `DocumentChunk`

```python
@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, Any]
```

Line-by-line:

- `@dataclass(frozen=True)` creates an immutable data container.
- `id` is the stable chunk id.
- `content` is the chunk text.
- `metadata` is the payload/provenance attached to the chunk.

Why:

- Qdrant upserts need a point id, vector, and payload.
- `DocumentChunk` cleanly represents the non-vector part of a chunk before storage.

### `RetrievedChunk`

```python
@dataclass(frozen=True)
class RetrievedChunk:
    id: str | None
    content: str
    score: float | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Line-by-line:

- `id` can be `None` because some legacy retrieval paths may not expose ids cleanly.
- `content` is the retrieved text.
- `score` is the primary score returned to downstream code.
- `dense_score` stores the dense retrieval score when available.
- `sparse_score` stores the sparse retrieval score when available.
- `fusion_score` stores the RRF score after dense/sparse fusion.
- `metadata` stores payload values like `user_id`, `document_id`, and `page_number`.

Why:

- The old retriever returned only a context string.
- Hybrid retrieval needs structured scores and metadata before context assembly.

## RRF Fusion

File: `backend/retrieval/fusion.py`

Function:

```python
def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    *,
    k: int = 60,
    limit: int,
) -> list[RetrievedChunk]:
```

Meaning:

- Accepts dense search results.
- Accepts sparse search results.
- Combines rankings using Reciprocal Rank Fusion.
- Returns the top `limit` fused results.

Why RRF:

- Dense and sparse scores live on different scales.
- Dense scores might be cosine similarities.
- Sparse scores might be SPLADE lexical scores.
- RRF avoids direct score normalization by using ranks.

### Internal maps

```python
by_id: dict[str, RetrievedChunk] = {}
scores: dict[str, float] = {}
```

Meaning:

- `by_id` stores the best known chunk object per chunk id.
- `scores` stores accumulated RRF scores per chunk id.

Why:

- Dense and sparse may return the same chunk.
- Fusion must merge duplicates instead of returning duplicate context.

### Dense loop

```python
for rank, chunk in enumerate(dense_results, start=1):
    if not chunk.id:
        continue
    by_id[chunk.id] = chunk
    scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
```

Line-by-line:

- `enumerate(..., start=1)` ranks results starting at 1.
- Missing ids are skipped because fusion needs stable ids.
- The chunk is stored by id.
- The RRF contribution is `1 / (k + rank)`.

Why:

- Rank 1 contributes more than rank 10.
- `k` prevents the top rank from dominating too aggressively.

### Sparse loop

```python
for rank, chunk in enumerate(sparse_results, start=1):
```

Meaning:

- Processes sparse results the same way as dense results.

If the sparse chunk was not seen in dense results:

```python
if existing is None:
    by_id[chunk.id] = chunk
```

If the chunk was already seen:

```python
by_id[chunk.id] = RetrievedChunk(...)
```

Meaning:

- Merges the sparse score into the existing dense chunk.
- Preserves dense score and metadata.
- Adds sparse score.

Why:

- A chunk found by both dense and sparse retrieval should become stronger.

### Ranking fused chunks

```python
ranked_ids = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
```

Meaning:

- Sorts ids from highest fused score to lowest.

Then:

```python
for chunk_id in ranked_ids[:limit]:
```

Meaning:

- Keeps only the requested number of final results.

Why:

- The LLM context window is limited.
- We do not want to pass every candidate to generation.

## SPLADE Sparse Encoder

File: `backend/retrieval/sparse.py`

### `SparseVector`

```python
@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]
```

Meaning:

- Sparse vectors are represented as index/value pairs.
- `indices` are token or vocabulary ids.
- `values` are sparse weights.

Why:

- Qdrant sparse vectors use this sparse format.
- Storing only non-zero weights is efficient.

### `SparseEncoderUnavailable`

```python
class SparseEncoderUnavailable(RuntimeError):
    pass
```

Meaning:

- Custom error for missing SPLADE configuration or dependencies.

Why:

- A clear production error is better than a vague import/model failure.

### `SpladeSparseEncoder.__init__`

```python
def __init__(self, model_name: str | None = None):
    self.model_name = (model_name or config.QDRANT_SPARSE_MODEL).strip()
    self._model = None
```

Line-by-line:

- Allows explicit model override.
- Falls back to `QDRANT_SPARSE_MODEL`.
- Strips whitespace.
- Delays model construction until the first call to `encode`.

Why:

- Avoids loading a model at import time.
- Keeps startup lighter.
- Lets tests validate config behavior without downloading model assets.

### Missing model check

```python
if not self.model_name:
    raise SparseEncoderUnavailable(...)
```

Meaning:

- Qdrant hybrid mode cannot generate sparse vectors without a configured sparse model.

Why:

- Silent fallback to dense-only would make evaluation misleading.
- Production should know when hybrid retrieval is not actually enabled.

### Optional dependency import

```python
try:
    from fastembed import SparseTextEmbedding
except ImportError as exc:
    raise SparseEncoderUnavailable(...)
```

Meaning:

- Imports `fastembed` only when sparse encoding is used.

Why:

- Keeps default Chroma mode from requiring SPLADE dependencies at runtime.
- Gives a clear install/config error in Qdrant mode.

### Lazy model construction

```python
if self._model is None:
    self._model = SparseTextEmbedding(model_name=self.model_name)
```

Meaning:

- Builds the sparse embedding model on first use.

Why:

- Avoids repeated model construction for multiple encode calls on the same encoder instance.

### Sparse output normalization

```python
indices = getattr(sparse_embedding, "indices", None)
values = getattr(sparse_embedding, "values", None)
```

Meaning:

- Reads fastembed sparse output fields.

Then:

```python
SparseVector(
    indices=[int(index) for index in indices],
    values=[float(value) for value in values],
)
```

Meaning:

- Converts sparse output into plain Python `int` and `float` lists.

Why:

- Qdrant client expects serializable values.
- This avoids numpy scalar serialization surprises.

## Qdrant Store

File: `backend/storage/qdrant_store.py`

This file owns all direct Qdrant client operations.

Why isolate it:

- Keeps Qdrant-specific API calls out of `ingest.py` and `retriever.py`.
- Makes future changes to Qdrant APIs easier.
- Makes testing simpler.

### Error classes

```python
class QdrantDependencyError(RuntimeError):
    pass
```

Used when `qdrant-client` is not installed.

```python
class QdrantMetadataError(ValueError):
    pass
```

Used when required production metadata is missing.

Why:

- These errors separate dependency failures from data-quality failures.

### Required payload fields

```python
REQUIRED_PAYLOAD_FIELDS = {
    "user_id",
    "document_id",
    "chunk_id",
    "source",
    "file_type",
    "page_number",
    "content",
    "dense_model",
    "sparse_model",
}
```

Meaning:

- Every Qdrant point must contain these payload fields.

Why each field exists:

- `user_id`: mandatory user isolation.
- `document_id`: document filtering, deletion, and audit.
- `chunk_id`: stable chunk-level identity.
- `source`: filename/source attribution.
- `file_type`: parser/type-specific behavior.
- `page_number`: citations and evaluation.
- `content`: retrieved text to pass to the LLM.
- `dense_model`: embedding lineage.
- `sparse_model`: sparse-vector lineage.

Why this matters:

- This directly enforces the migration rule: weak Chroma metadata should not be copied blindly.

### Constructor

```python
def __init__(self) -> None:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise QdrantDependencyError(...)

    self._client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
```

Line-by-line:

- Imports `QdrantClient` inside the constructor.
- Raises a project-specific error if missing.
- Creates a Qdrant client using configured URL/API key.

Why:

- Default Chroma mode should still import the app even if Qdrant is not installed.
- Qdrant dependency is needed only when Qdrant is used.

### `ensure_collection`

```python
existing = {collection.name for collection in self._client.get_collections().collections}
if config.QDRANT_COLLECTION in existing:
    return
```

Meaning:

- Checks whether the configured collection already exists.
- Does nothing if it exists.

Why:

- Collection creation should be idempotent.

Collection creation:

```python
self._client.create_collection(
    collection_name=config.QDRANT_COLLECTION,
    vectors_config={
        config.QDRANT_DENSE_VECTOR_NAME: models.VectorParams(
            size=config.QDRANT_VECTOR_SIZE,
            distance=_qdrant_distance(models, config.QDRANT_DISTANCE),
        )
    },
    sparse_vectors_config={
        config.QDRANT_SPARSE_VECTOR_NAME: models.SparseVectorParams()
    },
)
```

Line-by-line:

- Creates the configured collection.
- Adds a named dense vector.
- Uses configured vector size.
- Uses configured distance metric.
- Adds a named sparse vector.

Why:

- Dense and sparse retrieval both need first-class vector slots in Qdrant.

### `upsert_chunks`

```python
if not (len(chunks) == len(dense_vectors) == len(sparse_vectors)):
    raise ValueError(...)
```

Meaning:

- Ensures every chunk has exactly one dense vector and one sparse vector.

Why:

- Partial indexing would corrupt retrieval.

Then:

```python
self.ensure_collection()
```

Meaning:

- Creates the collection if missing.

Then for each chunk:

```python
payload = _build_payload(chunk)
```

Meaning:

- Builds and validates payload metadata.

Then:

```python
models.PointStruct(
    id=chunk.id,
    vector={
        config.QDRANT_DENSE_VECTOR_NAME: dense_vector,
        config.QDRANT_SPARSE_VECTOR_NAME: models.SparseVector(...),
    },
    payload=payload,
)
```

Meaning:

- Creates one Qdrant point per chunk.
- Stores dense vector under the dense vector name.
- Stores sparse vector under the sparse vector name.
- Stores metadata and text in payload.

Why:

- This is the core dense + SPLADE sparse Qdrant representation.

Finally:

```python
self._client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
```

Meaning:

- Inserts or updates points in Qdrant.

Why:

- `upsert` supports idempotent re-indexing by chunk id.

### `dense_search`

```python
response = self._client.query_points(
    collection_name=config.QDRANT_COLLECTION,
    query=query_vector,
    using=config.QDRANT_DENSE_VECTOR_NAME,
    query_filter=_user_filter(user_id),
    limit=top_k,
    with_payload=True,
)
```

Line-by-line:

- Searches the configured collection.
- Uses the dense embedding vector as the query.
- Searches using the named dense vector.
- Applies a mandatory `user_id` payload filter.
- Limits results to `top_k`.
- Returns payload so context text and metadata are available.

Why:

- This replaces Chroma dense vector search for Qdrant mode.
- The `user_id` filter is the production isolation boundary.

### `sparse_search`

```python
query=models.SparseVector(
    indices=query_sparse_vector.indices,
    values=query_sparse_vector.values,
),
using=config.QDRANT_SPARSE_VECTOR_NAME,
```

Meaning:

- Sends a sparse SPLADE query vector to Qdrant.
- Searches using the named sparse vector.

Why:

- This is the lexical/sparse branch of hybrid retrieval.

The same user filter is applied:

```python
query_filter=_user_filter(user_id)
```

Why:

- Dense and sparse retrieval must both enforce user isolation.

### `hybrid_search`

```python
dense_results = self.dense_search(...)
sparse_results = self.sparse_search(...)
return reciprocal_rank_fusion(...)
```

Meaning:

- Runs dense retrieval.
- Runs sparse retrieval.
- Fuses both result lists with RRF.

Why:

- This implements hybrid retrieval while keeping dense and sparse branches independently testable.

### `_build_payload`

```python
payload = {
    **chunk.metadata,
    "chunk_id": chunk.id,
    "content": chunk.content,
    "dense_model": config.EMBED_MODEL,
    "sparse_model": config.QDRANT_SPARSE_MODEL,
}
```

Meaning:

- Starts with metadata from ingestion.
- Adds chunk id.
- Adds content text.
- Adds model lineage.

Why:

- Qdrant search returns payload, so payload must contain everything needed for context, citations, filtering, and audit.

Validation:

```python
missing = sorted(field for field in REQUIRED_PAYLOAD_FIELDS if not payload.get(field))
if missing:
    raise QdrantMetadataError(...)
```

Meaning:

- Finds required fields that are missing or empty.
- Raises a hard error if any are missing.

Why:

- Prevents old weak metadata from entering Qdrant.

### `_user_filter`

```python
if not user_id:
    raise QdrantMetadataError("Qdrant retrieval requires a user_id payload filter.")
```

Meaning:

- Retrieval cannot proceed without a user id.

Why:

- The system must fail closed for user isolation.

Filter construction:

```python
return models.Filter(
    must=[
        models.FieldCondition(
            key="user_id",
            match=models.MatchValue(value=user_id),
        )
    ]
)
```

Meaning:

- Builds a Qdrant payload filter.
- Only points with matching `payload.user_id` can be retrieved.

Why:

- This is the Qdrant equivalent of:

```sql
WHERE metadata.user_id = current_user_id
```

### `_scored_point_to_chunk`

```python
payload = dict(point.payload or {})
content = str(payload.get("content", ""))
metadata = {key: value for key, value in payload.items() if key != "content"}
score = float(point.score) if point.score is not None else None
```

Meaning:

- Extracts payload from the Qdrant scored point.
- Pulls chunk text out of payload.
- Keeps the rest as metadata.
- Converts score to a plain float.

Why:

- The rest of the app should work with `RetrievedChunk`, not raw Qdrant objects.

Then:

```python
RetrievedChunk(
    id=str(point.id) if point.id is not None else None,
    content=content,
    score=score,
    dense_score=score if score_type == "dense" else None,
    sparse_score=score if score_type == "sparse" else None,
    metadata=metadata,
)
```

Meaning:

- Converts Qdrant result into the project retrieval schema.
- Stores score under dense or sparse score depending on the branch.

Why:

- RRF fusion needs structured candidate objects.

### `_qdrant_distance`

```python
distances = {
    "COSINE": models.Distance.COSINE,
    "DOT": models.Distance.DOT,
    "EUCLID": models.Distance.EUCLID,
    "MANHATTAN": models.Distance.MANHATTAN,
}
```

Meaning:

- Maps environment string values to Qdrant enum values.

Why:

- Environment variables are strings; Qdrant expects enum values.

Invalid values:

```python
if normalized not in distances:
    raise ValueError(...)
```

Meaning:

- Fails early if the distance name is not supported.

## Ingestion Changes

File: `backend/ingest.py`

### New imports

```python
from retrieval.schemas import DocumentChunk
from retrieval.sparse import SpladeSparseEncoder
from storage.qdrant_store import QdrantVectorStore
```

Meaning:

- `DocumentChunk` structures chunks before Qdrant upsert.
- `SpladeSparseEncoder` generates sparse vectors.
- `QdrantVectorStore` writes points to Qdrant.

Why:

- Ingestion now has a Qdrant branch.

### `ingest_pdf` signature

```python
def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    *,
    user_id: str | None = None,
    document_id: str | None = None,
) -> int:
```

Meaning:

- Still accepts PDF bytes and filename.
- Adds optional keyword-only `user_id` and `document_id`.

Why keyword-only:

- Prevents accidental positional argument confusion.
- Keeps old calls working: `ingest_pdf(file_bytes, filename)`.

Why user/document ids:

- Qdrant metadata needs `user_id` and `document_id`.
- These are mandatory for production filtering, deletion, and audit.

### Page-aware extraction

```python
page_texts = []
with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
    set_attribute(span, "pdf.page_count", len(pdf.pages))
    for page_number, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            page_texts.append((page_number, page_text))
```

Line-by-line:

- Starts an empty list for page-level text.
- Opens the PDF from bytes.
- Records page count for observability.
- Enumerates pages starting at 1.
- Extracts text from each page.
- Stores `(page_number, page_text)` for non-empty pages.

Why:

- The old code concatenated all PDF text into one string before chunking.
- That lost page-level provenance.
- Qdrant payloads need `page_number`.

### Chunking

```python
chunks = _chunk_pdf_pages(page_texts)
```

Meaning:

- Chunk each page separately.

Why:

- This preserves the page number for every chunk.

### IDs and documents

```python
ids = [str(uuid.uuid4()) for _ in chunks]
documents = [chunk["text"] for chunk in chunks]
metadatas = [{"source": filename} for _ in chunks]
```

Meaning:

- Generates stable ids for this ingestion run.
- Extracts text for embedding/storage.
- Keeps old Chroma metadata as `source` only.

Why:

- Chroma compatibility is maintained.
- Qdrant gets stronger metadata in a separate branch.

### Qdrant branch

```python
if config.use_qdrant():
    return _ingest_qdrant_chunks(...)
```

Meaning:

- If `VECTOR_DB=qdrant`, skip Chroma add and write to Qdrant.

Why:

- Keeps one public ingestion function but two backend storage paths.

### Chroma branch

The existing Chroma branch remains mostly unchanged:

```python
get_collection().add(
    documents=documents,
    metadatas=metadatas,
    ids=ids,
)
```

Why:

- Backward compatibility.
- The default app still behaves as before when `VECTOR_DB=chroma`.

### `_chunk_pdf_pages`

```python
def _chunk_pdf_pages(page_texts: list[tuple[int, str]]) -> list[dict]:
```

Meaning:

- Accepts page-numbered text.
- Returns dictionaries containing text and page metadata.

Inside:

```python
chunker = TokenChunker(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
)
```

Meaning:

- Uses the existing chunking strategy.

Why:

- This increment changes storage/retrieval, not chunking strategy.

Loop:

```python
for page_number, page_text in page_texts:
    for chunk_index_on_page, chunk in enumerate(chunker.chunk(page_text), start=1):
```

Meaning:

- Chunks each page independently.
- Tracks chunk index within the page.

Why:

- Enables page-level citations and debugging.

Empty chunk guard:

```python
text = chunk.text.strip()
if not text:
    continue
```

Why:

- Avoids storing blank chunks.

Returned chunk:

```python
{
    "text": text,
    "page_number": page_number,
    "chunk_index_on_page": chunk_index_on_page,
}
```

Meaning:

- Keeps chunk content and page provenance together.

### `_ingest_qdrant_chunks`

This function handles the production Qdrant ingestion path.

User id:

```python
effective_user_id = (user_id or config.DEFAULT_USER_ID).strip()
if not effective_user_id:
    raise ValueError(...)
```

Meaning:

- Uses explicit `user_id` first.
- Falls back to `DEFAULT_USER_ID`.
- Fails if neither exists.

Why:

- Qdrant must not ingest chunks that cannot be isolated by user.

Document id:

```python
effective_document_id = (document_id or str(uuid.uuid4())).strip()
```

Meaning:

- Uses provided document id or creates a new one.

Why:

- Every point needs a document id for deletion, filtering, and audit.

Vector generation:

```python
dense_vectors = embed_texts(documents)
sparse_vectors = SpladeSparseEncoder().encode(documents)
```

Meaning:

- Generates dense embeddings with the existing embedding logic.
- Generates sparse SPLADE vectors.

Why:

- Qdrant hybrid retrieval needs both dense and sparse vector representations.

Chunk object construction:

```python
DocumentChunk(
    id=chunk_id,
    content=document,
    metadata={...},
)
```

Meaning:

- Wraps each chunk into a structured object.

Metadata includes:

- `user_id`
- `document_id`
- `source`
- `file_type`
- `page_number`
- `chunk_index`
- `chunk_index_on_page`
- `chunk_type`
- `parser`
- `chunker`

Why:

- These fields support filtering, citations, re-indexing, and evaluation.

Store call:

```python
QdrantVectorStore().upsert_chunks(
    document_chunks,
    dense_vectors,
    sparse_vectors,
)
```

Meaning:

- Sends chunks and both vector types to Qdrant.

Why:

- Keeps Qdrant API details inside `qdrant_store.py`.

## Retrieval Changes

File: `backend/retriever.py`

### Public function signature

```python
def retrieve(query: str, top_k: int = config.TOP_K, user_id: str | None = None) -> str:
```

Meaning:

- Keeps the public retrieval function.
- Adds optional `user_id`.

Why:

- Chroma mode can still work as before.
- Qdrant mode requires `user_id`.

### Query embedding

```python
query_embeddings = embed_texts([query])
query_embedding = query_embeddings[0] if query_embeddings else []
```

Meaning:

- Embeds the question with the existing embedding function.

Why:

- Dense retrieval still needs query embeddings.

### Qdrant switch

```python
if config.use_qdrant():
    return _retrieve_from_qdrant(query, query_embedding, top_k=top_k, user_id=user_id)
```

Meaning:

- Routes retrieval to Qdrant only when configured.

Why:

- Keeps default Chroma behavior stable.

### Chroma path

The existing Chroma path remains unchanged:

```python
collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k,
    include=["documents", "metadatas", "distances"],
)
```

Why:

- Backward compatibility.

Important:

- Chroma path still lacks production user filtering.
- That is why Qdrant is the target production path.

### `_retrieve_from_qdrant`

User id check:

```python
effective_user_id = (user_id or config.DEFAULT_USER_ID).strip()
if not effective_user_id:
    raise ValueError(...)
```

Meaning:

- Qdrant retrieval cannot run without a user id.

Why:

- Prevents cross-user data leakage.

Sparse query generation:

```python
query_sparse_vector = SpladeSparseEncoder().encode([query])[0]
```

Meaning:

- Converts the raw question into a SPLADE sparse query vector.

Why:

- Sparse retrieval catches exact or keyword-sensitive matches.

Hybrid search:

```python
chunks = QdrantVectorStore().hybrid_search(
    query_embedding,
    query_sparse_vector,
    user_id=effective_user_id,
    top_k=top_k,
)
```

Meaning:

- Runs dense + sparse Qdrant retrieval.
- Applies user filtering inside Qdrant store.
- Returns fused results.

Why:

- This is the new target retrieval behavior.

Context build:

```python
context, retrieval_documents, current_tokens = _build_context(chunks)
```

Meaning:

- Converts structured retrieval results into the context string expected by the existing LLM prompt builder.

Why:

- Avoids changing the LLM flow in this increment.

### `_build_context`

This function turns `RetrievedChunk` objects into:

- a context string
- structured retrieval documents for tracing
- estimated token count

Token estimate:

```python
estimated_tokens = len(chunk.content) // 4
```

Meaning:

- Uses the existing rough estimate of 4 characters per token.

Why:

- Maintains existing context-budget behavior.

Budget check:

```python
included_in_context = (
    not context_budget_exhausted
    and current_tokens + estimated_tokens <= config.MAX_CTX_TOKENS
)
```

Meaning:

- Adds chunks until the context token budget is exhausted.

Why:

- Prevents sending too much context to the LLM.

Metadata added:

```python
"dense_score": chunk.dense_score,
"sparse_score": chunk.sparse_score,
"fusion_score": chunk.fusion_score,
```

Meaning:

- Keeps retrieval-score observability for Qdrant hybrid search.

Why:

- Production debugging needs to know why a chunk was selected.

Final context:

```python
return "\n\n---\n\n".join(context_chunks), retrieval_documents, current_tokens
```

Meaning:

- Uses the same chunk separator as the Chroma path.

Why:

- Avoids changing prompt behavior.

## API Changes

File: `backend/main.py`

### Upload endpoint

```python
async def upload(
    file: UploadFile = File(...),
    user_id: str | None = Form(None),
    document_id: str | None = Form(None),
):
```

Meaning:

- Existing PDF upload still works.
- Optional form fields allow Qdrant metadata.

Why:

- Qdrant ingestion needs `user_id` and `document_id`.
- Chroma mode does not require them.

### Chat request

```python
class ChatRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    user_id: str | None = None
```

Meaning:

- Existing chat request still works.
- Optional `user_id` is available for Qdrant retrieval.

Why:

- Qdrant retrieval requires user isolation.

### Retrieval call

```python
context = retrieve(
    rewritten_question,
    top_k=settings["top_k"],
    user_id=request.user_id,
)
```

Meaning:

- Passes `user_id` to the retriever.

Why:

- The retriever decides whether it is needed based on `VECTOR_DB`.

## Environment Example

File: `.env.example`

New config:

```env
VECTOR_DB=chroma
DEFAULT_USER_ID=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_chunks
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_DISTANCE=Cosine
QDRANT_VECTOR_SIZE=1024
QDRANT_SPARSE_MODEL=
QDRANT_HYBRID_RRF_K=60
QDRANT_DENSE_PREFETCH_LIMIT=20
QDRANT_SPARSE_PREFETCH_LIMIT=20
```

Why:

- Makes Qdrant deployment explicit.
- Keeps Chroma as default until Qdrant is fully provisioned.

## Requirements

File: `requirements.txt`

Added:

```text
qdrant-client
fastembed
```

Meaning:

- `qdrant-client` is the Python SDK for Qdrant.
- `fastembed` provides sparse text embedding support.

Why:

- Qdrant hybrid retrieval needs both a Qdrant client and a sparse encoder.

Important:

- Installing `fastembed` does not by itself select or download a SPLADE model.
- A model must be configured through `QDRANT_SPARSE_MODEL`.

## Tests

File: `tests/test_qdrant_planning_helpers.py`

### RRF test

```python
def test_rrf_merges_dense_and_sparse_results(self):
```

What it validates:

- Dense and sparse results are fused.
- Duplicate chunk `b` is merged.
- The chunk found by both branches ranks first.
- Dense and sparse scores are preserved.

Why:

- Hybrid retrieval quality depends on correct fusion behavior.

### Sparse encoder config test

```python
def test_sparse_encoder_requires_configured_model(self):
```

What it validates:

- Missing `QDRANT_SPARSE_MODEL` raises `SparseEncoderUnavailable`.

Why:

- The system should not pretend hybrid retrieval is enabled if no sparse model is configured.

### Weak metadata rejection test

```python
def test_payload_rejects_old_source_only_metadata(self):
```

What it validates:

- A chunk with only `source` metadata is rejected.

Why:

- This enforces the migration rule:

```text
Do not migrate old weak metadata blindly.
```

### In-memory Qdrant integration test

```python
def test_in_memory_qdrant_hybrid_search_enforces_user_filter(self):
```

What it validates:

- Creates an in-memory Qdrant client.
- Inserts one chunk for `user-1`.
- Inserts another chunk for `user-2`.
- Searches as `user-1`.
- Only `user-1` content is returned.

Why:

- This verifies the most important production safety property: user isolation.

### Required metadata acceptance test

```python
def test_payload_accepts_required_qdrant_metadata(self):
```

What it validates:

- Complete metadata is accepted.
- Payload contains `user_id`, `chunk_id`, `content`, `dense_model`, and `sparse_model`.

Why:

- Ensures valid Qdrant payloads are constructed correctly.

## Validation Performed

Commands run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Result:

```text
Ran 5 tests
OK
```

Command:

```powershell
.\.venv\Scripts\python.exe -m compileall backend tests
```

Result:

```text
OK
```

Command:

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'backend'); import main; from fastapi.testclient import TestClient; c=TestClient(main.app); print(c.get('/health').status_code); print(c.get('/settings').status_code)"
```

Result:

```text
200
200
```

Additional validation:

- Installed `qdrant-client==1.18.0`.
- Found that `QdrantClient.search(...)` is not available in that version.
- Patched the implementation to use `QdrantClient.query_points(...)`.
- Ran in-memory Qdrant hybrid retrieval successfully.

## Current Limitations

This is not yet a complete production migration.

Remaining work:

- Select and configure the actual SPLADE sparse model.
- Run a real Qdrant service locally or in cloud.
- Re-ingest source documents with strong metadata.
- Run the golden dataset benchmark against Qdrant dense and Qdrant hybrid.
- Add real authentication instead of relying on optional `user_id` in request body.
- Add document deletion/re-indexing flows.
- Add structured retrieval metadata to `/chat` responses in debug/eval mode.
- Add production Docker/compose or deployment instructions for Qdrant.

## How to Enable Qdrant Mode

Example `.env`:

```env
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_chunks
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
QDRANT_DISTANCE=Cosine
QDRANT_VECTOR_SIZE=1024
QDRANT_SPARSE_MODEL=<chosen-fastembed-sparse-model>
DEFAULT_USER_ID=local-dev-user
```

For production:

- Do not use `DEFAULT_USER_ID`.
- Use authenticated user identity.
- Pass the authenticated user id into ingestion and retrieval internally.

## Mental Model

Think of each chunk as one Qdrant point:

```text
Qdrant point
  id: chunk id
  dense vector: semantic embedding
  sparse vector: SPLADE lexical vector
  payload:
    user_id
    document_id
    source
    page_number
    content
    model metadata
```

At query time:

```text
Question
 -> dense embedding
 -> sparse SPLADE vector
 -> dense search filtered by user_id
 -> sparse search filtered by user_id
 -> RRF fusion
 -> top chunks
 -> context string
 -> LLM
```

This gives the system:

- semantic retrieval through dense vectors
- exact/lexical retrieval through SPLADE sparse vectors
- safer multi-user isolation through Qdrant payload filters
- better evaluation/debugging through structured scores and metadata

## Production Design Principle

The most important design decision is not Qdrant itself. The most important decision is failing closed on metadata.

The implementation intentionally rejects incomplete Qdrant payloads because production RAG systems usually fail in subtle ways when metadata is treated as optional.

For this system, Qdrant mode requires:

```text
user_id
document_id
chunk_id
source
file_type
page_number
content
dense_model
sparse_model
```

That requirement protects:

- user privacy
- document deletion
- page citations
- auditability
- evaluation correctness
- future migration safety

