# RAG Observability Implementation Notes

## Summary

This project now has Phoenix/OpenTelemetry observability for the RAG pipeline, a working Hugging Face chat model configuration, runtime Hugging Face key rotation, and an embedded Phoenix traces view inside the frontend settings modal.

The main goal was to see traces for each RAG query in Arize Phoenix, including embedding, vector search, prompt building, and model-call spans.

## Backend Observability

Phoenix tracing was added through OpenTelemetry and `arize-phoenix-otel`.

New file:

- `backend/observability.py`

This module:

- Registers Phoenix tracing once during FastAPI startup.
- Sends spans to `http://localhost:6006/v1/traces`.
- Uses Phoenix project name `rag-project`.
- Instruments FastAPI.
- Provides safe helper functions for spans and attributes.
- Keeps tracing best-effort, so tracing failures do not break the RAG app.

Environment configuration:

```env
PHOENIX_TRACING_ENABLED=true
PHOENIX_PROJECT_NAME=rag-project
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces
PHOENIX_CAPTURE_CONTENT=true
PHOENIX_CAPTURE_CONTENT_MAX_CHARS=4000
```

`PHOENIX_CAPTURE_CONTENT=true` makes Phoenix show the user question, system prompt, retrieved context, and answer output in span attributes. Long content is truncated using `PHOENIX_CAPTURE_CONTENT_MAX_CHARS`.

## RAG Trace Structure

Each `/chat` request now creates this trace structure:

```text
rag.chat
  embedding
  vector_search
  prompt_build
  model_call
```

The parent span `rag.chat` is created in `backend/main.py`.

Captured attributes include:

- Question length/context metadata
- History length
- Top-k value
- Temperature
- Max token setting
- Context character count
- Answer character count
- User question content
- System prompt content
- Answer output content

## Span Details

### Embedding

Implemented in `backend/ingest.py`.

Captured attributes:

```text
embedding.vendor = huggingface
embedding.model = BAAI/bge-large-en-v1.5
embedding.input_count
embedding.dimension
embedding.duration_ms
```

This span is used for both:

- PDF ingestion embeddings
- Query embeddings during retrieval

### Vector Search

Implemented in `backend/retriever.py`.

Captured attributes:

```text
vector_store.vendor = chromadb
retrieval.top_k
retrieval.max_context_tokens
retrieval.returned_chunks
retrieval.context_tokens_estimate
vector_search.duration_ms
```

Retrieved context content is captured only when `PHOENIX_CAPTURE_CONTENT=true`, and long values are truncated to avoid oversized traces.

### Prompt Build

Implemented in `backend/llm.py`.

Captured attributes:

```text
prompt.history_messages_received
prompt.history_messages_used
prompt.context_chars
prompt.question_chars
prompt.message_count
prompt_build.duration_ms
prompt.system_prompt
prompt.user_question
prompt.retrieved_context
input.value
```

### Model Call

Implemented in `backend/llm.py`.

Captured attributes:

```text
llm.vendor = huggingface
llm.model
llm.resolved_model
llm.fallback_model_count
llm.temperature
llm.max_tokens
llm.output_chars
llm.duration_ms
output.value
llm.answer
```

The model-call span records errors if Hugging Face rejects a model or token.

## Ingestion Observability

PDF uploads now create this trace structure:

```text
rag.ingest
  pdf_parse
  chunking
  embedding
  vector_add
```

Implemented in:

- `backend/main.py`
- `backend/ingest.py`

Captured attributes include:

- Uploaded filename
- Uploaded file size
- PDF page count
- Extracted text character count
- Chunk size and overlap
- Chunk count
- Chroma document count
- Vector add duration

## Hugging Face Model Fix

The original model:

```text
mistralai/Mistral-7B-Instruct-v0.2
```

was failing because Hugging Face provider routing reported it was not supported by enabled providers.

The working default model is now:

```env
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

Fallback models were also added:

```env
LLM_FALLBACK_MODELS=mistralai/Mistral-7B-Instruct-v0.3,HuggingFaceH4/zephyr-7b-beta
```

If the primary model is rejected as unsupported or unauthorized, the backend tries the fallback models before returning an error.

## Runtime Hugging Face Key Rotation

The backend no longer locks the Hugging Face API key at import/startup time.

Implemented in:

- `backend/config.py`
- `backend/ingest.py`
- `backend/llm.py`

`backend/config.py` now has:

```python
get_hf_api_key()
```

This reads the current `HF_API_KEY` from `.env` at request time.

The embedding client and LLM client both recreate their Hugging Face `InferenceClient` automatically when the token changes.

To rotate keys, update only:

```env
HF_API_KEY=your_new_huggingface_key
```

Then send the next chat or upload request. A backend restart should not be required for key changes.

## Frontend Changes

The settings modal in `frontend/index.html` was expanded into a larger two-column layout.

Left column:

- System prompt
- Temperature
- Max new tokens
- Top-k chunks

Right column:

- Embedded Phoenix traces and observability view
- Refresh button
- Open button

Embedded Phoenix URL:

```text
http://127.0.0.1:6006/projects/
```

The settings modal was enlarged to nearly full screen:

```css
width: 96vw;
height: 92vh;
```

This makes the Phoenix traces view readable inside the RAG app.

## Dependency Changes

Updated `requirements.txt` with:

```text
python-dotenv
huggingface_hub
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-fastapi
arize-phoenix-otel
```

These are required for:

- Loading `.env`
- Hugging Face inference
- OpenTelemetry tracing
- Phoenix trace export
- FastAPI instrumentation

## Files Changed

Backend:

- `backend/config.py`
- `backend/main.py`
- `backend/observability.py`
- `backend/ingest.py`
- `backend/retriever.py`
- `backend/llm.py`

Frontend:

- `frontend/index.html`

Configuration:

- `.env.example`
- `requirements.txt`

Documentation:

- `OBSERVABILITY_IMPLEMENTATION.md`

## How To Run

Start Phoenix:

```powershell
phoenix serve
```

Phoenix UI:

```text
http://127.0.0.1:6006/projects/
```

Start backend from `backend/`:

```powershell
..\.venv\Scripts\uvicorn.exe main:app --host 127.0.0.1 --port 8000
```

Backend health:

```text
http://127.0.0.1:8000/health
```

Open frontend:

```text
frontend/index.html
```

## Verification Completed

The following checks were run successfully:

```text
python -m compileall backend
import main
GET /health
POST /chat
```

A real `/chat` request returned `200 OK` after the Hugging Face token/model fix.

Phoenix was also verified as reachable at:

```text
http://127.0.0.1:6006/projects/
```

## Expected Phoenix Result

After a chat request, Phoenix should show traces under the `rag-project` project with spans like:

```text
rag.chat
embedding
vector_search
prompt_build
model_call
```

After uploading a PDF, Phoenix should show:

```text
rag.ingest
pdf_parse
chunking
embedding
vector_add
```
