# RAG Evaluation Implementation Notes

This document explains the code-level implementation added for offline RAG evaluation.

## Purpose

The goal is to measure the complete RAG pipeline with a golden dataset:

1. Load a human-reviewed or generated evaluation dataset.
2. Run each question through the same retriever and LLM used by `/chat`.
3. Score retrieval quality and answer quality with RAGAS metrics.
4. Compute derived business-facing metrics such as hallucination rate, retrieval F1, and overall RAG score.
5. Save both machine-readable JSON and human-readable Markdown reports.

## Files Added

### `backend/evaluation/eval_config.py`

Defines default paths, thresholds, score weights, and evaluator model configuration.

Important objects:

- `EVAL_THRESHOLDS`: pass/fail gates for every metric.
- `LOWER_IS_BETTER`: identifies metrics where lower values are better, currently `hallucination_rate`.
- `OVERALL_WEIGHTS`: weighted formula for `overall_rag_score`.
- `DEFAULT_EVALUATION_CONFIG`: runtime settings used by the evaluator when API or CLI callers do not override them.

The evaluator LLM defaults to `gpt-4o-mini`, and evaluator embeddings default to `text-embedding-3-small`. These can be changed through environment variables:

- `EVALUATOR_LLM`
- `EVALUATOR_EMBEDDING`
- `EVALUATOR_OPENAI_API_KEY`
- `EVALUATOR_OPENAI_BASE_URL`

### `backend/evaluation/golden_dataset_generator.py`

Generates a golden dataset JSON from a PDF.

Flow:

1. Extract PDF text with `pdfplumber`.
2. Chunk text with the same `TokenChunker` settings used by production ingestion: `config.CHUNK_SIZE` and `config.CHUNK_OVERLAP`.
3. Ask the configured LLM to generate question-answer pairs from each chunk.
4. Save samples using this schema:

```json
{
  "id": "q_001",
  "question": "...",
  "ground_truth": "...",
  "expected_contexts": ["..."],
  "metadata": {
    "category": "factual",
    "difficulty": "easy",
    "source_chunk_index": 0,
    "source_page": [],
    "topic": "general"
  }
}
```

Why this design:

- The generated `expected_contexts` preserve the source evidence for each golden answer.
- Metadata gives you a way to filter reports by topic, difficulty, or source chunk later.
- The generator writes JSON so you can manually review and edit questions before running evaluation.

### `backend/evaluation/metrics.py`

Runs the actual evaluation.

Main entry point:

```python
run_evaluation(golden_dataset_path, config, output_json_path, output_markdown_path)
```

For each sample:

1. Calls `retrieve(question, top_k=...)`.
2. Splits the retrieved context back into a list of chunks.
3. Builds the same prompt shape as `/chat` through `build_prompt(...)`.
4. Calls `ask_llm(...)`.
5. Records diagnostic fields:
   - retrieval latency
   - generation latency
   - estimated token usage
   - chunk count used

RAGAS-native metrics loaded:

- `context_precision`
- `context_recall`
- `context_entity_recall`
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `answer_similarity`

Compatibility note:

Installed RAGAS `0.4.3` no longer exposes the old `context_relevancy` metric as a simple top-level metric. The implementation therefore computes `context_relevancy` with a deterministic lexical fallback: it checks how many retrieved chunks share significant terms with the question. The report keeps the planned `context_relevancy` field name for schema stability.

Derived metrics computed locally:

- `hallucination_rate = 1 - faithfulness`
- `groundedness_score = faithfulness`
- `retrieval_f1 = harmonic_mean(context_precision, context_recall)`
- `overall_rag_score = weighted average from eval_config.OVERALL_WEIGHTS`

### `backend/evaluation/report.py`

Builds and saves reports.

Outputs:

- `backend/evaluation/report.json`
- `backend/evaluation/report.md`

The JSON report contains:

- `summary`
- `aggregate_scores`
- `per_question_scores`
- `failed_questions`
- `thresholds`
- `config_used`

The Markdown report is intended for quick human review.

### `backend/evaluation/run_evaluation.py`

Adds CLI support.

Generate a dataset:

```bash
python -m evaluation.run_evaluation --generate --pdf path/to/golden.pdf --golden-dataset evaluation/golden_dataset.json
```

Run evaluation:

```bash
python -m evaluation.run_evaluation --golden-dataset evaluation/golden_dataset.json --output evaluation/report.json
```

Run both:

```bash
python -m evaluation.run_evaluation --generate --pdf path/to/golden.pdf --golden-dataset evaluation/golden_dataset.json --output evaluation/report.json
```

Run these commands from the `backend` directory, or make sure `backend` is on `PYTHONPATH`.

## API Changes

### `POST /evaluation/generate-golden-dataset`

Accepts a PDF upload and writes:

```text
backend/evaluation/golden_dataset.json
```

Purpose:

- Create the first draft of the golden dataset from a source PDF.
- Keep chunking aligned with production ingestion.

### `POST /evaluation/run`

Runs evaluation against a golden dataset and writes report outputs.

Request body:

```json
{
  "golden_dataset_path": "backend/evaluation/golden_dataset.json",
  "output_path": "backend/evaluation/report.json",
  "markdown_output_path": "backend/evaluation/report.md",
  "top_k": 3
}
```

If `top_k` is omitted, the endpoint uses the current app setting.

### `GET /evaluation/report`

Returns the latest `report.json`.

If no report exists yet, it returns HTTP 404.

## Existing Pipeline Change

### `backend/ingest.py`

Chroma initialization was moved from module import time to first use.

Why:

- The existing `backend/chroma_db` directory currently triggers a Chroma Rust binding panic.
- Before this change, importing `main.py` failed, so FastAPI could not start.
- Lazy initialization lets the API boot and keeps vector-store failures scoped to upload/chat/evaluation operations.

A non-destructive recovery path was also added:

- Primary DB: `backend/chroma_db`
- Recovery DB: `backend/chroma_db_recovered`

If the primary Chroma DB cannot initialize, the app uses the recovery DB instead. The old DB is not deleted. Because the recovery DB starts empty, documents must be re-uploaded before chat or evaluation has useful context.

## Dependencies Added

`requirements.txt` now includes:

```text
ragas>=0.2.0
datasets
langchain
langchain-openai
langchain-community
```

RAGAS is the scoring engine. `datasets` provides the Hugging Face Dataset object expected by RAGAS. LangChain OpenAI wrappers are used when RAGAS accepts explicit evaluator LLM and embedding objects.

## Verification Completed

Commands run successfully:

```bash
python -m compileall backend
python -c "import sys; sys.path.insert(0, 'backend'); import main; print('app import ok')"
python -c "import sys; sys.path.insert(0, 'backend'); from evaluation.metrics import _load_ragas_metrics; metrics=_load_ragas_metrics(); assert len(metrics)==7; print([m.name for m in metrics])"
python -c "import sys; sys.path.insert(0, 'backend'); from fastapi.testclient import TestClient; import main; client=TestClient(main.app); assert client.get('/health').json()['status']=='ok'; print('api smoke ok')"
```

RAGAS dependencies were installed with:

```bash
python -m pip install -r requirements.txt
```

## Current Runtime Requirements

To run a real evaluation:

1. Re-upload documents because the app is now using `backend/chroma_db_recovered`.
2. Create or generate `backend/evaluation/golden_dataset.json`.
3. Set a valid evaluator key:

```text
EVALUATOR_OPENAI_API_KEY=...
```

4. Run `/evaluation/run` or the CLI.

Phoenix tracing is still configured. If Phoenix is not running at `localhost:6006`, the app may log exporter retry messages, but the evaluation code still runs.

## Phoenix Dashboard Integration

Evaluation metrics are now emitted to Phoenix/OpenTelemetry through the existing `observability.py` helpers.

Phoenix spans added:

```text
evaluation.run
├── evaluation.questions
│   └── evaluation.question
│       ├── vector_search
│       ├── prompt_building
│       └── OpenAI/Hugging Face LLM spans from auto-instrumentation
├── evaluation.metrics
│   ├── evaluation.ragas
│   │   └── evaluator LLM / embedding spans from RAGAS, when emitted
│   └── evaluation.question.scores
├── evaluation.report.build
└── evaluation.report.save
```

Span purposes:

- `evaluation.run`: parent span for the complete evaluation run.
- `evaluation.questions`: groups all golden-question RAG executions.
- `evaluation.question`: one span for each golden question as it goes through retriever and generator.
- `evaluation.metrics`: groups all scoring work.
- `evaluation.ragas`: RAGAS scoring call.
- `evaluation.question.scores`: one span per question after metric scores are merged.
- `evaluation.report.build`: builds pass/fail report object.
- `evaluation.report.save`: writes JSON/Markdown reports.

Aggregate metric attributes on `evaluation.run` and `evaluation.ragas`:

```text
evaluation.metric.context_precision
evaluation.metric.context_recall
evaluation.metric.context_relevancy
evaluation.metric.context_entity_recall
evaluation.metric.faithfulness
evaluation.metric.answer_relevancy
evaluation.metric.answer_correctness
evaluation.metric.answer_similarity
evaluation.metric.hallucination_rate
evaluation.metric.groundedness_score
evaluation.metric.retrieval_f1
evaluation.metric.overall_rag_score
evaluation.pass
evaluation.failed_question_count
evaluation.total_questions
```

Per-question attributes on `evaluation.question` and `evaluation.question.scores`:

```text
evaluation.question_id
evaluation.question_category
evaluation.question_difficulty
evaluation.source_chunk_index
evaluation.retrieval_latency_ms
evaluation.generation_latency_ms
evaluation.chunk_count_used
evaluation.input_tokens_estimate
evaluation.output_tokens_estimate
evaluation.context_relevancy
evaluation.metric.faithfulness
evaluation.metric.answer_correctness
```

How to view in Phoenix:

1. Start Phoenix on the configured endpoint, currently `http://localhost:6006`.
2. Run evaluation through `POST /evaluation/run` or the CLI after calling `setup_observability`.
3. In Phoenix, filter spans by names beginning with `evaluation.`.
4. Inspect `evaluation.metric.*` attributes for the scores.

Security note:

Evaluation config sent to Phoenix is redacted for keys containing `api_key`, `token`, or `secret`.
