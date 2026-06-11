# RAGAS-Based LLM Evaluation for RAG Chatbot

## Background

The existing RAG chatbot at [rag-project](file:///d:/AI%20Builders%20Community/RAG/rag-project) has a working pipeline: **PDF Ingestion → ChromaDB Vector Store → Retriever → Prompt Builder → Generator LLM → Answer**. Phoenix/OpenTelemetry observability is already wired in.

This plan adds an **offline evaluation layer** using **RAGAS** that measures how well both the **retriever** and the **generator** perform, aligned exactly with the architecture diagrams you provided.

---

## All Evaluation Metrics — Quick Reference

The table below lists **every metric** we will calculate, what it measures, what data it needs, and what a good score looks like.

### Retrieval Metrics (Is the retriever finding the right chunks?)

| # | Metric | What It Measures | Inputs Needed | Score Range | Good Threshold |
|---|--------|-----------------|---------------|-------------|----------------|
| 1 | **Context Precision** | Are the **relevant** chunks ranked at the **top** of the retrieved list? (signal-to-noise) | `question`, `contexts`, `ground_truth` | 0 → 1 | ≥ 0.85 |
| 2 | **Context Recall** | Did the retriever find **all** the chunks needed to answer? (coverage) | `question`, `contexts`, `ground_truth` | 0 → 1 | ≥ 0.80 |
| 3 | **Context Relevancy** | What fraction of the retrieved context is actually **relevant** to the question? | `question`, `contexts` | 0 → 1 | ≥ 0.75 |
| 4 | **Context Entity Recall** | Did the retrieved chunks capture the **key entities** from the ground truth? | `contexts`, `ground_truth` | 0 → 1 | ≥ 0.70 |

### Generation Metrics (Is the LLM answering correctly and faithfully?)

| # | Metric | What It Measures | Inputs Needed | Score Range | Good Threshold |
|---|--------|-----------------|---------------|-------------|----------------|
| 5 | **Faithfulness** | Are the claims in the answer **supported** by the retrieved context? (no invention) | `question`, `answer`, `contexts` | 0 → 1 | ≥ 0.85 |
| 6 | **Answer Relevancy** | Is the answer actually **relevant** to the question asked? | `question`, `answer`, `contexts` | 0 → 1 | ≥ 0.80 |
| 7 | **Answer Correctness** | Does the answer **match** the golden reference answer? (factual + semantic) | `answer`, `ground_truth` | 0 → 1 | ≥ 0.70 |
| 8 | **Answer Similarity** | Semantic similarity between the generated answer and the reference answer | `answer`, `ground_truth` | 0 → 1 | ≥ 0.75 |

### Derived / Composite Metrics (Calculated from the above)

| # | Metric | Formula | Good Threshold |
|---|--------|---------|----------------|
| 9 | **Hallucination Rate** | `1 − Faithfulness` (% of claims not grounded in context) | ≤ 0.15 |
| 10 | **Groundedness Score** | Same as Faithfulness (% of claims grounded in context) | ≥ 0.85 |
| 11 | **Retrieval F1** | Harmonic mean of Context Precision & Context Recall | ≥ 0.80 |
| 12 | **Overall RAG Score** | Weighted average: `0.3×ContextRecall + 0.2×ContextPrecision + 0.25×Faithfulness + 0.15×AnswerRelevancy + 0.1×AnswerCorrectness` | ≥ 0.75 |

### Per-Question Diagnostic Metrics (Logged but not aggregated)

| # | Metric | What It Captures |
|---|--------|-----------------|
| 13 | **Latency (retrieval)** | Time for vector search + reranking |
| 14 | **Latency (generation)** | Time for LLM response |
| 15 | **Token Usage** | Input/output token counts per question |
| 16 | **Chunk Count Used** | How many chunks ended up in the prompt |

---

## Golden Dataset — Structure & Preparation Strategy

When you upload your golden document, the system will prepare a golden dataset JSON file with this schema:

```jsonc
{
  "metadata": {
    "source_document": "your_document.pdf",
    "created_at": "2026-06-09T20:00:00Z",
    "version": "1.0",
    "total_questions": 30
  },
  "samples": [
    {
      "id": "q_001",
      "question": "What are the chunking strategies available?",
      "ground_truth": "The golden reference answer written by a human or extracted from the doc...",
      "expected_contexts": [
        "Chunk text 1 that SHOULD be retrieved to answer this question...",
        "Chunk text 2 that SHOULD be retrieved..."
      ],
      "metadata": {
        "category": "factual",           // factual | reasoning | multi-hop | summary
        "difficulty": "easy",            // easy | medium | hard
        "source_page": [3, 4],           // page numbers in the PDF
        "topic": "chunking"              // topic tag for filtering
      }
    }
    // ... more samples
  ]
}
```

### How the Golden Dataset Will Be Prepared

```
┌─────────────────────────────────────────────────┐
│              Your Golden Document (PDF)           │
└────────────────────┬────────────────────────────┘
                     │  1. Parse + Extract Text
                     ▼
┌─────────────────────────────────────────────────┐
│          Chunk the document (same settings       │
│          as production: 400 tokens, 50 overlap)  │
└────────────────────┬────────────────────────────┘
                     │  2. Generate Q&A pairs
                     ▼
┌─────────────────────────────────────────────────┐
│  For each chunk (or group of chunks):            │
│   • Generate 1-3 questions of varying difficulty │
│   • Write golden reference answers               │
│   • Tag expected_contexts (the chunk IDs/text)   │
│   • Add metadata (page, topic, difficulty)       │
└────────────────────┬────────────────────────────┘
                     │  3. Human review (optional)
                     ▼
┌─────────────────────────────────────────────────┐
│         golden_dataset.json  (saved to           │
│         backend/evaluation/golden_dataset.json)  │
└─────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> The golden dataset generator will use an LLM to auto-generate Q&A pairs from your document, then you can review/edit the JSON. This ensures high coverage while letting you curate quality.

---

## Proposed Changes

### Component 1: Evaluation Module (NEW)

All new files live under `backend/evaluation/`.

---

#### [NEW] [__init__.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/__init__.py)

Empty init file to make `evaluation` a Python package.

---

#### [NEW] [golden_dataset_generator.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/golden_dataset_generator.py)

Responsible for:
- Accepting a PDF (the golden document)
- Parsing & chunking it using the **same** `ingest.py` logic (ensuring consistency)
- Using the configured LLM to generate question–answer–expected_context triples
- Outputting `golden_dataset.json`

Key functions:
```python
def generate_golden_dataset(pdf_path: str, output_path: str) -> dict
def _generate_qa_from_chunks(chunks: list[str]) -> list[dict]
def _classify_question(question: str) -> dict  # category, difficulty
```

---

#### [NEW] [metrics.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/metrics.py)

Core evaluation logic using RAGAS. Responsible for:
- Loading the golden dataset
- Running each question through the RAG pipeline (retriever → LLM)
- Collecting the `question`, `answer`, `contexts`, `ground_truth` tuples
- Computing **all 12 metrics** via RAGAS
- Computing derived metrics (hallucination rate, groundedness, retrieval F1, overall score)

Key functions:
```python
def run_evaluation(golden_dataset_path: str, config: dict) -> EvaluationReport
def _run_single_question(question: str, top_k: int) -> dict
def _compute_derived_metrics(ragas_scores: dict) -> dict
```

RAGAS metrics used:
```python
from ragas.metrics import (
    context_precision,
    context_recall,
    context_relevancy,
    context_entity_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
    answer_similarity,
)
```

---

#### [NEW] [report.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/report.py)

Generates the final evaluation report:
- JSON report with all per-question and aggregate scores
- Markdown summary table for human reading
- Pass/fail gates against configurable thresholds

Output structure:
```jsonc
{
  "summary": {
    "overall_rag_score": 0.82,
    "pass": true,
    "timestamp": "2026-06-09T20:00:00Z",
    "total_questions": 30
  },
  "aggregate_scores": {
    "context_precision": 0.87,
    "context_recall": 0.83,
    "context_relevancy": 0.79,
    "faithfulness": 0.91,
    "answer_relevancy": 0.85,
    "answer_correctness": 0.74,
    "hallucination_rate": 0.09,
    "groundedness_score": 0.91,
    "retrieval_f1": 0.85
  },
  "per_question_scores": [ /* ... */ ],
  "failed_questions": [ /* questions below threshold */ ],
  "config_used": { /* RAG settings used during evaluation */ }
}
```

---

#### [NEW] [eval_config.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/eval_config.py)

Evaluation configuration:
```python
EVAL_THRESHOLDS = {
    "context_precision": 0.85,
    "context_recall": 0.80,
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "answer_correctness": 0.70,
    "hallucination_rate": 0.15,   # upper bound (lower is better)
    "overall_rag_score": 0.75,
}

OVERALL_WEIGHTS = {
    "context_recall": 0.30,
    "context_precision": 0.20,
    "faithfulness": 0.25,
    "answer_relevancy": 0.15,
    "answer_correctness": 0.10,
}

EVALUATOR_LLM = "gpt-4o-mini"  # LLM used by RAGAS to judge
EVALUATOR_EMBEDDING = "text-embedding-3-small"  # Embedding used by RAGAS
```

---

### Component 2: API Endpoints

---

#### [MODIFY] [main.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/main.py)

Add three new endpoints:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/evaluation/generate-golden-dataset` | Upload golden PDF → generate `golden_dataset.json` |
| `POST` | `/evaluation/run` | Run full evaluation against golden dataset |
| `GET`  | `/evaluation/report` | Fetch the latest evaluation report |

---

### Component 3: Dependencies

---

#### [MODIFY] [requirements.txt](file:///d:/AI%20Builders%20Community/RAG/rag-project/requirements.txt)

Add:
```text
ragas>=0.2.0
datasets
langchain
langchain-openai
langchain-community
```

---

### Component 4: CLI Runner (Optional Convenience)

---

#### [NEW] [run_evaluation.py](file:///d:/AI%20Builders%20Community/RAG/rag-project/backend/evaluation/run_evaluation.py)

A standalone CLI script so you can run evaluation without starting the FastAPI server:

```bash
python -m evaluation.run_evaluation --golden-dataset evaluation/golden_dataset.json --output evaluation/report.json
```

---

## Architecture Alignment

Your architecture diagram shows this exact flow — here's how it maps:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Golden Evaluation Dataset                              │
│                  (golden_dataset.json)                                    │
├────────────┬──────────────────────────┬──────────────────────────────────┤
│  Question  │  Reference Answer /      │  Reference Contexts /            │
│            │  Golden Answer           │  Expected Chunks                 │
└─────┬──────┴──────────────────────────┴──────────────────────────────────┘
      │
      ▼
┌─────────────────────┐
│    RAG Pipeline      │   ← Your existing pipeline
│  retriever.py        │   (Vector Search → Retrieved Chunks →
│  llm.py              │    Reranker → Top-K → Prompt → LLM)
└─────────┬───────────┘
          │
          ▼  Generated Answer
┌─────────────────────────────────────────────────────────────────────────┐
│                     RAGAS Evaluation Layer                               │
│                     (metrics.py)                                        │
│                                                                          │
│  Uses: Evaluator LLM (gpt-4o-mini) + Embedding Model (text-embedding)  │
└─────────┬───────────────────────────────────────────────────────────────┘
          │
          ▼  Metric Scores
┌─────────┴─────────┬────────────────┬──────────────────┬────────────────┐
│ Context Precision │ Context Recall │  Faithfulness    │ Answer         │
│ Context Relevancy │ Entity Recall  │  Groundedness    │ Relevancy      │
│                   │                │  Hallucination   │ Correctness    │
└───────────────────┴────────────────┴──────────────────┴────────────────┘
          │
          ▼
┌──────────────────────────────────┐
│   Final Evaluation Report        │
│   (report.json + report.md)      │
│   Dashboard / CI Gate / Tracking │
└──────────────────────────────────┘
```

---

## How Each Metric Is Calculated (Short Summary)

| Metric | How RAGAS Computes It |
|--------|----------------------|
| **Context Precision** | Evaluator LLM checks if each retrieved chunk is useful for the question given the ground truth. Computes weighted precision (rank-aware). |
| **Context Recall** | Evaluator LLM decomposes ground_truth into claims → checks if each claim is attributable to retrieved contexts. |
| **Context Relevancy** | Evaluator LLM extracts sentences from contexts that are relevant to the question → `relevant_sentences / total_sentences`. |
| **Context Entity Recall** | Extracts named entities from ground_truth and contexts → `entities_in_context / entities_in_ground_truth`. |
| **Faithfulness** | Evaluator LLM decomposes the answer into atomic claims → checks each claim against the contexts → `supported_claims / total_claims`. |
| **Answer Relevancy** | Generates N synthetic questions from the answer → computes cosine similarity between original question and synthetic questions → average. |
| **Answer Correctness** | F1 overlap of claims between answer and ground_truth (factual) + semantic similarity (weighted combination). |
| **Answer Similarity** | Pure cosine similarity between answer and ground_truth embeddings. |
| **Hallucination Rate** | `1 − Faithfulness`. Higher = worse. |
| **Groundedness** | Same as Faithfulness. Higher = better. |

---

## User Review Required

> [!IMPORTANT]
> **Evaluator LLM Choice**: The RAGAS evaluation layer needs its own LLM to act as a "judge". I've defaulted to `gpt-4o-mini` (via OpenAI API). If you prefer to use a HuggingFace model for evaluation too, let me know — but be aware that evaluation quality is significantly better with GPT-4 class models.

> [!IMPORTANT]
> **Golden Dataset Generation**: When you upload the golden document, should I:
> - **(A)** Fully auto-generate Q&A pairs using the LLM (faster, less manual work)
> - **(B)** Generate drafts and output a CSV/JSON for you to manually review and edit before running evaluation

> [!WARNING]
> **API Key Requirement**: RAGAS evaluation requires an OpenAI API key (for the evaluator LLM and embedding model). Your current `.env` already has `OPENAI_API_KEY` and `EMBEDDING_OPENAI_API_KEY` fields — we'll reuse those. Make sure a valid OpenAI key is set.

## Open Questions

1. **How many Q&A pairs** should we target in the golden dataset? (Recommended: 25–50 for meaningful statistical coverage)
2. **Question categories**: Should we include multi-hop reasoning questions (requiring info from multiple chunks), or stick to single-chunk factual questions?
3. **Do you want a reranker** in the pipeline before evaluation? Your diagram shows a "Reranker" step between Retrieved Chunks and Top-K Context, but the current code doesn't implement one yet.

---

## Verification Plan

### Automated Tests

```bash
# 1. Compile check — all new Python files parse correctly
python -m compileall backend/evaluation/

# 2. Generate golden dataset from a test PDF
python -m evaluation.run_evaluation --generate --pdf test.pdf --output evaluation/golden_dataset.json

# 3. Run full evaluation
python -m evaluation.run_evaluation --golden-dataset evaluation/golden_dataset.json --output evaluation/report.json

# 4. Verify report structure
python -c "import json; r=json.load(open('backend/evaluation/report.json')); assert 'summary' in r; assert 'aggregate_scores' in r; print('✅ Report valid')"
```

### Manual Verification

- Review generated `golden_dataset.json` for question quality
- Check all 12 metrics appear in the report
- Verify hallucination_rate = 1 − faithfulness
- Compare scores against threshold table
- Confirm API endpoints work via Swagger at `/docs`
