# LLM Evaluation Flow

## Is RAGAS Used?

Yes. The evaluation implementation uses the RAGAS framework as the primary scoring engine.

The key code path is:

```python
from datasets import Dataset
from ragas import evaluate
```

inside:

```text
backend/evaluation/metrics.py
```

The function that calls RAGAS is:

```python
_compute_ragas_metrics(...)
```

## End-to-End Flow

```text
evaluation.py
  -> ingest reference PDF
  -> load golden_dataset.json
  -> run each question through the RAG app
  -> collect retrieved chunks and generated answers
  -> send rows to RAGAS
  -> compute fallback values for missing metrics
  -> compute derived metrics
  -> write report.json, report.md, report_card.yaml
```

## Step 1: Ingest Reference PDF

The runner uses:

```python
ingest_pdf(...)
```

from:

```text
backend/ingest.py
```

This parses the PDF, chunks it with the same production chunking settings, embeds the chunks, and stores them in Chroma.

Purpose:

- The evaluator must test the real retriever against the same source document used to create the golden dataset.

## Step 2: Load Golden Dataset

Input:

```text
backend/evaluation/golden_dataset.json
```

Each sample contains:

```text
question
ground_truth
expected_contexts
metadata
```

Purpose:

- `question` is sent to the RAG app.
- `ground_truth` is the expected answer.
- `expected_contexts` are the chunks the retriever should ideally return.

## Step 3: Run The RAG App

For each golden question:

```python
_run_single_question(...)
```

calls:

```python
retrieve(...)
build_prompt(...)
ask_llm(...)
```

This is the same core RAG behavior used by `/chat`.

Collected per question:

```text
retrieved contexts
generated answer
reference answer
expected contexts
retrieval latency
generation latency
token estimates
chunk count
```

## Step 4: RAGAS Evaluation

The rows are converted into a Hugging Face `Dataset` with RAGAS-compatible fields:

```text
user_input
response
retrieved_contexts
reference
```

RAGAS metrics loaded:

```text
context_precision
context_recall
context_entity_recall
faithfulness
answer_relevancy
answer_correctness
answer_similarity
```

RAGAS compares:

- retrieved chunks against the reference answer and expected evidence
- generated answer against the retrieved context
- generated answer against the golden answer

## Requested YAML Report-Card Metrics

### Context Precision

Checks whether retrieved chunks are relevant and ranked well.

High score means the retriever is not adding much noise.

### Context Recall

Checks whether the retriever found the expected evidence needed to answer the question.

High score means the retriever is not missing important chunks.

### Answer Relevance

Checks whether the generated answer actually answers the question.

High score means the answer stays on-topic.

### Faithfulness

Checks whether the generated answer is supported by the retrieved context.

High score means the answer is grounded in the retrieved chunks.

### Groundedness

Same value as faithfulness in this implementation.

Purpose:

- Make grounded answer quality explicit in the report card.

### Retrieval Score

Computed as retrieval F1:

```text
2 * context_precision * context_recall / (context_precision + context_recall)
```

Purpose:

- Combines retrieval precision and recall into one retriever-quality score.

### Hallucination Rate

Computed as:

```text
1 - faithfulness
```

Lower is better.

Purpose:

- Show the estimated amount of answer content not grounded in retrieved context.

## Fallback Metric Completion

RAGAS remains the primary evaluator.

However, the current Hugging Face router can fail for some RAGAS judge or embedding jobs. When a RAGAS metric is unavailable, the implementation computes a deterministic fallback so the YAML report is still complete.

Fallback examples:

- Context precision/recall use overlap between retrieved contexts and expected contexts.
- Answer relevance uses overlap between question terms and answer terms.
- Faithfulness uses overlap between answer terms and retrieved context terms.
- Answer correctness and similarity use overlap between generated answer and golden answer.

This keeps the report operational while still preserving RAGAS as the main evaluation framework.

## Output Files

```text
backend/evaluation/report.json
backend/evaluation/report.md
backend/evaluation/report_card.yaml
```

The YAML report-card contains:

```text
report_card.summary
report_card.metrics
report_card.failed_questions
per_question_results
```

## How To Run

From the project root:

```bash
py -3 evaluation.py
```

Run one golden-dataset question:

```bash
py -3 evaluation_single.py --id q_001
```

or:

```bash
py -3 evaluation_single.py --question "What is the process of chunking in the context of RAG systems?"
```

Optional explicit command:

```bash
py -3 evaluation.py --reference-pdf "C:\Users\shake\Downloads\Deep_Dive_RAG_Chunking_Strategies.pdf" --top-k 3
```

Skip ingestion if the PDF is already loaded:

```bash
py -3 evaluation.py --skip-ingest
```

## Phoenix Tracing

If Phoenix is running at `localhost:6006`, evaluation creates spans:

```text
evaluation.run
evaluation.questions
evaluation.question
evaluation.metrics
evaluation.ragas
evaluation.question.scores
evaluation.report.build
evaluation.report.save
```

Metric attributes appear as:

```text
evaluation.metric.context_precision
evaluation.metric.context_recall
evaluation.metric.answer_relevancy
evaluation.metric.faithfulness
evaluation.metric.groundedness_score
evaluation.metric.retrieval_f1
evaluation.metric.hallucination_rate
```
