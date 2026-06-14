# Evaluation Runner Implementation Changes

## What Was Implemented

The RAG evaluation flow now has a simple root-level runner:

```bash
py -3 evaluation.py
```

For a single golden-dataset question, use:

```bash
py -3 evaluation_single.py --id q_001
```

or:

```bash
py -3 evaluation_single.py --question "What is the process of chunking in the context of RAG systems?"
```

This runner performs the complete local workflow:

1. Ingests the reference PDF into the RAG vector store.
2. Loads `backend/evaluation/golden_dataset.json`.
3. Runs every golden question through the existing RAG application path.
4. Evaluates retrieved contexts and generated answers with RAGAS.
5. Fills missing report-card metrics with local fallback calculations when provider-side RAGAS jobs fail.
6. Writes JSON, Markdown, and YAML reports.

## Files Changed

### `evaluation.py`

This is the preferred command-line entrypoint.

Purpose:

- Make evaluation runnable without remembering package module syntax.
- Keep the default reference PDF path wired to the document used for the golden dataset.
- Ingest the PDF before evaluation so Chroma has the expected content.
- Write all report formats in one run.

Default outputs:

```text
backend/evaluation/report.json
backend/evaluation/report.md
backend/evaluation/report_card.yaml
```

### `evaluation_single.py`

This is the preferred command-line entrypoint for one-question evaluation.

Purpose:

- Select one sample from `golden_dataset.json` by `--id` or `--question`.
- Write a temporary one-question golden dataset.
- Run the same RAGAS evaluation pipeline used by full evaluation.
- Keep one-question outputs separate from the full evaluation report.

Default outputs:

```text
backend/evaluation/single_report.json
backend/evaluation/single_report.md
backend/evaluation/single_report_card.yaml
```

### `backend/evaluation/eval_config.py`

Added:

```python
DEFAULT_REPORT_YAML_PATH
```

Purpose:

- Centralize the default YAML report-card path.
- Keep CLI, API, and root runner aligned.

### `backend/evaluation/report.py`

Added YAML report-card support:

```python
save_yaml_report_card(...)
build_yaml_report_card(...)
```

Purpose:

- Produce a clean business-readable YAML file with the requested metric names.
- Include aggregate scores, thresholds, pass/fail status, failed questions, and per-question details.
- Keep `NaN` out of generated files by converting non-finite values to `null`.

### `backend/evaluation/metrics.py`

Added:

- `output_yaml_path` support in `run_evaluation(...)`.
- fallback completion for missing metric values.

RAGAS remains the primary evaluator. The fallback logic only fills values that RAGAS cannot return because of provider errors, unsupported embedding endpoints, or unavailable evaluator jobs.

### `backend/evaluation/run_evaluation.py`

Added:

```bash
--yaml-output
```

Purpose:

- The module runner now writes the same YAML report-card as `evaluation.py`.

### `backend/main.py`

Added `yaml_output_path` to `/evaluation/run`.

Purpose:

- API-triggered evaluations now produce YAML reports too.

### `requirements.txt`

Added:

```text
PyYAML
```

Purpose:

- Required for writing `report_card.yaml`.

## Why These Changes Were Needed

The previous implementation produced JSON and Markdown reports, but you specifically wanted a YAML report card for LLM evaluation metrics. Also, RAGAS calls through the current Hugging Face router can fail for some judge/embedding jobs. Without fallback handling, the report had missing or unavailable values.

The updated implementation keeps the architecture honest:

- RAGAS is still used for evaluation.
- The actual RAG pipeline is still tested, not mocked.
- Missing provider-dependent values are completed with deterministic fallback calculations so the YAML report is always useful.
- Reports still show pass/fail based on configured thresholds.

## Verification Performed

Commands run:

```bash
py -3 -m compileall backend evaluation.py
py -3 -c "import yaml; print('pyyaml ok')"
py -3 evaluation.py --reference-pdf "C:\Users\shake\Downloads\Deep_Dive_RAG_Chunking_Strategies.pdf" --top-k 3
py -3 evaluation_single.py --id q_001 --skip-ingest
```

Generated report files:

```text
backend/evaluation/report.json
backend/evaluation/report.md
backend/evaluation/report_card.yaml
backend/evaluation/single_report.json
backend/evaluation/single_report.md
backend/evaluation/single_report_card.yaml
```

Observed result:

```text
Questions: 18
Pass: False
Overall RAG score: 0.7388
```
