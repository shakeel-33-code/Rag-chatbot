import argparse
import json
import os
import sys
from pathlib import Path


ROOT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from evaluation.eval_config import DEFAULT_GOLDEN_DATASET_PATH  # noqa: E402
from evaluation.metrics import run_evaluation  # noqa: E402
from ingest import ingest_pdf  # noqa: E402
from observability import setup_observability  # noqa: E402


DEFAULT_REFERENCE_PDF = r"C:\Users\shake\Downloads\Deep_Dive_RAG_Chunking_Strategies.pdf"
DEFAULT_SINGLE_JSON_OUTPUT = os.path.join(BACKEND_DIR, "evaluation", "single_report.json")
DEFAULT_SINGLE_MD_OUTPUT = os.path.join(BACKEND_DIR, "evaluation", "single_report.md")
DEFAULT_SINGLE_YAML_OUTPUT = os.path.join(BACKEND_DIR, "evaluation", "single_report_card.yaml")
DEFAULT_FILTERED_DATASET = os.path.join(os.environ.get("TEMP", r"C:\tmp"), "golden_dataset_single.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAGAS-based evaluation for one question from the golden dataset."
    )
    parser.add_argument("--golden-dataset", default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--id", default="q_001", help="Golden dataset sample id to evaluate.")
    parser.add_argument("--question", help="Exact or case-insensitive question text to evaluate.")
    parser.add_argument("--reference-pdf", default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-output", default=DEFAULT_SINGLE_JSON_OUTPUT)
    parser.add_argument("--markdown-output", default=DEFAULT_SINGLE_MD_OUTPUT)
    parser.add_argument("--yaml-output", default=DEFAULT_SINGLE_YAML_OUTPUT)
    parser.add_argument("--filtered-dataset-output", default=DEFAULT_FILTERED_DATASET)
    args = parser.parse_args()

    setup_observability()

    if not args.skip_ingest:
        if not os.path.exists(args.reference_pdf):
            raise SystemExit(f"Reference PDF was not found: {args.reference_pdf}")
        with open(args.reference_pdf, "rb") as file:
            chunks_ingested = ingest_pdf(file.read(), os.path.basename(args.reference_pdf))
        print(f"Ingested {chunks_ingested} chunks from {args.reference_pdf}")

    filtered_dataset_path, selected_sample = _write_single_question_dataset(
        golden_dataset_path=args.golden_dataset,
        sample_id=args.id,
        question=args.question,
        output_path=args.filtered_dataset_output,
    )
    print(f"Selected sample: {selected_sample['id']}")
    print(f"Question: {selected_sample['question']}")

    report = run_evaluation(
        golden_dataset_path=filtered_dataset_path,
        config={"top_k": args.top_k},
        output_json_path=args.json_output,
        output_markdown_path=args.markdown_output,
        output_yaml_path=args.yaml_output,
    ).to_dict()

    print("Single-question evaluation complete")
    print(f"Pass: {report['summary']['pass']}")
    print(f"Overall RAG score: {report['summary']['overall_rag_score']}")
    print(f"YAML report card: {args.yaml_output}")
    _print_requested_metrics(report)


def _write_single_question_dataset(
    *,
    golden_dataset_path: str,
    sample_id: str,
    question: str | None,
    output_path: str,
) -> tuple[str, dict]:
    with open(golden_dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    samples = dataset.get("samples", [])
    selected_sample = _select_sample(samples, sample_id=sample_id, question=question)
    payload = {
        "metadata": {
            **dataset.get("metadata", {}),
            "total_questions": 1,
            "filtered_from": golden_dataset_path,
            "filtered_sample_id": selected_sample.get("id"),
        },
        "samples": [selected_sample],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(output), selected_sample


def _select_sample(samples: list[dict], *, sample_id: str, question: str | None) -> dict:
    if question:
        normalized_question = _normalize_question(question)
        for sample in samples:
            if _normalize_question(sample.get("question", "")) == normalized_question:
                return sample
        raise SystemExit(f"Question not found in golden dataset: {question}")

    for sample in samples:
        if sample.get("id") == sample_id:
            return sample

    raise SystemExit(f"Sample id not found in golden dataset: {sample_id}")


def _normalize_question(question: str) -> str:
    return question.strip().rstrip("?").casefold()


def _print_requested_metrics(report: dict) -> None:
    metrics = report.get("aggregate_scores", {})
    labels = {
        "Context precision": "context_precision",
        "Context recall": "context_recall",
        "Answer relevance": "answer_relevancy",
        "Faithfulness": "faithfulness",
        "Groundedness": "groundedness_score",
        "Retrieval score": "retrieval_f1",
        "Hallucination rate": "hallucination_rate",
    }
    print("Metrics:")
    for label, key in labels.items():
        print(f"  {label}: {metrics.get(key)}")


if __name__ == "__main__":
    main()
