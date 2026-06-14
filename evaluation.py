import argparse
import os
import sys


ROOT_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from evaluation.eval_config import (  # noqa: E402
    DEFAULT_GOLDEN_DATASET_PATH,
    DEFAULT_REPORT_JSON_PATH,
    DEFAULT_REPORT_MD_PATH,
    DEFAULT_REPORT_YAML_PATH,
)
from evaluation.metrics import run_evaluation  # noqa: E402
from ingest import ingest_pdf  # noqa: E402
from observability import setup_observability  # noqa: E402


DEFAULT_REFERENCE_PDF = r"C:\Users\shake\Downloads\Deep_Dive_RAG_Chunking_Strategies.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS-based LLM evaluation for the RAG app.")
    parser.add_argument("--golden-dataset", default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--reference-pdf", default=DEFAULT_REFERENCE_PDF)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json-output", default=DEFAULT_REPORT_JSON_PATH)
    parser.add_argument("--markdown-output", default=DEFAULT_REPORT_MD_PATH)
    parser.add_argument("--yaml-output", default=DEFAULT_REPORT_YAML_PATH)
    args = parser.parse_args()

    setup_observability()

    if not args.skip_ingest:
        if not os.path.exists(args.reference_pdf):
            raise SystemExit(f"Reference PDF was not found: {args.reference_pdf}")
        with open(args.reference_pdf, "rb") as file:
            chunks_ingested = ingest_pdf(file.read(), os.path.basename(args.reference_pdf))
        print(f"Ingested {chunks_ingested} chunks from {args.reference_pdf}")

    report = run_evaluation(
        golden_dataset_path=args.golden_dataset,
        config={"top_k": args.top_k},
        output_json_path=args.json_output,
        output_markdown_path=args.markdown_output,
        output_yaml_path=args.yaml_output,
    ).to_dict()

    print("Evaluation complete")
    print(f"Questions: {report['summary']['total_questions']}")
    print(f"Pass: {report['summary']['pass']}")
    print(f"Overall RAG score: {report['summary']['overall_rag_score']}")
    print(f"YAML report card: {args.yaml_output}")


if __name__ == "__main__":
    main()
