import argparse
import os

from evaluation.eval_config import (
    DEFAULT_GOLDEN_DATASET_PATH,
    DEFAULT_REPORT_JSON_PATH,
    DEFAULT_REPORT_YAML_PATH,
)
from evaluation.golden_dataset_generator import generate_golden_dataset
from evaluation.metrics import run_evaluation
from observability import setup_observability


def main() -> None:
    setup_observability()

    parser = argparse.ArgumentParser(description="Run offline RAG evaluation.")
    parser.add_argument("--golden-dataset", default=DEFAULT_GOLDEN_DATASET_PATH)
    parser.add_argument("--output", default=DEFAULT_REPORT_JSON_PATH)
    parser.add_argument("--markdown-output")
    parser.add_argument("--yaml-output", default=DEFAULT_REPORT_YAML_PATH)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--pdf")
    parser.add_argument("--questions-per-chunk", type=int, default=1)
    args = parser.parse_args()

    if args.generate:
        if not args.pdf:
            raise SystemExit("--pdf is required when --generate is used.")
        generate_golden_dataset(
            pdf_path=args.pdf,
            output_path=args.golden_dataset,
            questions_per_chunk=args.questions_per_chunk,
        )

    config = {}
    if args.top_k is not None:
        config["top_k"] = args.top_k

    markdown_output = args.markdown_output
    if markdown_output is None:
        markdown_output = os.path.splitext(args.output)[0] + ".md"

    report = run_evaluation(
        golden_dataset_path=args.golden_dataset,
        config=config,
        output_json_path=args.output,
        output_markdown_path=markdown_output,
        output_yaml_path=args.yaml_output,
    )
    print(report.to_dict()["summary"])


if __name__ == "__main__":
    main()
