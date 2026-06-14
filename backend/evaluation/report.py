import json
import math
import os
from datetime import datetime, timezone
from typing import Any

from evaluation.eval_config import EVAL_THRESHOLDS, LOWER_IS_BETTER


REPORT_CARD_METRICS = {
    "Context precision": "context_precision",
    "Context recall": "context_recall",
    "Answer relevance": "answer_relevancy",
    "Faithfulness": "faithfulness",
    "Groundedness": "groundedness_score",
    "Retrieval score": "retrieval_f1",
    "Hallucination rate": "hallucination_rate",
}


def build_report(
    aggregate_scores: dict[str, float | None],
    per_question_scores: list[dict[str, Any]],
    config_used: dict[str, Any],
) -> dict[str, Any]:
    aggregate_scores = _clean_report_value(aggregate_scores)
    per_question_scores = _clean_report_value(per_question_scores)
    failed_questions = _failed_questions(per_question_scores)
    summary_pass = _passes_thresholds(aggregate_scores)
    return {
        "summary": {
            "overall_rag_score": aggregate_scores.get("overall_rag_score"),
            "pass": summary_pass,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_questions": len(per_question_scores),
        },
        "aggregate_scores": aggregate_scores,
        "per_question_scores": per_question_scores,
        "failed_questions": failed_questions,
        "thresholds": EVAL_THRESHOLDS,
        "config_used": config_used,
    }


def save_report(report: dict[str, Any], json_path: str, markdown_path: str | None = None) -> None:
    report = _clean_report_value(report)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)

    if markdown_path:
        with open(markdown_path, "w", encoding="utf-8") as file:
            file.write(render_markdown_report(report))


def save_yaml_report_card(report: dict[str, Any], yaml_path: str) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write YAML evaluation report cards.") from exc

    report_card = build_yaml_report_card(report)
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(report_card, file, sort_keys=False, allow_unicode=True)


def build_yaml_report_card(report: dict[str, Any]) -> dict[str, Any]:
    report = _clean_report_value(report)
    aggregate_scores = report.get("aggregate_scores", {})
    thresholds = report.get("thresholds", {})
    summary = report.get("summary", {})

    return {
        "report_card": {
            "summary": {
                "timestamp": summary.get("timestamp"),
                "total_questions": summary.get("total_questions"),
                "pass": summary.get("pass"),
                "overall_rag_score": summary.get("overall_rag_score"),
            },
            "metrics": {
                label: {
                    "score": aggregate_scores.get(metric_key),
                    "threshold": thresholds.get(metric_key),
                    "status": _metric_status(
                        metric_key,
                        aggregate_scores.get(metric_key),
                        thresholds.get(metric_key),
                    ),
                }
                for label, metric_key in REPORT_CARD_METRICS.items()
            },
            "failed_questions": report.get("failed_questions", []),
        },
        "per_question_results": [
            _build_question_report_card(row, thresholds)
            for row in report.get("per_question_scores", [])
        ],
    }


def _build_question_report_card(row: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "question": row.get("question"),
        "generated_answer": row.get("answer"),
        "reference_answer": row.get("ground_truth"),
        "retrieved_contexts": row.get("contexts", []),
        "expected_contexts": row.get("expected_contexts", []),
        "metrics": {
            label: {
                "score": row.get(metric_key),
                "threshold": thresholds.get(metric_key),
                "status": _metric_status(metric_key, row.get(metric_key), thresholds.get(metric_key)),
            }
            for label, metric_key in REPORT_CARD_METRICS.items()
        },
        "diagnostics": {
            "retrieval_latency_ms": row.get("retrieval_latency_ms"),
            "generation_latency_ms": row.get("generation_latency_ms"),
            "chunk_count_used": row.get("chunk_count_used"),
            "token_usage": row.get("token_usage"),
        },
    }


def load_latest_report(json_path: str) -> dict[str, Any]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No evaluation report found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Timestamp: {summary['timestamp']}",
        f"- Questions evaluated: {summary['total_questions']}",
        f"- Overall RAG score: {_format_score(summary.get('overall_rag_score'))}",
        f"- Pass: {summary['pass']}",
        "",
        "## Aggregate Scores",
        "",
        "| Metric | Score | Threshold | Status |",
        "|---|---:|---:|---|",
    ]

    for metric, score in report["aggregate_scores"].items():
        threshold = report["thresholds"].get(metric)
        status = _metric_status(metric, score, threshold)
        lines.append(
            f"| {metric} | {_format_score(score)} | {_format_score(threshold)} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Failed Questions",
            "",
        ]
    )
    if report["failed_questions"]:
        for item in report["failed_questions"]:
            lines.append(f"- {item['id']}: {item['question']}")
    else:
        lines.append("No per-question threshold failures.")

    lines.append("")
    return "\n".join(lines)


def _passes_thresholds(scores: dict[str, float | None]) -> bool:
    for metric, threshold in EVAL_THRESHOLDS.items():
        score = scores.get(metric)
        if not _is_finite_number(score):
            return False
        if metric in LOWER_IS_BETTER:
            if score > threshold:
                return False
        elif score < threshold:
            return False
    return True


def _failed_questions(per_question_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed = []
    for row in per_question_scores:
        score_failures = {}
        for metric, threshold in EVAL_THRESHOLDS.items():
            if metric == "overall_rag_score":
                continue
            score = row.get(metric)
            if not _is_finite_number(score):
                continue
            if metric in LOWER_IS_BETTER and score > threshold:
                score_failures[metric] = score
            elif metric not in LOWER_IS_BETTER and score < threshold:
                score_failures[metric] = score
        if score_failures:
            failed.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "failed_metrics": score_failures,
                }
            )
    return failed


def _metric_status(metric: str, score: float | None, threshold: float | None) -> str:
    if not _is_finite_number(score) or threshold is None:
        return "n/a"
    if metric in LOWER_IS_BETTER:
        return "pass" if score <= threshold else "fail"
    return "pass" if score >= threshold else "fail"


def _format_score(score: float | None) -> str:
    if not _is_finite_number(score):
        return "n/a"
    return f"{score:.3f}"


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _clean_report_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_report_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_report_value(item) for item in value]
    return value
