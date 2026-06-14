import json
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any

from evaluation.eval_config import DEFAULT_EVALUATION_CONFIG, OVERALL_WEIGHTS
from evaluation.report import build_report, save_report, save_yaml_report_card
from llm import ask_llm, build_prompt
from observability import (
    record_exception,
    set_attribute,
    set_attributes,
    set_input,
    set_output,
    start_span,
    timed_span,
)
from retriever import retrieve


@dataclass
class EvaluationReport:
    report: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.report


def run_evaluation(
    golden_dataset_path: str,
    config: dict[str, Any] | None = None,
    output_json_path: str | None = None,
    output_markdown_path: str | None = None,
    output_yaml_path: str | None = None,
) -> EvaluationReport:
    config_used = {**DEFAULT_EVALUATION_CONFIG, **(config or {})}
    with timed_span(
        "evaluation.run",
        "evaluation.duration_ms",
        {
            "evaluation.golden_dataset_path": golden_dataset_path,
            "evaluation.top_k": config_used.get("top_k"),
            "evaluation.evaluator_llm": config_used.get("evaluator_llm"),
            "evaluation.evaluator_embedding": config_used.get("evaluator_embedding"),
        },
        span_kind="CHAIN",
    ) as span:
        try:
            set_input(
                span,
                {
                    "golden_dataset_path": golden_dataset_path,
                    "config_used": _safe_observability_config(config_used),
                },
                mime_type="application/json",
            )
            samples = _load_samples(golden_dataset_path)
            set_attribute(span, "evaluation.total_questions", len(samples))

            with timed_span(
                "evaluation.questions",
                "evaluation.questions.duration_ms",
                {"evaluation.total_questions": len(samples)},
                span_kind="CHAIN",
            ) as questions_span:
                rag_rows = [_run_single_question(sample, config_used) for sample in samples]
                set_attribute(questions_span, "evaluation.questions.completed", len(rag_rows))

            with timed_span(
                "evaluation.metrics",
                "evaluation.metrics.duration_ms",
                {"evaluation.total_questions": len(rag_rows)},
                span_kind="CHAIN",
            ) as metrics_span:
                ragas_scores = _compute_ragas_metrics(rag_rows)
                per_question_scores = _merge_per_question_scores(rag_rows, ragas_scores)
                aggregate_scores = _compute_aggregate_scores(ragas_scores, per_question_scores)
                _set_aggregate_metric_attributes(metrics_span, aggregate_scores)
                set_output(
                    metrics_span,
                    {
                        "aggregate_scores": aggregate_scores,
                        "per_question_score_count": len(per_question_scores),
                    },
                    mime_type="application/json",
                )

            with timed_span(
                "evaluation.report.build",
                "evaluation.report.build.duration_ms",
                {"evaluation.total_questions": len(per_question_scores)},
                span_kind="CHAIN",
            ) as report_span:
                report = build_report(aggregate_scores, per_question_scores, config_used)
                set_attributes(
                    report_span,
                    {
                        "evaluation.pass": report["summary"]["pass"],
                        "evaluation.failed_question_count": len(report["failed_questions"]),
                        "evaluation.overall_rag_score": report["summary"].get("overall_rag_score"),
                    },
                )
                set_output(
                    report_span,
                    {
                        "summary": report["summary"],
                        "failed_question_count": len(report["failed_questions"]),
                    },
                    mime_type="application/json",
                )

            if output_json_path:
                with timed_span(
                    "evaluation.report.save",
                    "evaluation.report.save.duration_ms",
                    {
                        "evaluation.report_json_path": output_json_path,
                        "evaluation.report_markdown_path": output_markdown_path,
                        "evaluation.report_yaml_path": output_yaml_path,
                    },
                    span_kind="CHAIN",
                ) as report_save_span:
                    save_report(report, output_json_path, output_markdown_path)
                    if output_yaml_path:
                        save_yaml_report_card(report, output_yaml_path)
                    set_attribute(report_save_span, "evaluation.report.saved", True)
                    set_output(
                        report_save_span,
                        {
                            "json_path": output_json_path,
                            "markdown_path": output_markdown_path,
                            "yaml_path": output_yaml_path,
                        },
                        mime_type="application/json",
                    )

            _set_aggregate_metric_attributes(span, aggregate_scores)
            set_attribute(span, "evaluation.pass", report["summary"]["pass"])
            set_attribute(span, "evaluation.failed_question_count", len(report["failed_questions"]))
            set_output(
                span,
                {
                    "summary": report["summary"],
                    "aggregate_scores": aggregate_scores,
                    "failed_question_count": len(report["failed_questions"]),
                },
                mime_type="application/json",
            )
        except Exception as exc:
            record_exception(span, exc)
            raise

    return EvaluationReport(report=report)


def _load_samples(golden_dataset_path: str) -> list[dict[str, Any]]:
    with open(golden_dataset_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Golden dataset must contain a non-empty 'samples' list.")
    return samples


def _run_single_question(sample: dict[str, Any], config_used: dict[str, Any]) -> dict[str, Any]:
    question = sample["question"]
    with timed_span(
        "evaluation.question",
        "evaluation.question.duration_ms",
        {
            "evaluation.question_id": sample.get("id"),
            "evaluation.question_category": sample.get("metadata", {}).get("category"),
            "evaluation.question_difficulty": sample.get("metadata", {}).get("difficulty"),
            "evaluation.source_chunk_index": sample.get("metadata", {}).get("source_chunk_index"),
            "evaluation.top_k": config_used.get("top_k"),
        },
        span_kind="CHAIN",
    ) as span:
        try:
            set_input(
                span,
                {
                    "id": sample.get("id"),
                    "question": question,
                    "ground_truth": sample.get("ground_truth"),
                    "expected_context_count": len(sample.get("expected_contexts", [])),
                    "metadata": sample.get("metadata", {}),
                },
                mime_type="application/json",
            )

            retrieval_start = time.perf_counter()
            context = retrieve(question, top_k=int(config_used["top_k"]))
            retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
            contexts = _split_context(context)

            messages = build_prompt(
                system_prompt=config_used["system_prompt"],
                context=context,
                history=[],
                question=question,
            )
            generation_start = time.perf_counter()
            answer = ask_llm(
                messages,
                temperature=float(config_used["temperature"]),
                max_tokens=int(config_used["max_new_tokens"]),
            ).strip()
            generation_latency_ms = round((time.perf_counter() - generation_start) * 1000, 2)

            row = {
                "id": sample.get("id"),
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": sample["ground_truth"],
                "reference": sample["ground_truth"],
                "expected_contexts": sample.get("expected_contexts", []),
                "metadata": sample.get("metadata", {}),
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": generation_latency_ms,
                "token_usage": {
                    "input_tokens_estimate": (len(context) + len(question)) // 4,
                    "output_tokens_estimate": len(answer) // 4,
                },
                "chunk_count_used": len(contexts),
                "context_relevancy": _context_relevancy_fallback(question, contexts),
            }

            set_attributes(
                span,
                {
                    "evaluation.retrieval_latency_ms": retrieval_latency_ms,
                    "evaluation.generation_latency_ms": generation_latency_ms,
                    "evaluation.chunk_count_used": row["chunk_count_used"],
                    "evaluation.input_tokens_estimate": row["token_usage"]["input_tokens_estimate"],
                    "evaluation.output_tokens_estimate": row["token_usage"]["output_tokens_estimate"],
                    "evaluation.context_relevancy": row["context_relevancy"],
                    "evaluation.answer_chars": len(answer),
                    "evaluation.context_chars": len(context),
                },
            )
            set_output(
                span,
                {
                    "answer": answer,
                    "chunk_count_used": row["chunk_count_used"],
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "generation_latency_ms": generation_latency_ms,
                    "context_relevancy": row["context_relevancy"],
                },
                mime_type="application/json",
            )
            return row
        except Exception as exc:
            record_exception(span, exc)
            raise


def _compute_ragas_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are not installed. Run 'pip install -r requirements.txt'."
        ) from exc

    dataset = Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "question": row["question"],
                "response": row["answer"],
                "answer": row["answer"],
                "retrieved_contexts": row["contexts"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
                "reference": row["ground_truth"],
            }
            for row in rows
        ]
    )
    metrics = _load_ragas_metrics()
    kwargs = {"dataset": dataset, "metrics": metrics}
    kwargs.update(_build_ragas_judge_kwargs(evaluate))
    with timed_span(
        "evaluation.ragas",
        "evaluation.ragas.duration_ms",
        {
            "evaluation.ragas.metric_count": len(metrics),
            "evaluation.ragas.question_count": len(rows),
            "evaluation.ragas.metrics": [getattr(metric, "name", type(metric).__name__) for metric in metrics],
        },
        span_kind="CHAIN",
    ) as span:
        try:
            set_input(
                span,
                {
                    "question_count": len(rows),
                    "metrics": [getattr(metric, "name", type(metric).__name__) for metric in metrics],
                },
                mime_type="application/json",
            )
            result = evaluate(**kwargs)
            normalized = _normalize_ragas_result(result)
            _set_aggregate_metric_attributes(span, normalized["aggregate"])
            set_output(span, normalized["aggregate"], mime_type="application/json")
            return normalized
        except Exception as exc:
            record_exception(span, exc)
            raise


def _build_ragas_judge_kwargs(evaluate_func: Any) -> dict[str, Any]:
    signature = inspect.signature(evaluate_func)
    kwargs = {}
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from evaluation import eval_config
    except ImportError:
        return kwargs

    if not eval_config.EVALUATOR_OPENAI_API_KEY:
        return kwargs

    if "llm" in signature.parameters:
        kwargs["llm"] = ChatOpenAI(
            model=eval_config.EVALUATOR_LLM,
            api_key=eval_config.EVALUATOR_OPENAI_API_KEY,
            base_url=eval_config.EVALUATOR_OPENAI_BASE_URL,
            temperature=0,
        )
    if "embeddings" in signature.parameters:
        kwargs["embeddings"] = OpenAIEmbeddings(
            model=eval_config.EVALUATOR_EMBEDDING,
            api_key=eval_config.EVALUATOR_OPENAI_API_KEY,
            base_url=eval_config.EVALUATOR_OPENAI_BASE_URL,
        )
    return kwargs


def _load_ragas_metrics() -> list[Any]:
    from ragas import metrics as legacy_ragas_metrics
    metric_modules = [legacy_ragas_metrics]

    metric_aliases = {
        "context_precision": ("context_precision",),
        "context_recall": ("context_recall",),
        "context_entity_recall": ("context_entity_recall", "context_entities_recall"),
        "faithfulness": ("faithfulness",),
        "answer_relevancy": ("answer_relevancy", "answer_relevance"),
        "answer_correctness": ("answer_correctness",),
        "answer_similarity": ("answer_similarity",),
    }

    loaded_metrics = []
    missing = []
    for name, aliases in metric_aliases.items():
        metric = None
        for metric_module in metric_modules:
            for alias in aliases:
                metric = getattr(metric_module, alias, None)
                if hasattr(metric, "metric"):
                    metric = metric.metric
                if metric is not None:
                    break
            if metric is not None:
                break
        if metric is None:
            missing.append(name)
        else:
            loaded_metrics.append(metric)

    if missing:
        raise RuntimeError(f"RAGAS installation is missing required metrics: {', '.join(missing)}")
    return loaded_metrics


def _normalize_ragas_result(result: Any) -> dict[str, Any]:
    aggregate = {}
    per_question = []

    if hasattr(result, "to_pandas"):
        dataframe = result.to_pandas()
        per_question = dataframe.to_dict(orient="records")
        for column in dataframe.columns:
            if column in {
                "user_input",
                "question",
                "response",
                "answer",
                "retrieved_contexts",
                "contexts",
                "ground_truth",
                "reference",
            }:
                continue
            values = [_to_float(value) for value in dataframe[column].tolist()]
            values = [value for value in values if value is not None]
            if values:
                aggregate[_normalize_metric_name(column)] = sum(values) / len(values)

    if hasattr(result, "scores") and isinstance(result.scores, list):
        per_question = result.scores

    if isinstance(result, dict):
        aggregate.update({_normalize_metric_name(key): _to_float(value) for key, value in result.items()})
    elif hasattr(result, "__iter__"):
        try:
            aggregate.update(
                {_normalize_metric_name(key): _to_float(value) for key, value in dict(result).items()}
            )
        except Exception:
            pass

    return {
        "aggregate": {key: value for key, value in aggregate.items() if value is not None},
        "per_question": per_question,
    }


def _compute_aggregate_scores(
    ragas_scores: dict[str, Any],
    per_question_scores: list[dict[str, Any]],
) -> dict[str, float | None]:
    aggregate = {key: _round_score(value) for key, value in ragas_scores["aggregate"].items()}
    for metric in _score_metric_names():
        if aggregate.get(metric) is None:
            aggregate[metric] = _average_metric(per_question_scores, metric)
    derived = _compute_derived_metrics(aggregate)
    return {**aggregate, **derived}


def _compute_derived_metrics(scores: dict[str, float | None]) -> dict[str, float | None]:
    faithfulness = scores.get("faithfulness")
    context_precision = scores.get("context_precision")
    context_recall = scores.get("context_recall")

    derived = {
        "hallucination_rate": _round_score(1 - faithfulness) if faithfulness is not None else None,
        "groundedness_score": faithfulness,
        "retrieval_f1": _f1(context_precision, context_recall),
        "overall_rag_score": _weighted_average(scores, OVERALL_WEIGHTS),
    }
    return derived


def _merge_per_question_scores(
    rows: list[dict[str, Any]],
    ragas_scores: dict[str, Any],
) -> list[dict[str, Any]]:
    metric_rows = ragas_scores.get("per_question") or [{} for _ in rows]
    merged = []
    for index, row in enumerate(rows):
        metric_row = metric_rows[index] if index < len(metric_rows) else {}
        normalized_metrics = {
            _normalize_metric_name(key): _round_score(value)
            for key, value in metric_row.items()
            if key not in {"question", "answer", "contexts", "ground_truth", "reference"}
            if key not in {
                "user_input",
                "question",
                "response",
                "answer",
                "retrieved_contexts",
                "contexts",
                "ground_truth",
                "reference",
            }
        }
        normalized_metrics = _apply_metric_fallbacks(row, normalized_metrics)
        derived = _compute_derived_metrics(normalized_metrics)
        merged_row = {
            **row,
            **normalized_metrics,
            **derived,
        }
        _trace_per_question_scores(merged_row)
        merged.append(merged_row)
    return merged


def _trace_per_question_scores(row: dict[str, Any]) -> None:
    with start_span(
        "evaluation.question.scores",
        {
            "evaluation.question_id": row.get("id"),
            "evaluation.question": row.get("question"),
        },
        span_kind="CHAIN",
    ) as span:
        metric_attributes = {
            f"evaluation.metric.{metric}": row.get(metric)
            for metric in _score_metric_names()
            if row.get(metric) is not None
        }
        set_attributes(span, metric_attributes)
        set_attributes(
            span,
            {
                "evaluation.retrieval_latency_ms": row.get("retrieval_latency_ms"),
                "evaluation.generation_latency_ms": row.get("generation_latency_ms"),
                "evaluation.chunk_count_used": row.get("chunk_count_used"),
            },
        )
        set_output(
            span,
            {
                "id": row.get("id"),
                "scores": {
                    metric: row.get(metric)
                    for metric in _score_metric_names()
                    if row.get(metric) is not None
                },
            },
            mime_type="application/json",
        )


def _split_context(context: str) -> list[str]:
    return [chunk.strip() for chunk in context.split("\n\n---\n\n") if chunk.strip()]


def _context_relevancy_fallback(question: str, contexts: list[str]) -> float | None:
    question_terms = _significant_terms(question)
    if not question_terms or not contexts:
        return None

    relevant_chunks = 0
    for context in contexts:
        context_terms = _significant_terms(context)
        overlap = question_terms.intersection(context_terms)
        if len(overlap) / len(question_terms) >= 0.2:
            relevant_chunks += 1

    return _round_score(relevant_chunks / len(contexts))


def _significant_terms(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
        "is", "it", "of", "on", "or", "the", "to", "what", "when", "where", "which",
        "who", "why", "with",
    }
    terms = []
    current = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            term = "".join(current)
            if len(term) > 2 and term not in stopwords:
                terms.append(term)
            current = []
    if current:
        term = "".join(current)
        if len(term) > 2 and term not in stopwords:
            terms.append(term)
    return set(terms)


def _average_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
    values = [_to_float(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return _round_score(sum(values) / len(values))


def _apply_metric_fallbacks(
    row: dict[str, Any],
    metrics: dict[str, float | None],
) -> dict[str, float | None]:
    completed = dict(metrics)
    contexts = row.get("contexts", [])
    expected_contexts = row.get("expected_contexts", [])
    question = row.get("question", "")
    answer = row.get("answer", "")
    ground_truth = row.get("ground_truth", "")

    fallback_values = {
        "context_precision": _context_precision_fallback(contexts, expected_contexts),
        "context_recall": _context_recall_fallback(contexts, expected_contexts),
        "context_entity_recall": _entity_recall_fallback(contexts, ground_truth),
        "faithfulness": _faithfulness_fallback(answer, contexts),
        "answer_relevancy": _answer_relevancy_fallback(question, answer),
        "answer_correctness": _answer_correctness_fallback(answer, ground_truth),
        "answer_similarity": _answer_similarity_fallback(answer, ground_truth),
        "context_relevancy": row.get("context_relevancy"),
    }

    for metric, fallback_score in fallback_values.items():
        if completed.get(metric) is None and fallback_score is not None:
            completed[metric] = fallback_score
    return completed


def _context_precision_fallback(contexts: list[str], expected_contexts: list[str]) -> float | None:
    if not contexts:
        return None
    relevant = sum(
        1
        for context in contexts
        if _max_text_overlap(context, expected_contexts) >= 0.2
    )
    return _round_score(relevant / len(contexts))


def _context_recall_fallback(contexts: list[str], expected_contexts: list[str]) -> float | None:
    if not expected_contexts:
        return None
    recalled = sum(
        1
        for expected_context in expected_contexts
        if _max_text_overlap(expected_context, contexts) >= 0.2
    )
    return _round_score(recalled / len(expected_contexts))


def _entity_recall_fallback(contexts: list[str], ground_truth: str) -> float | None:
    truth_terms = _significant_terms(ground_truth)
    if not truth_terms:
        return None
    context_terms = _significant_terms(" ".join(contexts))
    return _round_score(len(truth_terms.intersection(context_terms)) / len(truth_terms))


def _faithfulness_fallback(answer: str, contexts: list[str]) -> float | None:
    answer_terms = _significant_terms(answer)
    if not answer_terms:
        return None
    context_terms = _significant_terms(" ".join(contexts))
    return _round_score(len(answer_terms.intersection(context_terms)) / len(answer_terms))


def _answer_relevancy_fallback(question: str, answer: str) -> float | None:
    question_terms = _significant_terms(question)
    if not question_terms:
        return None
    answer_terms = _significant_terms(answer)
    return _round_score(len(question_terms.intersection(answer_terms)) / len(question_terms))


def _answer_correctness_fallback(answer: str, ground_truth: str) -> float | None:
    return _f1(
        _text_precision(answer, ground_truth),
        _text_recall(answer, ground_truth),
    )


def _answer_similarity_fallback(answer: str, ground_truth: str) -> float | None:
    answer_terms = _significant_terms(answer)
    truth_terms = _significant_terms(ground_truth)
    if not answer_terms or not truth_terms:
        return None
    union = answer_terms.union(truth_terms)
    return _round_score(len(answer_terms.intersection(truth_terms)) / len(union))


def _max_text_overlap(text: str, candidates: list[str]) -> float:
    if not candidates:
        return 0.0
    return max((_text_recall(text, candidate) or 0.0) for candidate in candidates)


def _text_precision(predicted: str, reference: str) -> float | None:
    predicted_terms = _significant_terms(predicted)
    reference_terms = _significant_terms(reference)
    if not predicted_terms:
        return None
    return _round_score(len(predicted_terms.intersection(reference_terms)) / len(predicted_terms))


def _text_recall(predicted: str, reference: str) -> float | None:
    predicted_terms = _significant_terms(predicted)
    reference_terms = _significant_terms(reference)
    if not reference_terms:
        return None
    return _round_score(len(predicted_terms.intersection(reference_terms)) / len(reference_terms))


def _set_aggregate_metric_attributes(span: Any, scores: dict[str, Any]) -> None:
    metric_attributes = {
        f"evaluation.metric.{metric}": _round_score(score)
        for metric, score in scores.items()
        if _round_score(score) is not None
    }
    set_attributes(span, metric_attributes)


def _score_metric_names() -> tuple[str, ...]:
    return (
        "context_precision",
        "context_recall",
        "context_relevancy",
        "context_entity_recall",
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "answer_similarity",
        "hallucination_rate",
        "groundedness_score",
        "retrieval_f1",
        "overall_rag_score",
    )


def _safe_observability_config(config_used: dict[str, Any]) -> dict[str, Any]:
    redacted = {}
    for key, value in config_used.items():
        key_lower = key.lower()
        if "api_key" in key_lower or "token" in key_lower or "secret" in key_lower:
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted


def _normalize_metric_name(metric_name: str) -> str:
    aliases = {
        "context_relevance": "context_relevancy",
        "answer_relevance": "answer_relevancy",
        "context_entities_recall": "context_entity_recall",
    }
    return aliases.get(metric_name, metric_name)


def _weighted_average(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in weights.items():
        score = scores.get(metric)
        if score is None:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return _round_score(weighted_sum / total_weight)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return _round_score(2 * precision * recall / (precision + recall))


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _round_score(value: Any) -> float | None:
    numeric = _to_float(value)
    if numeric is None:
        return None
    return round(numeric, 4)
