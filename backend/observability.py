import json
import logging
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional

import config

logger = logging.getLogger(__name__)

_initialized = False
_tracing_available = False
_tracer = None


def setup_observability(app: Optional[Any] = None) -> None:
    """Configure Phoenix export, FastAPI tracing, and OpenInference auto instrumentation."""
    global _initialized, _tracing_available, _tracer

    if _initialized:
        return

    _initialized = True

    if not config.PHOENIX_TRACING_ENABLED:
        logger.info("Phoenix tracing is disabled.")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from openinference.instrumentation import OITracer, TraceConfig
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register
    except ImportError as exc:
        logger.warning("Phoenix/OpenInference tracing dependencies are not installed: %s", exc)
        return
    except Exception as exc:
        logger.warning("Phoenix/OpenInference tracing imports failed; continuing without traces: %s", exc)
        return

    try:
        provider = register(
            endpoint=config.PHOENIX_COLLECTOR_ENDPOINT,
            project_name=config.PHOENIX_PROJECT_NAME,
            batch=True,
            auto_instrument=False,
            verbose=False,
        )
        _tracer = OITracer(
            trace.get_tracer("rag-project.openinference"),
            TraceConfig(
                hide_inputs=not config.PHOENIX_CAPTURE_CONTENT,
                hide_outputs=not config.PHOENIX_CAPTURE_CONTENT,
                hide_input_text=not config.PHOENIX_CAPTURE_CONTENT,
                hide_output_text=not config.PHOENIX_CAPTURE_CONTENT,
                hide_embedding_vectors=True,
            ),
        )
        OpenAIInstrumentor().instrument(
            tracer_provider=provider,
            config=TraceConfig(
                hide_inputs=not config.PHOENIX_CAPTURE_CONTENT,
                hide_outputs=not config.PHOENIX_CAPTURE_CONTENT,
                hide_input_text=not config.PHOENIX_CAPTURE_CONTENT,
                hide_output_text=not config.PHOENIX_CAPTURE_CONTENT,
                hide_embedding_vectors=True,
            ),
        )

        if app is not None:
            FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

        _tracing_available = True
        logger.info(
            "Phoenix OpenInference tracing enabled for project '%s' at %s.",
            config.PHOENIX_PROJECT_NAME,
            config.PHOENIX_COLLECTOR_ENDPOINT,
        )
    except Exception as exc:
        logger.warning("Phoenix tracing setup failed; continuing without traces: %s", exc)


def get_tracer() -> Optional[Any]:
    if not _tracing_available:
        return None
    return _tracer


@contextmanager
def start_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    span_kind: str = "CHAIN",
) -> Iterator[Any]:
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from openinference.semconv.trace import OpenInferenceSpanKindValues

        openinference_span_kind = OpenInferenceSpanKindValues[span_kind]
    except Exception:
        openinference_span_kind = span_kind

    with tracer.start_as_current_span(
        name,
        attributes=normalize_attributes(attributes or {}),
        openinference_span_kind=openinference_span_kind,
    ) as span:
        yield span


@contextmanager
def timed_span(
    name: str,
    duration_attribute: str,
    attributes: Optional[Dict[str, Any]] = None,
    span_kind: str = "CHAIN",
) -> Iterator[Any]:
    start = time.perf_counter()
    context = start_span(name, attributes, span_kind) if _tracing_available else nullcontext(None)

    with context as span:
        try:
            yield span
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            set_attribute(span, duration_attribute, round(duration_ms, 2))


def normalize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _normalize_attribute_value(value)
        for key, value in attributes.items()
        if value is not None
    }


def set_attribute(span: Any, key: str, value: Any) -> None:
    if span is None or value is None:
        return

    try:
        span.set_attribute(key, _normalize_attribute_value(value))
    except Exception:
        logger.debug("Failed to set span attribute '%s'.", key, exc_info=True)


def set_content_attribute(span: Any, key: str, value: Any) -> None:
    if not config.PHOENIX_CAPTURE_CONTENT or value is None:
        return

    set_attribute(span, key, _truncate_content(value))


def set_attributes(span: Any, attributes: Dict[str, Any]) -> None:
    for key, value in attributes.items():
        set_attribute(span, key, value)


def set_input(span: Any, value: Any, mime_type: str = "text/plain") -> None:
    if span is None or not config.PHOENIX_CAPTURE_CONTENT:
        return

    try:
        from openinference.instrumentation import get_input_attributes

        set_attributes(span, get_input_attributes(_truncate_content(value), mime_type=mime_type))
    except Exception:
        set_content_attribute(span, "input.value", value)
        set_attribute(span, "input.mime_type", mime_type)


def set_output(span: Any, value: Any, mime_type: str = "text/plain") -> None:
    if span is None or not config.PHOENIX_CAPTURE_CONTENT:
        return

    try:
        from openinference.instrumentation import get_output_attributes

        set_attributes(span, get_output_attributes(_truncate_content(value), mime_type=mime_type))
    except Exception:
        set_content_attribute(span, "output.value", value)
        set_attribute(span, "output.mime_type", mime_type)


def set_llm_attributes(
    span: Any,
    *,
    model_name: str,
    provider: str,
    invocation_parameters: Dict[str, Any],
    input_messages: Optional[List[Dict[str, str]]] = None,
    output_message: Optional[str] = None,
) -> None:
    if span is None:
        return

    try:
        from openinference.instrumentation import get_llm_attributes

        attrs = get_llm_attributes(
            provider=provider,
            model_name=model_name,
            invocation_parameters=invocation_parameters,
            input_messages=input_messages if config.PHOENIX_CAPTURE_CONTENT else None,
            output_messages=(
                [{"role": "assistant", "content": output_message}]
                if config.PHOENIX_CAPTURE_CONTENT and output_message is not None
                else None
            ),
        )
        set_attributes(span, attrs)
    except Exception:
        set_attribute(span, "llm.provider", provider)
        set_attribute(span, "llm.model_name", model_name)
        set_attribute(span, "llm.invocation_parameters", invocation_parameters)


def set_embedding_attributes(span: Any, *, model_name: str, input_count: int, dimension: Optional[int] = None) -> None:
    if span is None:
        return

    try:
        from openinference.instrumentation import get_embedding_attributes

        set_attributes(span, get_embedding_attributes(model_name=model_name))
    except Exception:
        set_attribute(span, "embedding.model_name", model_name)

    set_attribute(span, "embedding.input_count", input_count)
    set_attribute(span, "embedding.dimension", dimension)


def set_retrieval_documents(span: Any, documents: List[Dict[str, Any]]) -> None:
    if span is None:
        return

    try:
        from openinference.instrumentation import get_retriever_attributes

        openinference_documents = []
        for document in documents:
            item = {
                "content": _truncate_content(document.get("content", "")),
                "metadata": document.get("metadata", {}),
            }
            if document.get("id") is not None:
                item["id"] = document["id"]
            if document.get("score") is not None:
                item["score"] = document["score"]
            openinference_documents.append(item)

        set_attributes(span, get_retriever_attributes(documents=openinference_documents))
    except Exception:
        set_attribute(span, "retrieval.documents", documents)


def set_prompt_template(
    span: Any,
    *,
    template: str,
    variables: Dict[str, Any],
    version: str = "rag-v1",
) -> None:
    if span is None:
        return

    try:
        from openinference.semconv.trace import SpanAttributes

        if config.PHOENIX_CAPTURE_CONTENT:
            set_content_attribute(span, SpanAttributes.LLM_PROMPT_TEMPLATE, template)
            set_content_attribute(span, SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, variables)
        set_attribute(span, SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, version)
    except Exception:
        set_content_attribute(span, "llm.prompt_template.template", template)
        set_content_attribute(span, "llm.prompt_template.variables", variables)
        set_attribute(span, "llm.prompt_template.version", version)


def record_exception(span: Any, exc: Exception) -> None:
    if span is None:
        return

    try:
        span.record_exception(exc)
        span.set_attribute("error.type", exc.__class__.__name__)
        span.set_attribute("error.message", str(exc))
    except Exception:
        logger.debug("Failed to record span exception.", exc_info=True)


def _normalize_attribute_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, (list, tuple)) and all(isinstance(item, (str, bool, int, float)) for item in value):
        return list(value)

    return _safe_json(value)


def _truncate_content(value: Any) -> str:
    text = value if isinstance(value, str) else _safe_json(value)
    max_chars = max(config.PHOENIX_CAPTURE_CONTENT_MAX_CHARS, 0)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)
