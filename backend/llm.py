from openai import OpenAI

import config
from observability import (
    set_attribute,
    set_input,
    set_output,
    set_prompt_template,
    timed_span,
)

_client = None
_client_config = None


def build_prompt(system_prompt: str, context: str, history: list, question: str) -> list:
    """Build a messages array with proper system/user/assistant roles."""
    with timed_span(
        "prompt_building",
        "prompt_building.duration_ms",
        {
            "prompt.history_messages_received": len(history),
            "prompt.history_messages_used": len(history[-6:]),
            "prompt.context_chars": len(context),
            "prompt.question_chars": len(question),
            "prompt.context_tokens_estimate": len(context) // 4,
        },
        span_kind="PROMPT",
    ) as span:
        set_prompt_template(
            span,
            template="{system_prompt}\n\nContext:\n{context}",
            variables={
                "system_prompt": system_prompt,
                "context": context,
                "question": question,
                "history_messages_used": len(history[-6:]),
            },
        )
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"}
        ]
        for msg in history[-6:]:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": question})
        set_attribute(span, "prompt.message_count", len(messages))
        set_input(
            span,
            {
                "system_prompt": system_prompt,
                "question": question,
                "history": history[-6:],
                "retrieved_context": context,
            },
            mime_type="application/json",
        )
        set_output(span, messages, mime_type="application/json")
        return messages


def ask_llm(messages: list, temperature: float = 0.2, max_tokens: int = config.MAX_NEW_TOKENS) -> str:
    model_candidates = _model_candidates()
    if not model_candidates:
        raise RuntimeError("No OpenAI-compatible LLM model candidates are configured.")

    last_exception = None
    for model in model_candidates:
        try:
            response = _get_client().chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            answer = (response.choices[0].message.content or "").strip()
            return answer
        except Exception as exc:
            last_exception = exc
            err = str(exc)
            if config.get_llm_provider() == "huggingface" and (
                "503" in err or "unavailable" in err.lower()
            ):
                return "Model is warming up on Hugging Face servers. Please retry in 20-30 seconds."
            if _should_try_next_model(err, model, model_candidates):
                continue
            raise

    if last_exception is not None:
        raise last_exception

    raise RuntimeError("No OpenAI-compatible LLM model candidates are configured.")


def _model_candidates() -> list:
    candidates = [config.get_llm_model(), *config.get_llm_fallback_models()]
    deduped = []
    for model in candidates:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _get_client() -> OpenAI:
    global _client, _client_config

    api_key = config.get_llm_api_key()
    base_url = config.get_llm_base_url()
    if not api_key:
        raise RuntimeError(
            "The selected LLM provider is missing its API key in .env."
        )

    current_config = (config.get_llm_provider(), api_key, base_url)
    if _client is None or current_config != _client_config:
        _client = OpenAI(api_key=api_key, base_url=base_url)
        _client_config = current_config

    return _client


def _should_try_next_model(error: str, model: str, candidates: list) -> bool:
    if model == candidates[-1]:
        return False

    retryable_markers = (
        "model_not_supported",
        "not supported",
        "401 unauthorized",
        "403 forbidden",
        "404 not found",
    )
    error_lower = error.lower()
    return any(marker in error_lower for marker in retryable_markers)
