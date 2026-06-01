from huggingface_hub import InferenceClient
import config

_client = InferenceClient(token=config.HF_API_KEY)

def build_prompt(system_prompt: str, context: str, history: list, question: str) -> list:
    """Build a messages array with proper system/user/assistant roles."""
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"}
    ]
    for msg in history[-6:]:
        role = msg.get("role", "user")
        # Normalise any legacy 'model' role from older frontend sessions
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": question})
    return messages

def ask_llm(messages: list, temperature: float = 0.2, max_tokens: int = config.MAX_NEW_TOKENS) -> str:
    try:
        response = _client.chat_completion(
            messages=messages,
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "503" in err or "unavailable" in err.lower():
            return "⏳ Model is warming up on Hugging Face servers. Please retry in 20–30 seconds."
        raise
