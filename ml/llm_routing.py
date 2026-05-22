"""Маршрутизация облачных LLM: Groq, Gemini, OpenAI-совместимые (OpenRouter и др.)."""

from __future__ import annotations

import os
from typing import Optional


def _s(name: str) -> str:
    return (os.getenv(name) or "").strip()


def primary_cloud_llm() -> str:
    """
    Документы: JSON-фичи, compliance-LLM, резервный OCR.
    LLM_PROVIDER=openai|groq|gemini|none — явно.
    Иначе: OpenAI-совместимый (OpenRouter) → Gemini → Groq → none.
    """
    p = _s("LLM_PROVIDER").lower()
    if p == "openai" and _s("OPENAI_API_KEY"):
        return "openai"
    if p in ("groq", "gemini", "none"):
        return p
    if _s("OPENAI_API_KEY"):
        return "openai"
    if _s("GEMINI_API_KEY"):
        return "gemini"
    if _s("GROQ_API_KEY"):
        return "groq"
    return "none"


def openai_doc_model() -> str:
    return (
        _s("OPENAI_DOC_MODEL")
        or _s("OPENAI_EXPERT_MODEL")
        or "openai/gpt-oss-120b:free"
    ).strip()


def openai_doc_chat(
    *,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.15,
    max_tokens: int = 8192,
) -> str:
    """Извлечение JSON-фич из PDF через OpenRouter / OpenAI / DeepSeek и т.д."""
    okey = _s("OPENAI_API_KEY")
    if not okey:
        raise RuntimeError("OPENAI_API_KEY не задан")
    base = _s("OPENAI_API_BASE").strip() or None
    return openai_compatible_chat(
        api_key=okey,
        base_url=base or "https://api.openai.com/v1",
        model=openai_doc_model(),
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def expert_opinion_provider() -> str:
    """
    Экспертное заключение. EXPERT_OPINION_PROVIDER имеет приоритет
    (OpenAI-совместимые API: OpenAI, DeepSeek, NVIDIA NIM, OpenRouter).
    Иначе Gemini при GEMINI_API_KEY; Groq — только без Gemini и с коротким промптом.
    """
    p = _s("EXPERT_OPINION_PROVIDER").lower()
    if p in ("openai", "gpt", "chatgpt") and _s("OPENAI_API_KEY"):
        return "openai"
    if p == "groq" and _s("GROQ_API_KEY"):
        return "groq"
    if p == "gemini" and _s("GEMINI_API_KEY"):
        return "gemini"
    if _s("GEMINI_API_KEY"):
        return "gemini"
    if _s("OPENAI_API_KEY"):
        return "openai"
    if _s("GROQ_API_KEY"):
        return "groq"
    return "gemini"


def expert_opinion_available() -> bool:
    prov = expert_opinion_provider()
    if prov == "groq":
        return bool(_s("GROQ_API_KEY"))
    if prov in ("openai", "gpt", "chatgpt"):
        return bool(_s("OPENAI_API_KEY"))
    return bool(_s("GEMINI_API_KEY"))


def openai_compatible_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.2,
    max_tokens: int = 8192,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Пустой ответ модели")
    return text


def groq_chat(
    *,
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> str:
    gkey = _s("GROQ_API_KEY")
    if not gkey:
        raise RuntimeError("GROQ_API_KEY не задан")
    m = (model or _s("GROQ_CHAT_MODEL") or _s("GROQ_EXPERT_MODEL") or "llama-3.1-8b-instant").strip()
    return openai_compatible_chat(
        api_key=gkey,
        base_url="https://api.groq.com/openai/v1",
        model=m,
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def groq_vision_ocr_page(*, png_bytes: bytes, prompt: str, model: Optional[str] = None) -> str:
    import base64

    gkey = _s("GROQ_API_KEY")
    if not gkey:
        raise RuntimeError("GROQ_API_KEY не задан")
    m = (model or _s("GROQ_OCR_MODEL") or "llama-3.2-11b-vision-preview").strip()
    b64 = base64.standard_b64encode(png_bytes).decode("ascii")
    url = f"data:image/png;base64,{b64}"
    from openai import OpenAI

    client = OpenAI(api_key=gkey, base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return (resp.choices[0].message.content or "").strip()
