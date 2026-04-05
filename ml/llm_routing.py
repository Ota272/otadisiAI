"""Маршрутизация облачных LLM: Groq (бесплатный tier) и Gemini."""

from __future__ import annotations

import os
from typing import Optional


def _s(name: str) -> str:
    return (os.getenv(name) or "").strip()


def primary_cloud_llm() -> str:
    """
    Документы: JSON-фичи, compliance-LLM, резервный OCR.
    LLM_PROVIDER=groq|gemini|none — явно.
    Иначе: Groq при GROQ_API_KEY, иначе Gemini при GEMINI_API_KEY, иначе none.
    """
    p = _s("LLM_PROVIDER").lower()
    if p in ("groq", "gemini", "none"):
        return p
    if _s("GROQ_API_KEY"):
        return "groq"
    if _s("GEMINI_API_KEY"):
        return "gemini"
    return "none"


def expert_opinion_provider() -> str:
    """
    Текст экспертного заключения.
    EXPERT_OPINION_PROVIDER переопределяет; иначе как primary_cloud_llm().
    Если ключей нет — 'gemini' (совместимость; generate_* проверит available).
    """
    p = _s("EXPERT_OPINION_PROVIDER").lower()
    if p in ("groq", "openai", "gpt", "chatgpt"):
        return p
    if p == "gemini":
        return "gemini"
    c = primary_cloud_llm()
    if c in ("groq", "gemini"):
        return c
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
