from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import HTTPException

from .config import config

CHAT_MODEL_PREFERENCE = [
    "gpt-5.4-mini",
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.2-chat-latest",
    "gpt-5.2",
    "gpt-4o-mini",
    "gpt-4o",
]

NON_CHAT_MODEL_TOKENS = ("image", "realtime", "audio", "tts", "transcribe")


@dataclass(frozen=True)
class GeneratedImage:
    image_bytes: bytes
    mime_type: str
    model: str
    revised_prompt: str
    size: str
    quality: str
    output_format: str


def configured_model_for_role(role: str) -> str | None:
    role_models = {
        "idea_generator": config.idea_generator_model,
        "idea_critic": config.idea_critic_model,
        "idea_finalizer": config.idea_final_model,
    }
    return role_models.get(role, "") or config.upstream_model or None


def _headers() -> dict[str, str]:
    if not config.upstream_api_key:
        raise HTTPException(status_code=500, detail="Missing UPSTREAM_API_KEY")
    return {
        "Authorization": f"Bearer {config.upstream_api_key}",
        "Content-Type": "application/json",
    }


def _auth_headers() -> dict[str, str]:
    if not config.upstream_api_key:
        raise HTTPException(status_code=500, detail="Missing UPSTREAM_API_KEY")
    return {"Authorization": f"Bearer {config.upstream_api_key}"}


async def _request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    root = (base_url or config.upstream_base_url).rstrip("/")
    url = f"{root}/{path.lstrip('/')}"
    headers = _headers() if api_key is None else {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout or config.request_timeout_seconds) as client:
        response = await client.request(method, url, headers=headers, json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()


async def list_models() -> dict[str, Any]:
    return await _request("GET", "/models")


def _normalize_model_id(model_id: str) -> str:
    return model_id.strip().lower()


def _normalize_temperature(model_id: str, temperature: float | None) -> float | None:
    if temperature is None:
        return None
    normalized = _normalize_model_id(model_id)
    if normalized.startswith("kimi-k2"):
        return 1.0
    return temperature


def _is_chat_friendly_model(model_id: str) -> bool:
    normalized = _normalize_model_id(model_id)
    if not normalized:
        return False
    if any(token in normalized for token in NON_CHAT_MODEL_TOKENS):
        return False
    if "codex-auto-review" in normalized:
        return False
    if normalized.startswith("gpt-image"):
        return False
    if "chat" in normalized:
        return True
    return normalized.startswith(("gpt-5", "gpt-4o", "o3", "o4"))


def _model_rank(model_id: str) -> tuple[int, int, str]:
    normalized = _normalize_model_id(model_id)
    if normalized in CHAT_MODEL_PREFERENCE:
        return (0, CHAT_MODEL_PREFERENCE.index(normalized), normalized)
    if normalized.startswith("gpt-5") and "chat" in normalized:
        return (1, 0, normalized)
    if normalized.startswith("gpt-5"):
        return (2, 0, normalized)
    if normalized.startswith("gpt-4o"):
        return (3, 0, normalized)
    if normalized.startswith(("o3", "o4")):
        return (4, 0, normalized)
    if "codex" in normalized:
        return (5, 0, normalized)
    return (9, 0, normalized)


async def resolve_model(requested_model: str | None = None) -> str:
    if requested_model:
        return requested_model
    if config.upstream_model:
        return config.upstream_model
    models = await list_models()
    data = models.get("data", [])
    if not data:
        raise HTTPException(status_code=502, detail="Upstream returned no models")
    model_ids = [str(item.get("id", "")).strip() for item in data if isinstance(item, dict)]
    model_ids = [model_id for model_id in model_ids if model_id]
    if not model_ids:
        raise HTTPException(status_code=502, detail="Upstream model payload missing id")
    for model_id in sorted(model_ids, key=_model_rank):
        if _is_chat_friendly_model(model_id):
            return model_id
    return model_ids[0]


async def chat_completions(payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    forwarded = dict(payload)
    selected_model = await resolve_model(payload.get("model"))
    forwarded["model"] = selected_model
    if "temperature" in forwarded:
        forwarded["temperature"] = _normalize_temperature(
            selected_model,
            forwarded.get("temperature"),
        )
    if timeout is None:
        return await _request("POST", "/chat/completions", forwarded)
    return await _request("POST", "/chat/completions", forwarded, timeout=timeout)


def extract_text_from_message(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
        return "\n".join(parts)
    if isinstance(message, dict):
        content = message.get("content")
        if content is not None:
            return extract_text_from_message(content)
    return str(message or "")


async def create_chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = 0.3,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": messages}
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return await chat_completions(payload, timeout=timeout)


async def generate_text(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = 0.3,
    timeout: float | None = None,
    max_tokens: int | None = None,
) -> str:
    # Reasoning models starve the visible ``content`` when no budget is set (the small default
    # cap is spent on ``reasoning_content``). Apply a generous default so the answer is emitted.
    if max_tokens is None and config.upstream_max_output_tokens > 0:
        max_tokens = config.upstream_max_output_tokens

    async def _complete(budget: int | None) -> tuple[str, str, dict]:
        result = await create_chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=budget,
        )
        choice = (result.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = extract_text_from_message(message.get("content", ""))
        return text, str(choice.get("finish_reason") or ""), result

    text, finish_reason, _result = await _complete(max_tokens)
    # Long stage prompts (brief/synthesis) make GLM-class models spend the whole budget on
    # reasoning; one retry with a doubled cap is cheaper than failing the whole workflow.
    if not text.strip() and finish_reason == "length" and max_tokens:
        text, finish_reason, _result = await _complete(min(int(max_tokens) * 2, 65536))
    return text


async def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    timeout: float | None = None,
    batch_size: int | None = None,
) -> list[list[float]]:
    """Embed ``texts`` via an OpenAI-compatible ``/embeddings`` endpoint.

    Reads its own ``embedding_*`` config so the embedding provider can differ from
    the chat provider (e.g. Aliyun/Doubao). Returns one vector per input text; raises
    on transport/HTTP errors so the caller can decide whether to degrade gracefully.

    Inputs are split into chunks of at most ``batch_size`` items (default from
    ``config.embedding_batch_size``) because some providers (e.g. Doubao) cap the
    number of inputs per request; vectors are re-assembled in the original order.
    """

    payload = [text if isinstance(text, str) else str(text) for text in texts]
    if not payload:
        return []
    base_url = (config.embedding_base_url or config.upstream_base_url).rstrip("/")
    api_key = config.embedding_api_key or config.upstream_api_key
    selected_model = model or config.embedding_model
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing embedding API key")
    if not selected_model:
        raise HTTPException(status_code=500, detail="Missing EMBEDDING_MODEL")
    chunk = max(1, batch_size or config.embedding_batch_size)
    vectors: list[list[float]] = []
    for start in range(0, len(payload), chunk):
        window = payload[start : start + chunk]
        result = await _request(
            "POST",
            "/embeddings",
            {"model": selected_model, "input": window},
            timeout=timeout,
            base_url=base_url,
            api_key=api_key,
        )
        data = result.get("data")
        if not isinstance(data, list):
            raise HTTPException(status_code=502, detail="Embedding payload missing data")
        ordered = sorted(
            (item for item in data if isinstance(item, dict)),
            key=lambda item: int(item.get("index", 0) or 0),
        )
        vectors.extend(
            [float(value) for value in (item.get("embedding") or [])] for item in ordered
        )
    return vectors


async def generate_image(
    *,
    prompt: str,
    model: str | None = None,
    size: str = "1536x1024",
    quality: str = "low",
    output_format: str = "png",
    input_image: bytes | None = None,
    input_image_name: str = "source.png",
    input_image_mime_type: str = "image/png",
) -> GeneratedImage:
    selected_model = model or config.image_model or "gpt-image-2"
    endpoint = "/images/edits" if input_image else "/images/generations"
    url = f"{config.upstream_base_url.rstrip('/')}{endpoint}"
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
    }
    async with httpx.AsyncClient(timeout=config.image_request_timeout_seconds) as client:
        if input_image:
            response = await client.post(
                url,
                headers=_auth_headers(),
                data=payload,
                files={"image": (input_image_name, input_image, input_image_mime_type)},
            )
        else:
            response = await client.post(url, headers=_headers(), json=payload)
    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.json()
    data = (result.get("data") or [{}])[0]
    b64_json = data.get("b64_json")
    if b64_json:
        return GeneratedImage(
            image_bytes=base64.b64decode(b64_json),
            mime_type=f"image/{output_format}",
            model=selected_model,
            revised_prompt=str(data.get("revised_prompt", "")),
            size=size,
            quality=quality,
            output_format=output_format,
        )
    image_url = data.get("url")
    if image_url:
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            image_response = await client.get(str(image_url))
        if image_response.status_code >= 400:
            raise HTTPException(status_code=image_response.status_code, detail=image_response.text)
        return GeneratedImage(
            image_bytes=image_response.content,
            mime_type=image_response.headers.get("content-type", f"image/{output_format}"),
            model=selected_model,
            revised_prompt=str(data.get("revised_prompt", "")),
            size=size,
            quality=quality,
            output_format=output_format,
        )
    raise HTTPException(status_code=502, detail="Upstream image payload missing b64_json/url")
