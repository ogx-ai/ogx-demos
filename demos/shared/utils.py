import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from ogx_client import OgxClient
from termcolor import colored

_ALLOWED_SCHEMES = {"http", "https"}


def _list_models(client: OgxClient) -> list:
    resp = client.models.list()
    return resp.data if hasattr(resp, "data") else list(resp)


def _get_model_type(model) -> str | None:
    for attr in ("model_type", "type", "model_kind", "kind", "model_family"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
    for metadata_attr in ("custom_metadata", "metadata"):
        metadata = getattr(model, metadata_attr, None)
        if isinstance(metadata, dict):
            value = metadata.get("model_type") or metadata.get("type")
            if isinstance(value, str):
                return value
    return None


def _is_llm_model(model) -> bool:
    model_type = _get_model_type(model)
    # If the client schema doesn't expose type fields, assume LLM.
    return model_type is None or model_type == "llm"


def _get_model_id(model) -> str | None:
    for attr in ("identifier", "model_id", "id", "name"):
        value = getattr(model, attr, None)
        if isinstance(value, str):
            return value
    return None


def resolve_model(client: OgxClient, model_id: str | None, env_var: str = "OGX_CHAT_MODEL") -> str | None:
    """Pick a chat model: explicit id > env var > OGX_CHAT_MODEL > auto-select.

    Auto-select filters by model_type == 'llm' and probes chat capability.
    """
    resolved = model_id or os.getenv(env_var) or os.getenv("OGX_CHAT_MODEL") or None
    if resolved:
        return resolved

    candidates = [
        _get_model_id(m)
        for m in _list_models(client)
        if _is_llm_model(m) and _get_model_id(m)
    ]

    for mid in candidates:
        if can_model_chat(client, mid):
            return mid

    print(colored("No available chat-capable models found.", "red"))
    return None


def resolve_openai_model(client, model_id: str | None) -> str | None:
    """Pick a model for OpenAI-compatible demos.

    Works with any OpenAI-compatible client that exposes ``client.models.list()``.
    """
    resolved = model_id or os.getenv("OGX_CHAT_MODEL") or os.getenv("OGX_MODEL") or None
    if resolved:
        return resolved
    models = _list_models(client)
    for m in models:
        candidate = _get_model_id(m)
        if candidate and _is_llm_model(m):
            return candidate
    print(colored("No available chat models found.", "red"))
    return None


def check_model_is_available(client: OgxClient, model: str) -> bool:
    all_ids = [_get_model_id(m) for m in _list_models(client)]
    all_ids = [mid for mid in all_ids if mid]

    if model not in all_ids:
        print(
            colored(
                f"Model `{model}` not found. Available models:\n\n{all_ids}\n",
                "red",
            )
        )
        return False

    return True


def can_model_chat(client: OgxClient, model_id: str) -> bool:
    try:
        client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception:
        return False
    return True


def get_any_available_model(client: OgxClient):
    resolved = os.getenv("OGX_CHAT_MODEL") or None
    if resolved:
        return resolved
    candidates = [
        _get_model_id(m)
        for m in _list_models(client)
        if _is_llm_model(m) and _get_model_id(m)
    ]
    if not candidates:
        print(colored("No available models.", "red"))
        return None
    return candidates[0]


def get_any_available_chat_model(client: OgxClient):
    resolved = os.getenv("OGX_CHAT_MODEL") or None
    if resolved:
        return resolved

    candidates = [
        _get_model_id(m)
        for m in _list_models(client)
        if _is_llm_model(m) and _get_model_id(m)
    ]
    if not candidates:
        print(colored("No available models.", "red"))
        return None

    for mid in candidates:
        if can_model_chat(client, mid):
            return mid

    print(colored("No available chat-capable models.", "red"))
    return None


def resolve_embedding_model(client: OgxClient, model_id: str | None = None) -> str | None:
    """Pick an embedding model: explicit id > OGX_EMBEDDING_MODEL > auto-select."""
    resolved = model_id or os.getenv("OGX_EMBEDDING_MODEL") or None
    if resolved:
        return resolved
    return _get_any_embedding_model(client)


def _get_any_embedding_model(client: OgxClient) -> str | None:
    embedding_models = [
        _get_model_id(m)
        for m in _list_models(client)
        if _get_model_id(m) and _get_model_type(m) == "embedding"
    ]
    if not embedding_models:
        print(colored("No available embedding models.", "red"))
        return None
    return embedding_models[0]


# Keep old name as alias
def get_embedding_dimension(client: OgxClient, model_id: str) -> int | None:
    for m in _list_models(client):
        if _get_model_id(m) == model_id:
            meta = getattr(m, "custom_metadata", None) or getattr(m, "metadata", None) or {}
            if isinstance(meta, dict) and isinstance(meta.get("embedding_dimension"), int):
                return meta["embedding_dimension"]
            dim = getattr(m, "embedding_dimension", None)
            if isinstance(dim, int):
                return dim
            break
    try:
        response = client.embeddings.create(model=model_id, input="dimension probe")
    except Exception:
        return None
    if not response.data:
        return None
    embedding = response.data[0].embedding
    if isinstance(embedding, list):
        return len(embedding)
    return None


def download_documents(urls: list[str], target_dir: Path) -> list[Path]:
    local_paths: list[Path] = []
    for url in urls:
        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            print(colored(f"Skipping URL with disallowed scheme: {url}", "red"))
            continue
        filename = url.rsplit("/", 1)[-1]
        target_path = target_dir / filename
        try:
            with urlopen(url, timeout=30) as response:
                content_bytes = response.read()
        except Exception as exc:
            print(colored(f"Failed to download {url}: {exc}", "red"))
            continue
        content_text = content_bytes.decode("utf-8", errors="ignore").strip()
        if not content_text:
            print(colored(f"Downloaded empty content from {url}", "red"))
            continue
        target_path = target_path.with_suffix(".txt")
        target_path.write_text(content_text, encoding="utf-8")
        local_paths.append(target_path)
    return local_paths


def build_context(search_results) -> str:
    if not search_results:
        return ""
    context_lines = ["Context from uploaded documents:"]
    for result in search_results:
        snippet = " ".join(
            content.text.strip() for content in result.content if getattr(content, "text", None)
        ).strip()
        if not snippet:
            continue
        score = result.score if result.score is not None else 0.0
        fname = getattr(result, "filename", "unknown") or "unknown"
        context_lines.append(f"- {fname} (score={score:.2f}): {snippet}")
    return "\n".join(context_lines)


def _to_dict(value) -> dict | None:
    """Convert an object to a dictionary if possible."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return None


def build_context_from_dicts(results: list) -> str:
    """Build context from dictionary results (e.g., from API responses parsed to dicts)."""
    if not results:
        return ""
    context_lines = ["Context from uploaded documents:"]
    for result_dict in results:
        if not isinstance(result_dict, dict):
            result_dict = _to_dict(result_dict) or {}
        content_list = result_dict.get("content") or []
        snippet_parts = []
        for chunk in content_list:
            chunk_dict = _to_dict(chunk) or {}
            text = chunk_dict.get("text")
            if text:
                snippet_parts.append(text.strip())
        snippet = " ".join(snippet_parts).strip()
        if not snippet:
            continue
        score = result_dict.get("score", 0.0)
        fname = result_dict.get("filename", "unknown") or "unknown"
        context_lines.append(f"- {fname} (score={score:.2f}): {snippet}")
    return "\n".join(context_lines)
