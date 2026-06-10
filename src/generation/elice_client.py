"""ELICE (course-provided) LLM API client.

Thin wrapper around the professor's ELICE endpoint. The payload/response are
OpenAI chat-completions shaped, but auth uses BOTH `Authorization: Bearer` and
`x-api-key` headers, and the endpoint URL is custom.

Credentials are read from config/secrets.yaml (git-ignored) first, then from the
ELICE_API_URL / ELICE_API_KEY env vars as a fallback (matches the professor's
snippet). NEVER hard-code or commit the key/URL — the professor said keep it
confidential.

    from src.generation.elice_client import elice_chat
    print(elice_chat("Reply with exactly: OK"))
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = ROOT / "config" / "secrets.yaml"
DEFAULT_MODEL = "openai/gpt-5-mini"


def load_credentials() -> tuple[str, str, str]:
    """Return (api_url, api_key, model). secrets.yaml wins; env vars are fallback."""
    url = key = model = None
    if SECRETS_PATH.exists():
        data = yaml.safe_load(SECRETS_PATH.read_text(encoding="utf-8")) or {}
        gen = data.get("generator", {}) or {}
        url, key, model = gen.get("api_url"), gen.get("api_key"), gen.get("model")

    url = url or os.environ.get("ELICE_API_URL")
    key = key or os.environ.get("ELICE_API_KEY")
    model = model or os.environ.get("ELICE_MODEL") or DEFAULT_MODEL

    placeholder = url in (None, "", "<ELICE_API_URL>") or key in (None, "", "<ELICE_API_KEY>")
    if placeholder:
        raise RuntimeError(
            "ELICE credentials not set. Copy config/secrets.example.yaml to "
            "config/secrets.yaml and fill api_url + api_key (or set ELICE_API_URL / "
            "ELICE_API_KEY env vars)."
        )
    return url, key, model


def elice_chat(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    timeout: int = 180,
    max_retries: int = 4,
    **extra,
) -> str:
    """Single chat completion. `extra` (e.g. temperature=0.7, seed=42, max_tokens=...)
    is merged into the payload — pass only params the endpoint accepts.

    NOTE: GPT-5-mini may reject a non-default `temperature`; omit it if a call 400s.
    """
    url, key, default_model = load_credentials()
    headers = {
        "Authorization": f"Bearer {key}",
        "x-api-key": key,
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": model or default_model, "messages": messages, **extra}

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:  # network / 5xx / rate-limit → backoff and retry
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"ELICE request failed after {max_retries} attempts: {last_err}")
