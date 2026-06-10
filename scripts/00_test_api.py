"""One-shot ELICE connectivity test.

Run AFTER filling config/secrets.yaml:

    python scripts/00_test_api.py

Prints the model's reply (should be 'OK') or a clear setup error. Makes exactly
one tiny API call — safe to run repeatedly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.generation.elice_client import elice_chat, load_credentials  # noqa: E402

try:
    url, _key, model = load_credentials()
    host = url.split("/")[2] if "//" in url else "?"  # host only — don't print the secret path/key
    print(f"creds loaded  | model={model}  | host={host}")
    reply = elice_chat("Reply with exactly: OK", model=model)
    print("reply:", repr(reply.strip()))
    print("\n✅ ELICE connection works.")
except Exception as e:
    print(f"❌ {type(e).__name__}: {e}")
    sys.exit(1)
