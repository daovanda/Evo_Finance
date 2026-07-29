"""Telegram notifications for the crypto production bot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from crypto.prod import trade_config


DEFAULT_ENV_FILE = Path(".env")


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> None:
    path = _resolve_env_path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_env_path(path: str | Path) -> Path:
    path = Path(path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        candidates.append(Path(__file__).resolve().parents[2] / path)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return path


def telegram_enabled(
    enabled_env: str = trade_config.TELEGRAM_NOTIFY_ENV,
) -> bool:
    load_env_file()
    value = os.getenv(enabled_env, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def telegram_configured(
    token_env: str = trade_config.TELEGRAM_BOT_TOKEN_ENV,
    chat_id_env: str = trade_config.TELEGRAM_CHAT_ID_ENV,
) -> bool:
    load_env_file()
    return bool(
        os.getenv(token_env)
        and os.getenv(chat_id_env)
    )


def send_telegram_message(
    text: str,
    timeout: float = 15.0,
    *,
    token_env: str = trade_config.TELEGRAM_BOT_TOKEN_ENV,
    chat_id_env: str = trade_config.TELEGRAM_CHAT_ID_ENV,
    enabled_env: str = trade_config.TELEGRAM_NOTIFY_ENV,
) -> dict[str, Any]:
    load_env_file()
    if not telegram_enabled(enabled_env):
        return {"ok": False, "skipped": True, "reason": "disabled"}

    token = os.getenv(token_env)
    chat_id = os.getenv(chat_id_env)
    if not token or not chat_id:
        return {"ok": False, "skipped": True, "reason": "missing_credentials"}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Telegram request failed: {exc}") from exc
    return json.loads(raw) if raw else {"ok": True}
