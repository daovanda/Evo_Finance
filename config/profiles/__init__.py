"""Dataset profiles for the shared stock evolution pipeline."""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Sequence


PROFILE_ENV_VAR = "EVO_FINANCE_PROFILE"
DEFAULT_PROFILE = "vnindex"
PROFILE_ALIASES = {
    "vn": "vnindex",
    "vnindex": "vnindex",
    "sp500": "sp500_15m",
    "sp500_15m": "sp500_15m",
}


def normalize_profile_name(name: str | None) -> str:
    clean = str(name or DEFAULT_PROFILE).strip().lower().replace("-", "_")
    try:
        return PROFILE_ALIASES[clean]
    except KeyError as exc:
        allowed = ", ".join(sorted(PROFILE_ALIASES))
        raise ValueError(f"Unknown profile {name!r}. Available profiles: {allowed}.") from exc


def bootstrap_profile_from_argv(argv: Sequence[str]) -> str:
    """Select a profile before modules import config.settings constants."""
    requested: str | None = None
    for index, arg in enumerate(argv):
        if arg == "--profile":
            if index + 1 >= len(argv):
                raise ValueError("--profile requires a value.")
            requested = argv[index + 1]
            break
        if arg.startswith("--profile="):
            requested = arg.split("=", 1)[1]
            break

    selected = normalize_profile_name(
        requested or os.getenv(PROFILE_ENV_VAR) or DEFAULT_PROFILE
    )
    os.environ[PROFILE_ENV_VAR] = selected
    return selected


def active_profile_name() -> str:
    return normalize_profile_name(os.getenv(PROFILE_ENV_VAR) or DEFAULT_PROFILE)


def load_active_profile() -> ModuleType:
    return importlib.import_module(f"config.profiles.{active_profile_name()}")
