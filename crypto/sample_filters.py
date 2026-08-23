"""Causal sample filters shared by all crypto label modes."""

from __future__ import annotations

import numpy as np
import pandas as pd


SAMPLE_FILTERS = frozenset({"none", "slope_accumulation"})


def canonical_sample_filter(value: str | None) -> str:
    selected = str(value or "none").strip().lower()
    if selected not in SAMPLE_FILTERS:
        raise ValueError(
            f"Unknown sample filter {selected!r}. Allowed: "
            f"{', '.join(sorted(SAMPLE_FILTERS))}."
        )
    return selected


def slope_accumulation_mask(
    close: pd.Series,
    *,
    window: int = 3,
    previous_windows: int = 3,
    ratio: float = 0.15,
) -> pd.Series:
    """Return a causal mask for low current slope versus recent slopes."""
    window = int(window)
    previous_windows = int(previous_windows)
    ratio = float(ratio)
    if window < 2:
        raise ValueError("window must be at least 2.")
    if previous_windows < 1:
        raise ValueError("previous_windows must be at least 1.")
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("ratio must be finite and non-negative.")

    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    logs = np.full(len(values), np.nan, dtype=float)
    valid_price = np.isfinite(values) & (values > 0.0)
    logs[valid_price] = np.log(values[valid_price])

    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    slopes = np.full(len(values), np.nan, dtype=float)
    if len(values) >= window:
        valid_windows = np.convolve(
            np.isfinite(logs).astype(np.int16),
            np.ones(window, dtype=np.int16),
            mode="valid",
        ) == window
        safe_logs = np.where(np.isfinite(logs), logs, 0.0)
        rolling_slopes = np.correlate(safe_logs, x, mode="valid") / denominator
        rolling_slopes[~valid_windows] = np.nan
        slopes[window - 1 :] = rolling_slopes

    current_abs = pd.Series(np.abs(slopes), index=close.index)
    previous_abs = pd.concat(
        [current_abs.shift(offset) for offset in range(1, previous_windows + 1)],
        axis=1,
    )
    all_previous_valid = previous_abs.notna().all(axis=1)
    max_previous_abs = previous_abs.max(axis=1)
    mask = all_previous_valid & current_abs.notna() & (
        current_abs <= ratio * max_previous_abs
    )
    return mask.astype(bool)

