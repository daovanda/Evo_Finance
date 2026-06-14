"""
Evo_Finance — Formula Evaluator
────────────────────────────────
Converts a Gene formula string into a pandas Series given a raw OHLCV DataFrame.

Supported grammar
  Terminals  : <base>_<window>   e.g. close_10, volume_5, open_1
  Unary ops  : rank(expr)  zscore(expr)  abs(expr)  signed_log(expr)
  Binary ops : (expr + expr)  (expr - expr)  (expr * expr)  (expr / expr)
  Window tag : (expr)_w<N>    created by Gene.change_window for complex exprs

The evaluator is recursive-descent — it handles arbitrary nesting depth.

All operations are applied **within each trading day cross-section** where
relevant (rank, zscore) so the signal is point-in-time safe.

IMPORTANT: df must only contain data up to and including close(t).
           Label computation (future returns) is the caller's responsibility.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Dict


# ─── Rolling helpers (per-stock time-series) ──────────────────────────────────

def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def _rank_cs(series: pd.Series) -> pd.Series:
    """Cross-sectional rank on each date (group is handled outside)."""
    return series.rank(pct=True)


def _zscore_cs(series: pd.Series) -> pd.Series:
    mu  = series.mean()
    std = series.std()
    return (series - mu) / (std + 1e-9)


def _signed_log(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


# ─── Evaluator ────────────────────────────────────────────────────────────────

# Regex for terminal:  base_window
_TERMINAL_RE = re.compile(r'^([a-z_]+)_(\d+)$')
# Regex for window-tagged complex:  (expr)_wN
_WTAG_RE = re.compile(r'^\((.+)\)_w(\d+)$')


def evaluate(formula: str, df: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    formula : str
        A Gene formula string.
    df : pd.DataFrame
        Multi-index (date, ticker) or flat DataFrame with columns
        [open, high, close, low, volume].  Values must be ≤ close(t).

    Returns
    -------
    pd.Series with the same index as df.
    """
    formula = formula.strip()

    # ── 1. Window-tagged complex expression: (expr)_wN ───────────────────────
    m = _WTAG_RE.match(formula)
    if m:
        inner_val = evaluate(m.group(1), df)
        w = int(m.group(2))
        # apply rolling mean over time axis per ticker
        if isinstance(df.index, pd.MultiIndex):
            return inner_val.groupby(level="ticker").transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
        return inner_val.rolling(w, min_periods=1).mean()

    # ── 2. Terminal: base_window ───────────────────────────────────────────────
    m = _TERMINAL_RE.match(formula)
    if m:
        base, w = m.group(1), int(m.group(2))
        col = _resolve_column(base, df)
        if isinstance(df.index, pd.MultiIndex):
            return col.groupby(level="ticker").transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
        return col.rolling(w, min_periods=1).mean()

    # ── 3. Unary ops ──────────────────────────────────────────────────────────
    for op in ("rank", "zscore", "abs", "signed_log"):
        if formula.startswith(f"{op}(") and formula.endswith(")"):
            inner = formula[len(op)+1:-1]
            inner_val = evaluate(inner, df)
            return _apply_unary(op, inner_val, df)

    # ── 4. Binary ops: (expr OP expr) ─────────────────────────────────────────
    if formula.startswith("(") and formula.endswith(")"):
        inner = formula[1:-1]
        op, left_expr, right_expr = _split_binary(inner)
        if op is not None:
            lv = evaluate(left_expr, df)
            rv = evaluate(right_expr, df)
            return _apply_binary(op, lv, rv)

    raise ValueError(f"Cannot evaluate formula: {formula!r}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

_COL_ALIASES = {
    "open":   "open",
    "high":   "high",
    "close":  "close",
    "low":    "low",
    "volume": "volume",
    "o":      "open",
    "h":      "high",
    "c":      "close",
    "l":      "low",
    "v":      "volume",
}


def _resolve_column(base: str, df: pd.DataFrame) -> pd.Series:
    col = _COL_ALIASES.get(base.lower())
    if col is None or col not in df.columns:
        raise ValueError(f"Unknown base column: {base!r}. "
                         f"Available: {list(df.columns)}")
    return df[col]


def _apply_unary(op: str, series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if op == "rank":
        if isinstance(df.index, pd.MultiIndex):
            return series.groupby(level="date").rank(pct=True)
        return series.rank(pct=True)
    if op == "zscore":
        if isinstance(df.index, pd.MultiIndex):
            return series.groupby(level="date").transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9)
            )
        return (series - series.mean()) / (series.std() + 1e-9)
    if op == "abs":
        return series.abs()
    if op == "signed_log":
        return _signed_log(series)
    raise ValueError(f"Unknown unary op: {op}")


def _apply_binary(op: str, lv: pd.Series, rv: pd.Series) -> pd.Series:
    if op == "+": return lv + rv
    if op == "-": return lv - rv
    if op == "*": return lv * rv
    if op == "/": return lv / rv.replace(0, np.nan)
    raise ValueError(f"Unknown binary op: {op}")


def _split_binary(inner: str):
    """
    Split 'expr OP expr' respecting nested parentheses.
    Returns (op, left_str, right_str) or (None, None, None).
    """
    depth = 0
    for i, ch in enumerate(inner):
        if ch in "([": depth += 1
        elif ch in ")]": depth -= 1
        if depth == 0 and i + 1 < len(inner):
            for op in (" + ", " - ", " * ", " / "):
                if inner[i+1:].startswith(op):
                    left  = inner[:i+1].strip()
                    right = inner[i+1+len(op):].strip()
                    return op.strip(), left, right
    return None, None, None