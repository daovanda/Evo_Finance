"""
Evo_Finance — Formula Evaluator (v2)
─────────────────────────────────────
Grammar
  Terminal    : <base>_<w>            close_20 → rolling_mean(close, 20)
  Rolling TS  : std(expr, w)          rolling std
                max(expr, w)          rolling max
                min(expr, w)          rolling min
                shift(expr, w)        lag w days
  CS Unary    : rank / zscore / abs / signed_log
  Binary      : (expr OP expr)   or   expr OP expr  (top-level)
  Window-tag  : (expr)_wN            rolling mean of complex expr

Financial features expressible (examples)
  momentum_20  : (close_1 - shift(close_1, 20)) / shift(close_1, 20)
  volatility   : std(close_1, 20)
  vol_ratio    : std(close_1, 5) / std(close_1, 20)
  drawdown     : (close_1 - max(close_1, 20)) / max(close_1, 20)
  atr_14       : std((high_1 - low_1), 14)
  bb_position  : (close_1 - shift(close_1, 20)) / std(close_1, 20)
  rsi_proxy    : rank(close_1 / shift(close_1, 14))
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

_COL_ALIASES = {
    "open":"open","high":"high","close":"close","low":"low","volume":"volume",
    "o":"open","h":"high","c":"close","l":"low","v":"volume",
}
_TS_OPS = ("std","max","min","shift")
_CS_OPS = ("rank","zscore","abs","signed_log")
_TERMINAL_RE = re.compile(r'^([a-z_]+)_(\d+)$')
_WTAG_RE     = re.compile(r'^\((.+)\)_w(\d+)$', re.DOTALL)


# ─── Public ───────────────────────────────────────────────────────────────────

def evaluate(formula: str, df: pd.DataFrame) -> pd.Series:
    formula = formula.strip()

    # 1. Window-tag  (expr)_wN  — must check before binary to avoid conflict
    m = _WTAG_RE.match(formula)
    if m:
        return _ts_apply(evaluate(m.group(1), df), "mean", int(m.group(2)), df)

    # 2. Terminal  base_N
    m = _TERMINAL_RE.match(formula)
    if m:
        col = _resolve(m.group(1), df)
        return _ts_apply(col, "mean", int(m.group(2)), df)

    # 3. Named fn call  name(...)
    fn, args = _parse_fn(formula)
    if fn in _TS_OPS:
        expr_s, w = _split_ts_args(args)
        if expr_s is not None:
            return _ts_apply(evaluate(expr_s, df), fn, w, df)
    if fn in _CS_OPS:
        return _cs_unary(fn, evaluate(args, df), df)

    # 4. Binary op — search at depth 0 across the FULL formula
    op, left, right = _split_binary(formula)
    if op is not None:
        return _binary(op, evaluate(left, df), evaluate(right, df))

    # 5. Parenthesised sub-expression  (expr)
    if formula.startswith("(") and formula.endswith(")") and _balanced(formula):
        return evaluate(formula[1:-1], df)

    raise ValueError(f"Cannot evaluate formula: {formula!r}")


# ─── Parsing ──────────────────────────────────────────────────────────────────

def _parse_fn(formula: str):
    """Return (fn_name, args_str) if formula is fn(...), else (None, None)."""
    for fn in _TS_OPS + _CS_OPS:
        prefix = fn + "("
        if not formula.startswith(prefix):
            continue
        # find matching close paren
        depth = 0
        for i, ch in enumerate(formula[len(fn):]):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    # i is relative to formula[len(fn):]
                    end = len(fn) + i
                    if end == len(formula) - 1:
                        return fn, formula[len(prefix): end]
                    break
    return None, None


def _split_ts_args(args: str):
    """Split 'expr , window' — last top-level comma is the separator."""
    depth, last_comma = 0, -1
    for i, ch in enumerate(args):
        if ch == "(":   depth += 1
        elif ch == ")": depth -= 1
        elif ch == "," and depth == 0:
            last_comma = i
    if last_comma == -1:
        return None, None
    w_str = args[last_comma + 1:].strip()
    if not w_str.isdigit():
        return None, None
    return args[:last_comma].strip(), int(w_str)


def _split_binary(formula: str):
    """
    Find the lowest-precedence binary op at depth 0 in *formula*.
    Precedence (lowest first): + - then * /
    Returns (op, left, right) or (None, None, None).
    """
    for op_tok in (" + ", " - ", " * ", " / "):
        result = _find_op(formula, op_tok)
        if result is not None:
            left, right = result
            return op_tok.strip(), left, right
    return None, None, None


def _find_op(formula: str, op_tok: str):
    """Return (left, right) for the LAST occurrence of op_tok at depth 0."""
    depth = 0
    last = -1
    for i, ch in enumerate(formula):
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        if depth == 0 and formula[i:].startswith(op_tok):
            last = i
    if last == -1:
        return None
    return formula[:last].strip(), formula[last + len(op_tok):].strip()


def _balanced(formula: str) -> bool:
    """True if the outer () wrap the ENTIRE expression."""
    if not (formula.startswith("(") and formula.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(formula):
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(formula) - 1:
                return False   # closes before end → not fully wrapped
    return depth == 0


# ─── Operations ───────────────────────────────────────────────────────────────

def _ts_apply(series: pd.Series, fn: str, w: int, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="ticker").transform(lambda s: _roll(s, fn, w))
    return _roll(series, fn, w)


def _roll(s: pd.Series, fn: str, w: int) -> pd.Series:
    if fn == "mean":  return s.rolling(w, min_periods=1).mean()
    if fn == "std":   return s.rolling(w, min_periods=2).std()
    if fn == "max":   return s.rolling(w, min_periods=1).max()
    if fn == "min":   return s.rolling(w, min_periods=1).min()
    if fn == "shift": return s.shift(w)
    raise ValueError(fn)


def _cs_unary(op: str, series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if op == "rank":
        if isinstance(df.index, pd.MultiIndex):
            return series.groupby(level="date").rank(pct=True)
        return series.rank(pct=True)
    if op == "zscore":
        if isinstance(df.index, pd.MultiIndex):
            return series.groupby(level="date").transform(
                lambda s: (s - s.mean()) / (s.std() + 1e-9))
        return (series - series.mean()) / (series.std() + 1e-9)
    if op == "abs":         return series.abs()
    if op == "signed_log":  return np.sign(series) * np.log1p(np.abs(series))
    raise ValueError(op)


def _binary(op: str, lv: pd.Series, rv: pd.Series) -> pd.Series:
    if op == "+": return lv + rv
    if op == "-": return lv - rv
    if op == "*": return lv * rv
    if op == "/": return lv / rv.replace(0, np.nan)
    raise ValueError(op)


def _resolve(base: str, df: pd.DataFrame) -> pd.Series:
    col = _COL_ALIASES.get(base.lower())
    if col is None or col not in df.columns:
        raise ValueError(f"Unknown base: {base!r}")
    return df[col]