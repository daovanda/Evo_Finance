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

from config import settings

_COL_ALIASES = {
    "open":"open","high":"high","close":"close","low":"low","volume":"volume",
    "o":"open","h":"high","c":"close","l":"low","v":"volume",
    "market_open":"market_open","market_high":"market_high",
    "market_close":"market_close","market_low":"market_low",
    "market_volume":"market_volume",
    "m_open":"market_open","m_high":"market_high","m_close":"market_close",
    "m_low":"market_low","m_volume":"market_volume",
    "vnindex_open":"market_open","vnindex_high":"market_high",
    "vnindex_close":"market_close","vnindex_low":"market_low",
    "vnindex_volume":"market_volume",
}
_TS_OPS = (
    "std", "max", "min", "shift", "sum", "ema",
    "median", "q25", "q75", "iqr", "skew", "kurt",
    "ts_rank", "ts_zscore", "decay_linear", "slope",
    "days_since_rolling_high", "days_since_rolling_low",
)
_PAIR_TS_OPS = ("ts_corr", "ts_cov", "ts_beta")
_SECTOR_CS_OPS = ("sector_rank", "sector_zscore", "sector_neutralize")
_CS_OPS = (
    "rank", "zscore", "abs", "signed_log",
    "sign", "clip", "pos_part", "neg_part",
    "winsorize", "neutralize",
) + _SECTOR_CS_OPS
_FIN_TS_OPS = (
    "ret", "logret", "delta", "vol", "drawdown", "breakout", "ma_ratio",
    "bb_pos", "bb_width", "rsi", "vol_scale", "efficiency_ratio", "ulcer_index",
)
_FIN_WINDOW_OPS = (
    "pos", "volume_ratio", "liquidity", "atr", "stoch", "obv", "mfi", "cmf",
    "adx", "cci", "willr", "keltner_pos", "keltner_width", "donchian_width",
    "vwap", "vwap_pos", "amihud", "parkinson_vol", "gk_vol", "rs_vol",
    "aroon_up", "aroon_down", "aroon_osc", "choppiness",
)
_MARKET_WINDOW_OPS = (
    "market_ret", "market_vol", "market_drawdown", "market_ma_ratio",
    "market_rsi", "market_pos", "market_volume_ratio",
    "rel_ret", "rel_strength", "market_corr", "market_beta", "market_alpha",
    "idiosyncratic_vol", "up_capture", "down_capture",
)
_BREADTH_NOARG_OPS = (
    "advance_count", "decline_count", "unchanged_count",
    "advance_ratio", "decline_ratio", "advance_decline_ratio",
    "advance_decline_spread", "advance_decline_net_pct", "cs_dispersion",
)
_BREADTH_WINDOW_OPS = ("pct_above_ma", "breadth_momentum")
_SECTOR_NOARG_OPS = (
    "sector_code", "sector_size",
    "sector_advance_count", "sector_decline_count", "sector_unchanged_count",
    "sector_advance_ratio", "sector_decline_ratio",
    "sector_advance_decline_ratio", "sector_advance_decline_spread",
    "sector_advance_decline_net_pct", "sector_dispersion",
)
_SECTOR_WINDOW_OPS = (
    "sector_ret", "sector_vol", "sector_drawdown", "sector_ma_ratio",
    "sector_rsi", "sector_pos", "sector_volume_ratio",
    "rel_sector_ret", "sector_rel_strength", "sector_corr", "sector_beta",
    "sector_alpha", "sector_idiosyncratic_vol",
    "sector_up_capture", "sector_down_capture",
    "sector_pct_above_ma", "sector_breadth_momentum",
)
_FIN_TWO_WINDOW_OPS = ("stoch_d",)
_FIN_NOARG_OPS = (
    "body", "range", "gap", "upper_wick", "lower_wick", "dollar_volume",
    "signed_volume", "typical_price", "money_flow",
)
_CUM_TS_OPS = (
    "cummax", "cummin", "cumret", "expanding_drawdown", "expanding_runup",
    "days_since_high", "days_since_low", "cum_sum", "up_streak", "down_streak",
)
_CUM_NOARG_OPS = ("cum_obv", "cum_adl", "cum_pvt")
_COMPARE_OPS = ("gt", "lt", "cross_above", "cross_below")
_CONDITIONAL_OPS = ("where", "rule_signal")
_CONST_OPS = ("const",)
_TERMINAL_RE = re.compile(r'^([a-z_]+)_(\d+)$')
DIV_ZERO_EPS = 1e-12


# ─── Public ───────────────────────────────────────────────────────────────────

def evaluate(formula: str, df: pd.DataFrame) -> pd.Series:
    formula = formula.strip()

    # 1. Window-tag  (expr)_wN  — must check before binary to avoid conflict
    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, window = parsed_tag
        return _ts_apply(evaluate(inner, df), "mean", window, df)

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
    if fn in _PAIR_TS_OPS:
        parsed = _parse_pair_ts_args(args)
        if parsed is not None:
            left_s, right_s, w = parsed
            return _ts_pair_apply(evaluate(left_s, df), evaluate(right_s, df), fn, w, df)
    if fn in _CS_OPS:
        return _cs_unary(fn, evaluate(args, df), df)
    if fn in _FIN_TS_OPS:
        expr_s, w = _split_ts_args(args)
        if expr_s is not None:
            return _finance_ts(fn, evaluate(expr_s, df), w, df)
    if fn in _FIN_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _finance_window(fn, w, df)
    if fn in _MARKET_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _market_window(fn, w, df)
    if fn in _BREADTH_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _breadth_window(fn, w, df)
    if fn in _BREADTH_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _breadth_noarg(fn, df)
    if fn in _SECTOR_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _sector_window(fn, w, df)
    if fn in _SECTOR_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _sector_noarg(fn, df)
    if fn in _FIN_TWO_WINDOW_OPS:
        windows = _parse_two_window_args(args)
        if windows is not None:
            return _finance_two_window(fn, windows[0], windows[1], df)
    if fn in _FIN_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _finance_noarg(fn, df)
    if fn in _CUM_TS_OPS:
        return _cumulative_ts(fn, evaluate(args, df), df)
    if fn in _CUM_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _cumulative_noarg(fn, df)
    if fn in _COMPARE_OPS:
        parsed = _parse_two_expr_args(args)
        if parsed is not None:
            return _compare(fn, evaluate(parsed[0], df), evaluate(parsed[1], df), df)
    if fn in _CONDITIONAL_OPS:
        parsed = _parse_three_expr_args(args)
        if parsed is not None:
            return _conditional(fn, parsed, df)
    if fn in _CONST_OPS:
        value = _parse_float_arg(args)
        if value is not None:
            return pd.Series(value, index=df.index)

    # 4. Binary op — search at depth 0 across the FULL formula
    op, left, right = _split_binary(formula)
    if op is not None:
        return _binary(op, evaluate(left, df), evaluate(right, df))

    # 5. Parenthesised sub-expression  (expr)
    if formula.startswith("(") and formula.endswith(")") and _balanced(formula):
        return evaluate(formula[1:-1], df)

    raise ValueError(f"Cannot evaluate formula: {formula!r}")


def has_division_by_zero(formula: str, df: pd.DataFrame) -> bool:
    """
    Return True if any division denominator in formula is zero on df.

    This is used by the mutator/domain admission checks. NaNs from lookback
    windows are allowed; exact/near-zero denominators are not.
    """
    return _division_zero_count(formula.strip(), df) > 0


# ─── Parsing ──────────────────────────────────────────────────────────────────

def _parse_fn(formula: str):
    """Return (fn_name, args_str) if formula is fn(...), else (None, None)."""
    for fn in (
        _TS_OPS + _PAIR_TS_OPS + _CS_OPS + _FIN_TS_OPS + _FIN_WINDOW_OPS
        + _MARKET_WINDOW_OPS + _BREADTH_WINDOW_OPS + _SECTOR_WINDOW_OPS
        + _FIN_TWO_WINDOW_OPS + _FIN_NOARG_OPS + _BREADTH_NOARG_OPS
        + _SECTOR_NOARG_OPS + _CUM_TS_OPS
        + _CUM_NOARG_OPS + _COMPARE_OPS + _CONDITIONAL_OPS + _CONST_OPS
    ):
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


def _parse_window_arg(args: str):
    args = args.strip()
    return int(args) if args.isdigit() else None


def _parse_two_window_args(args: str):
    parts = _split_top_level(args, ",")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def _parse_two_expr_args(args: str):
    parts = _split_top_level(args, ",")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _parse_pair_ts_args(args: str):
    parts = _split_top_level(args, ",")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return parts[0], parts[1], int(parts[2])


def _parse_three_expr_args(args: str):
    parts = _split_top_level(args, ",")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _parse_float_arg(args: str):
    try:
        return float(args.strip())
    except (TypeError, ValueError):
        return None


def _split_top_level(text: str, sep: str) -> list[str]:
    depth = 0
    parts = []
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def _parse_window_tag(formula: str):
    """Return (inner_expr, window) only for a true outer '(expr)_wN' tag."""
    suffix = re.search(r'_w(\d+)$', formula)
    if suffix is None or not formula.startswith("("):
        return None

    close_idx = suffix.start() - 1
    if close_idx <= 0 or formula[close_idx] != ")":
        return None

    depth = 0
    for i, ch in enumerate(formula[:close_idx + 1]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if i == close_idx:
                    return formula[1:close_idx], int(suffix.group(1))
                return None

    return None


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


def _division_zero_count(formula: str, df: pd.DataFrame) -> int:
    formula = formula.strip()

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _division_zero_count(inner, df)

    if _TERMINAL_RE.match(formula):
        return 0

    fn, args = _parse_fn(formula)
    if fn in _TS_OPS:
        expr_s, _ = _split_ts_args(args)
        if expr_s is not None:
            return _division_zero_count(expr_s, df)
    if fn in _PAIR_TS_OPS:
        parsed = _parse_pair_ts_args(args)
        if parsed is not None:
            left_s, right_s, w = parsed
            count = _division_zero_count(left_s, df) + _division_zero_count(right_s, df)
            if fn == "ts_beta":
                right = evaluate(right_s, df)
                denom = _ts_apply(right, "std", w, df) ** 2
                count += int(_zero_denominator_mask(denom).sum())
            return count
    if fn in _CS_OPS:
        return _division_zero_count(args, df)
    if fn in _FIN_TS_OPS:
        expr_s, w = _split_ts_args(args)
        if expr_s is not None:
            count = _division_zero_count(expr_s, df)
            return count + _finance_ts_zero_count(fn, evaluate(expr_s, df), w, df)
    if fn in _FIN_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _finance_window_zero_count(fn, w, df)
    if fn in _MARKET_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _market_window_zero_count(fn, w, df)
    if fn in _BREADTH_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _breadth_window_zero_count(fn, w, df)
    if fn in _SECTOR_WINDOW_OPS:
        w = _parse_window_arg(args)
        if w is not None:
            return _sector_window_zero_count(fn, w, df)
    if fn in _FIN_TWO_WINDOW_OPS:
        windows = _parse_two_window_args(args)
        if windows is not None:
            return _finance_two_window_zero_count(fn, windows[0], windows[1], df)
    if fn in _FIN_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _finance_noarg_zero_count(fn, df)
    if fn in _BREADTH_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _breadth_noarg_zero_count(fn, df)
    if fn in _SECTOR_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _sector_noarg_zero_count(fn, df)
    if fn in _CUM_TS_OPS:
        count = _division_zero_count(args, df)
        return count + _cumulative_ts_zero_count(fn, evaluate(args, df), df)
    if fn in _CUM_NOARG_OPS:
        if args is not None and args.strip() == "":
            return _cumulative_noarg_zero_count(fn, df)
    if fn in _COMPARE_OPS:
        parsed = _parse_two_expr_args(args)
        if parsed is not None:
            return _division_zero_count(parsed[0], df) + _division_zero_count(parsed[1], df)
    if fn in _CONDITIONAL_OPS:
        parsed = _parse_three_expr_args(args)
        if parsed is not None:
            return sum(_division_zero_count(part, df) for part in parsed)
    if fn in _CONST_OPS:
        if _parse_float_arg(args) is not None:
            return 0

    op, left, right = _split_binary(formula)
    if op is not None:
        count = _division_zero_count(left, df) + _division_zero_count(right, df)
        if op == "/":
            denom = evaluate(right, df)
            zero_mask = _zero_denominator_mask(denom)
            count += int(zero_mask.sum())
        return count

    if formula.startswith("(") and formula.endswith(")") and _balanced(formula):
        return _division_zero_count(formula[1:-1], df)

    raise ValueError(f"Cannot inspect formula for division by zero: {formula!r}")


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
    if fn == "sum":   return s.rolling(w, min_periods=1).sum()
    if fn == "ema":   return s.ewm(span=w, adjust=False, min_periods=1).mean()
    if fn == "median": return s.rolling(w, min_periods=1).median()
    if fn == "q25":    return s.rolling(w, min_periods=1).quantile(0.25)
    if fn == "q75":    return s.rolling(w, min_periods=1).quantile(0.75)
    if fn == "iqr":
        roll = s.rolling(w, min_periods=1)
        return roll.quantile(0.75) - roll.quantile(0.25)
    if fn == "skew":
        return s.rolling(w, min_periods=min(3, w)).apply(_skew_value, raw=True)
    if fn == "kurt":
        return s.rolling(w, min_periods=min(3, w)).apply(_kurt_value, raw=True)
    if fn == "ts_rank":
        return s.rolling(w, min_periods=1).apply(_last_rank_pct, raw=True)
    if fn == "ts_zscore":
        mean = s.rolling(w, min_periods=1).mean()
        std = s.rolling(w, min_periods=2).std()
        return _safe_div(s - mean, std)
    if fn == "decay_linear":
        return s.rolling(w, min_periods=1).apply(_decay_linear_value, raw=True)
    if fn == "slope":
        return s.rolling(w, min_periods=2).apply(_slope_value, raw=True)
    if fn == "days_since_rolling_high":
        return s.rolling(w, min_periods=1).apply(
            lambda values: _rolling_extreme_age_value(values, "high"),
            raw=True,
        )
    if fn == "days_since_rolling_low":
        return s.rolling(w, min_periods=1).apply(
            lambda values: _rolling_extreme_age_value(values, "low"),
            raw=True,
        )
    raise ValueError(fn)


def _last_rank_pct(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0 or np.isnan(values[-1]):
        return np.nan
    return float(pd.Series(valid).rank(pct=True).iloc[-1])


def _decay_linear_value(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    weights = np.arange(1, len(valid) + 1, dtype=float)
    return float(np.dot(valid, weights) / weights.sum())


def _slope_value(values: np.ndarray) -> float:
    valid_mask = ~np.isnan(values)
    y = values[valid_mask]
    if len(y) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[valid_mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if np.isclose(denom, 0.0):
        return np.nan
    return float(np.dot(x, y - y.mean()) / denom)


def _rolling_extreme_age_value(values: np.ndarray, mode: str) -> float:
    if len(values) == 0 or np.isnan(values[-1]):
        return np.nan

    valid_mask = ~np.isnan(values)
    if not valid_mask.any():
        return np.nan

    valid_positions = np.flatnonzero(valid_mask)
    valid_values = values[valid_mask]
    target = np.nanmax(valid_values) if mode == "high" else np.nanmin(valid_values)
    extreme_positions = valid_positions[np.isclose(valid_values, target, equal_nan=False)]
    if len(extreme_positions) == 0:
        return np.nan
    return float((len(values) - 1) - extreme_positions[-1])


def _skew_value(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 3:
        return np.nan
    centered = valid - valid.mean()
    std = float(np.sqrt(np.mean(centered ** 2)))
    if np.isclose(std, 0.0):
        return np.nan
    z = centered / std
    return float(np.mean(z ** 3))


def _kurt_value(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) < 3:
        return np.nan
    centered = valid - valid.mean()
    std = float(np.sqrt(np.mean(centered ** 2)))
    if np.isclose(std, 0.0):
        return np.nan
    z = centered / std
    return float(np.mean(z ** 4) - 3.0)


def _ts_rolling_apply(
    series: pd.Series,
    w: int,
    func,
    df: pd.DataFrame,
) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="ticker").transform(
            lambda s: s.rolling(w, min_periods=2).apply(func, raw=True)
        )
    return series.rolling(w, min_periods=2).apply(func, raw=True)


def _ts_pair_apply(
    left: pd.Series,
    right: pd.Series,
    fn: str,
    w: int,
    df: pd.DataFrame,
) -> pd.Series:
    pair = pd.concat([left, right], axis=1)
    pair.columns = ["left", "right"]
    result = pd.Series(np.nan, index=left.index, dtype=float)

    if isinstance(df.index, pd.MultiIndex):
        groups = pair.groupby(level="ticker", group_keys=False)
        for _, group in groups:
            result.loc[group.index] = _pair_roll(group["left"], group["right"], fn, w)
        return result

    return _pair_roll(pair["left"], pair["right"], fn, w)


def _pair_roll(left: pd.Series, right: pd.Series, fn: str, w: int) -> pd.Series:
    if fn == "ts_corr":
        return left.rolling(w, min_periods=3).corr(right)
    if fn == "ts_cov":
        return left.rolling(w, min_periods=3).cov(right)
    if fn == "ts_beta":
        cov = left.rolling(w, min_periods=3).cov(right)
        var = right.rolling(w, min_periods=3).var()
        return _safe_div(cov, var)
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
    if op == "sign":        return np.sign(series)
    if op == "clip":        return series.clip(-1.0, 1.0)
    if op == "pos_part":    return series.clip(lower=0.0)
    if op == "neg_part":    return (-series).clip(lower=0.0)
    if op == "winsorize":   return _cs_winsorize(series, df)
    if op == "neutralize":  return _cs_neutralize(series, df)
    if op == "sector_rank":
        return _sector_date_transform(series, lambda s: s.rank(pct=True), df)
    if op == "sector_zscore":
        return _sector_date_transform(
            series,
            lambda s: (s - s.mean()) / (s.std() + 1e-9),
            df,
        )
    if op == "sector_neutralize":
        return _sector_date_transform(series, lambda s: s - s.mean(), df)
    raise ValueError(op)


def _cs_winsorize(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    def _clip_group(s: pd.Series) -> pd.Series:
        lower = s.quantile(0.05)
        upper = s.quantile(0.95)
        return s.clip(lower, upper)

    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="date", group_keys=False).transform(_clip_group)
    return _clip_group(series)


def _cs_neutralize(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="date").transform(lambda s: s - s.mean())
    return series - series.mean()


def _finance_ts(fn: str, series: pd.Series, w: int, df: pd.DataFrame) -> pd.Series:
    shifted = _ts_apply(series, "shift", w, df)

    if fn == "ret":
        return _safe_div(series, shifted) - 1.0
    if fn == "logret":
        ratio = _safe_div(series, shifted)
        return np.log(ratio.where(ratio > 0))
    if fn == "delta":
        return series - shifted
    if fn == "vol":
        one_ret = _safe_div(series, _ts_apply(series, "shift", 1, df)) - 1.0
        return _ts_apply(one_ret, "std", w, df)
    if fn == "drawdown":
        rolling_max = _ts_apply(series, "max", w, df)
        return _safe_div(series, rolling_max) - 1.0
    if fn == "breakout":
        prior_max = _ts_apply(_ts_apply(series, "max", w, df), "shift", 1, df)
        return _safe_div(series, prior_max) - 1.0
    if fn == "ma_ratio":
        rolling_mean = _ts_apply(series, "mean", w, df)
        return _safe_div(series, rolling_mean) - 1.0
    if fn == "bb_pos":
        rolling_mean = _ts_apply(series, "mean", w, df)
        rolling_std = _ts_apply(series, "std", w, df)
        return _safe_div(series - rolling_mean, rolling_std)
    if fn == "bb_width":
        rolling_mean = _ts_apply(series, "mean", w, df)
        rolling_std = _ts_apply(series, "std", w, df)
        return _safe_div(rolling_std, rolling_mean)
    if fn == "rsi":
        delta = series - _ts_apply(series, "shift", 1, df)
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = _ts_apply(gain, "mean", w, df)
        avg_loss = _ts_apply(loss, "mean", w, df)
        rs = _safe_div(avg_gain, avg_loss)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.where(
            ~_zero_denominator_mask(avg_loss),
            np.where(avg_gain > 0, 100.0, 50.0),
        )
    if fn == "vol_scale":
        return _safe_div(series, _ts_apply(series, "std", w, df))
    if fn == "efficiency_ratio":
        net_move = (series - shifted).abs()
        one_step_move = (series - _ts_apply(series, "shift", 1, df)).abs()
        path_length = _ts_apply(one_step_move, "sum", w, df)
        return _safe_div(net_move, path_length)
    if fn == "ulcer_index":
        rolling_max = _ts_apply(series, "max", w, df)
        drawdown = _safe_div(series, rolling_max) - 1.0
        mean_sq = _ts_apply(drawdown ** 2, "mean", w, df)
        return np.sqrt(mean_sq)
    raise ValueError(fn)


def _finance_window(fn: str, w: int, df: pd.DataFrame) -> pd.Series:
    open_ = _resolve("open", df)
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    volume = _resolve("volume", df)

    if fn == "pos":
        rolling_high = _ts_apply(high, "max", w, df)
        rolling_low = _ts_apply(low, "min", w, df)
        return _safe_div(close - rolling_low, rolling_high - rolling_low)
    if fn == "volume_ratio":
        return _safe_div(volume, _ts_apply(volume, "mean", w, df)) - 1.0
    if fn == "liquidity":
        return _ts_apply(close * volume, "mean", w, df)
    if fn == "atr":
        prev_close = _ts_apply(close, "shift", 1, df)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return _safe_div(_ts_apply(true_range, "mean", w, df), close)
    if fn == "stoch":
        return _stoch(w, df)
    if fn == "obv":
        return _ts_apply(_finance_noarg("signed_volume", df), "sum", w, df)
    if fn == "mfi":
        typical = _typical_price(df)
        raw_flow = typical * volume
        delta = typical - _ts_apply(typical, "shift", 1, df)
        pos_flow = raw_flow.where(delta > 0, 0.0)
        neg_flow = raw_flow.where(delta < 0, 0.0)
        pos_sum = _ts_apply(pos_flow, "sum", w, df)
        neg_sum = _ts_apply(neg_flow, "sum", w, df)
        ratio = _safe_div(pos_sum, neg_sum)
        mfi = 100.0 - (100.0 / (1.0 + ratio))
        return mfi.where(
            ~_zero_denominator_mask(neg_sum),
            np.where(pos_sum > 0, 100.0, 50.0),
        )
    if fn == "cmf":
        multiplier = _money_flow_multiplier(df)
        mf_volume = multiplier * volume
        return _safe_div(_ts_apply(mf_volume, "sum", w, df),
                         _ts_apply(volume, "sum", w, df))
    if fn == "adx":
        return _adx(w, df)
    if fn == "cci":
        typical = _typical_price(df)
        ma = _ts_apply(typical, "mean", w, df)
        mad = _ts_rolling_apply(
            typical, w, lambda x: float(np.mean(np.abs(x - np.mean(x)))), df
        )
        return _safe_div(typical - ma, 0.015 * mad)
    if fn == "willr":
        high_max = _ts_apply(high, "max", w, df)
        low_min = _ts_apply(low, "min", w, df)
        return -100.0 * _safe_div(high_max - close, high_max - low_min)
    if fn == "keltner_pos":
        center = _ts_apply(close, "ema", w, df)
        atr_abs = _atr_abs(w, df)
        return _safe_div(close - center, 2.0 * atr_abs)
    if fn == "keltner_width":
        center = _ts_apply(close, "ema", w, df)
        atr_abs = _atr_abs(w, df)
        return _safe_div(4.0 * atr_abs, center)
    if fn == "donchian_width":
        high_max = _ts_apply(high, "max", w, df)
        low_min = _ts_apply(low, "min", w, df)
        return _safe_div(high_max - low_min, close)
    if fn == "vwap":
        typical = _typical_price(df)
        return _safe_div(
            _ts_apply(typical * volume, "sum", w, df),
            _ts_apply(volume, "sum", w, df),
        )
    if fn == "vwap_pos":
        vwap = _finance_window("vwap", w, df)
        return _safe_div(close, vwap) - 1.0
    if fn == "amihud":
        one_ret = (_safe_div(close, _ts_apply(close, "shift", 1, df)) - 1.0).abs()
        dollar_volume = close * volume
        return _ts_apply(_safe_div(one_ret, dollar_volume), "mean", w, df)
    if fn == "parkinson_vol":
        hl = np.log(_safe_div(high, low).where(_safe_div(high, low) > 0))
        variance = (hl ** 2) / (4.0 * np.log(2.0))
        return np.sqrt(_ts_apply(variance, "mean", w, df))
    if fn == "gk_vol":
        hl = np.log(_safe_div(high, low).where(_safe_div(high, low) > 0))
        co = np.log(_safe_div(close, open_).where(_safe_div(close, open_) > 0))
        variance = 0.5 * (hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (co ** 2)
        return np.sqrt(_ts_apply(variance.clip(lower=0.0), "mean", w, df))
    if fn == "rs_vol":
        ho = np.log(_safe_div(high, open_).where(_safe_div(high, open_) > 0))
        hc = np.log(_safe_div(high, close).where(_safe_div(high, close) > 0))
        lo = np.log(_safe_div(low, open_).where(_safe_div(low, open_) > 0))
        lc = np.log(_safe_div(low, close).where(_safe_div(low, close) > 0))
        variance = ho * hc + lo * lc
        return np.sqrt(_ts_apply(variance.clip(lower=0.0), "mean", w, df))
    if fn == "aroon_up":
        age = _ts_apply(high, "days_since_rolling_high", w, df)
        return 100.0 * (float(w) - age) / float(w)
    if fn == "aroon_down":
        age = _ts_apply(low, "days_since_rolling_low", w, df)
        return 100.0 * (float(w) - age) / float(w)
    if fn == "aroon_osc":
        return _finance_window("aroon_up", w, df) - _finance_window("aroon_down", w, df)
    if fn == "choppiness":
        tr_sum = _ts_apply(_true_range_abs(df), "sum", w, df)
        price_range = _ts_apply(high, "max", w, df) - _ts_apply(low, "min", w, df)
        ratio = _safe_div(tr_sum, price_range)
        return 100.0 * np.log10(ratio.where(ratio > 0)) / np.log10(float(w))
    raise ValueError(fn)


def _market_window(fn: str, w: int, df: pd.DataFrame) -> pd.Series:
    close = _resolve("close", df)
    market_close = _resolve("market_close", df)
    market_high = _resolve("market_high", df)
    market_low = _resolve("market_low", df)
    market_volume = _resolve("market_volume", df)

    stock_ret = _safe_div(close, _ts_apply(close, "shift", 1, df)) - 1.0
    market_ret = _safe_div(market_close, _ts_apply(market_close, "shift", 1, df)) - 1.0

    if fn == "market_ret":
        return _finance_ts("ret", market_close, w, df)
    if fn == "market_vol":
        return _finance_ts("vol", market_close, w, df)
    if fn == "market_drawdown":
        return _finance_ts("drawdown", market_close, w, df)
    if fn == "market_ma_ratio":
        return _finance_ts("ma_ratio", market_close, w, df)
    if fn == "market_rsi":
        return _finance_ts("rsi", market_close, w, df)
    if fn == "market_pos":
        rolling_high = _ts_apply(market_high, "max", w, df)
        rolling_low = _ts_apply(market_low, "min", w, df)
        return _safe_div(market_close - rolling_low, rolling_high - rolling_low)
    if fn == "market_volume_ratio":
        return _safe_div(market_volume, _ts_apply(market_volume, "mean", w, df)) - 1.0
    if fn == "rel_ret":
        return _finance_ts("ret", close, w, df) - _finance_ts("ret", market_close, w, df)
    if fn == "rel_strength":
        stock_ratio = _safe_div(close, _ts_apply(close, "shift", w, df))
        market_ratio = _safe_div(market_close, _ts_apply(market_close, "shift", w, df))
        return _safe_div(stock_ratio, market_ratio) - 1.0
    if fn == "market_corr":
        return _ts_pair_apply(stock_ret, market_ret, "ts_corr", w, df)
    if fn == "market_beta":
        return _ts_pair_apply(stock_ret, market_ret, "ts_beta", w, df)
    if fn == "market_alpha":
        beta = _market_window("market_beta", w, df)
        return stock_ret - beta * market_ret
    if fn == "idiosyncratic_vol":
        residual = _market_window("market_alpha", w, df)
        return _ts_apply(residual, "std", w, df)
    if fn == "up_capture":
        stock_up = stock_ret.where(market_ret > 0)
        market_up = market_ret.where(market_ret > 0)
        return _safe_div(_ts_apply(stock_up, "sum", w, df), _ts_apply(market_up, "sum", w, df))
    if fn == "down_capture":
        stock_down = stock_ret.where(market_ret < 0)
        market_down = market_ret.where(market_ret < 0)
        return _safe_div(
            _ts_apply(stock_down, "sum", w, df),
            _ts_apply(market_down, "sum", w, df),
        )
    raise ValueError(fn)


def _breadth_noarg(fn: str, df: pd.DataFrame) -> pd.Series:
    stock_ret = _stock_one_day_ret(df)
    advance = (stock_ret > 0) & stock_ret.notna()
    decline = (stock_ret < 0) & stock_ret.notna()
    unchanged = (stock_ret == 0) & stock_ret.notna()

    advance_count = _date_sum(advance.astype(float), df)
    decline_count = _date_sum(decline.astype(float), df)
    unchanged_count = _date_sum(unchanged.astype(float), df)
    total = _date_sum(stock_ret.notna().astype(float), df).replace(0.0, np.nan)

    if fn == "advance_count":
        return advance_count
    if fn == "decline_count":
        return decline_count
    if fn == "unchanged_count":
        return unchanged_count
    if fn == "advance_ratio":
        return _safe_div(advance_count, total)
    if fn == "decline_ratio":
        return _safe_div(decline_count, total)
    if fn == "advance_decline_ratio":
        return _safe_div(advance_count + 1.0, decline_count + 1.0)
    if fn == "advance_decline_spread":
        return advance_count - decline_count
    if fn == "advance_decline_net_pct":
        return _safe_div(advance_count - decline_count, total)
    if fn == "cs_dispersion":
        return _date_transform(stock_ret, lambda s: s.std(), df)
    raise ValueError(fn)


def _breadth_window(fn: str, w: int, df: pd.DataFrame) -> pd.Series:
    close = _resolve("close", df)

    if fn == "pct_above_ma":
        ma = _ts_apply(close, "mean", w, df)
        valid = close.notna() & ma.notna()
        above = ((close > ma) & valid).astype(float)
        above_count = _date_sum(above, df)
        total = _date_sum(valid.astype(float), df).replace(0.0, np.nan)
        return _safe_div(above_count, total)

    if fn == "breadth_momentum":
        net_pct = _breadth_noarg("advance_decline_net_pct", df)
        return _ts_apply(net_pct, "mean", w, df)

    raise ValueError(fn)


def _sector_noarg(fn: str, df: pd.DataFrame) -> pd.Series:
    sector = _sector_series(df)
    close = _resolve("close", df)
    stock_ret = _stock_one_day_ret(df)
    advance = (stock_ret > 0) & stock_ret.notna()
    decline = (stock_ret < 0) & stock_ret.notna()
    unchanged = (stock_ret == 0) & stock_ret.notna()

    advance_count = _sector_date_sum(advance.astype(float), df)
    decline_count = _sector_date_sum(decline.astype(float), df)
    unchanged_count = _sector_date_sum(unchanged.astype(float), df)
    total = _sector_date_sum(stock_ret.notna().astype(float), df).replace(0.0, np.nan)
    sector_size = _sector_date_sum(close.notna().astype(float), df)

    if fn == "sector_code":
        code_map = _sector_code_map()
        return sector.map(code_map).fillna(0.0).astype(float)
    if fn == "sector_size":
        return sector_size
    if fn == "sector_advance_count":
        return advance_count
    if fn == "sector_decline_count":
        return decline_count
    if fn == "sector_unchanged_count":
        return unchanged_count
    if fn == "sector_advance_ratio":
        return _safe_div(advance_count, total)
    if fn == "sector_decline_ratio":
        return _safe_div(decline_count, total)
    if fn == "sector_advance_decline_ratio":
        return _safe_div(advance_count + 1.0, decline_count + 1.0)
    if fn == "sector_advance_decline_spread":
        return advance_count - decline_count
    if fn == "sector_advance_decline_net_pct":
        return _safe_div(advance_count - decline_count, total)
    if fn == "sector_dispersion":
        return _sector_date_transform(stock_ret, lambda s: s.std(), df)
    raise ValueError(fn)


def _sector_window(fn: str, w: int, df: pd.DataFrame) -> pd.Series:
    close = _resolve("close", df)
    volume = _resolve("volume", df)
    sector_index = _sector_index(df)
    sector_ret = _sector_one_day_ret(df)
    stock_ret = _stock_one_day_ret(df)

    if fn == "sector_ret":
        return _finance_ts("ret", sector_index, w, df)
    if fn == "sector_vol":
        return _finance_ts("vol", sector_index, w, df)
    if fn == "sector_drawdown":
        return _finance_ts("drawdown", sector_index, w, df)
    if fn == "sector_ma_ratio":
        return _finance_ts("ma_ratio", sector_index, w, df)
    if fn == "sector_rsi":
        return _finance_ts("rsi", sector_index, w, df)
    if fn == "sector_pos":
        rolling_high = _ts_apply(sector_index, "max", w, df)
        rolling_low = _ts_apply(sector_index, "min", w, df)
        denom = rolling_high - rolling_low
        pos = _safe_div(sector_index - rolling_low, denom)
        return pos.where(~_zero_denominator_mask(denom), 0.5)
    if fn == "sector_volume_ratio":
        sector_volume = _sector_date_sum(volume, df)
        return _safe_div(sector_volume, _ts_apply(sector_volume, "mean", w, df)) - 1.0
    if fn == "rel_sector_ret":
        return _finance_ts("ret", close, w, df) - _finance_ts("ret", sector_index, w, df)
    if fn == "sector_rel_strength":
        stock_ratio = _safe_div(close, _ts_apply(close, "shift", w, df))
        sector_ratio = _safe_div(sector_index, _ts_apply(sector_index, "shift", w, df))
        return _safe_div(stock_ratio, sector_ratio) - 1.0
    if fn == "sector_corr":
        return _ts_pair_apply(stock_ret, sector_ret, "ts_corr", w, df)
    if fn == "sector_beta":
        return _ts_pair_apply(stock_ret, sector_ret, "ts_beta", w, df)
    if fn == "sector_alpha":
        beta = _sector_window("sector_beta", w, df)
        return stock_ret - beta * sector_ret
    if fn == "sector_idiosyncratic_vol":
        residual = _sector_window("sector_alpha", w, df)
        return _ts_apply(residual, "std", w, df)
    if fn == "sector_up_capture":
        stock_up = stock_ret.where(sector_ret > 0)
        sector_up = sector_ret.where(sector_ret > 0)
        return _safe_div(_ts_apply(stock_up, "sum", w, df), _ts_apply(sector_up, "sum", w, df))
    if fn == "sector_down_capture":
        stock_down = stock_ret.where(sector_ret < 0)
        sector_down = sector_ret.where(sector_ret < 0)
        return _safe_div(_ts_apply(stock_down, "sum", w, df), _ts_apply(sector_down, "sum", w, df))
    if fn == "sector_pct_above_ma":
        ma = _ts_apply(close, "mean", w, df)
        valid = close.notna() & ma.notna()
        above = ((close > ma) & valid).astype(float)
        above_count = _sector_date_sum(above, df)
        total = _sector_date_sum(valid.astype(float), df).replace(0.0, np.nan)
        return _safe_div(above_count, total)
    if fn == "sector_breadth_momentum":
        net_pct = _sector_noarg("sector_advance_decline_net_pct", df)
        return _ts_apply(net_pct, "mean", w, df)

    raise ValueError(fn)


def _finance_two_window(fn: str, w1: int, w2: int, df: pd.DataFrame) -> pd.Series:
    if fn == "stoch_d":
        return _ts_apply(_stoch(w1, df), "mean", w2, df)
    raise ValueError(fn)


def _stoch(w: int, df: pd.DataFrame) -> pd.Series:
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    high_max = _ts_apply(high, "max", w, df)
    low_min = _ts_apply(low, "min", w, df)
    return _safe_div(close - low_min, high_max - low_min)


def _typical_price(df: pd.DataFrame) -> pd.Series:
    return (_resolve("high", df) + _resolve("low", df) + _resolve("close", df)) / 3.0


def _money_flow_multiplier(df: pd.DataFrame) -> pd.Series:
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    return _safe_div(((close - low) - (high - close)), high - low)


def _true_range_abs(df: pd.DataFrame) -> pd.Series:
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    prev_close = _ts_apply(close, "shift", 1, df)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr_abs(w: int, df: pd.DataFrame) -> pd.Series:
    true_range = _true_range_abs(df)
    return _ts_apply(true_range, "mean", w, df)


def _adx(w: int, df: pd.DataFrame) -> pd.Series:
    high = _resolve("high", df)
    low = _resolve("low", df)
    up_move = high - _ts_apply(high, "shift", 1, df)
    down_move = _ts_apply(low, "shift", 1, df) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr_sum = _ts_apply(_atr_abs(1, df), "sum", w, df)
    plus_di = 100.0 * _safe_div(_ts_apply(plus_dm, "sum", w, df), tr_sum)
    minus_di = 100.0 * _safe_div(_ts_apply(minus_dm, "sum", w, df), tr_sum)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return _ts_apply(dx, "mean", w, df)


def _finance_noarg(fn: str, df: pd.DataFrame) -> pd.Series:
    open_ = _resolve("open", df)
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    volume = _resolve("volume", df)

    if fn == "body":
        return _safe_div(close - open_, open_)
    if fn == "range":
        return _safe_div(high - low, open_)
    if fn == "gap":
        return _safe_div(open_, _ts_apply(close, "shift", 1, df)) - 1.0
    if fn == "upper_wick":
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        return _safe_div(high - body_high, open_)
    if fn == "lower_wick":
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        return _safe_div(body_low - low, open_)
    if fn == "dollar_volume":
        return close * volume
    if fn == "signed_volume":
        direction = np.sign(close - _ts_apply(close, "shift", 1, df))
        return direction * volume
    if fn == "typical_price":
        return _typical_price(df)
    if fn == "money_flow":
        return _typical_price(df) * volume
    raise ValueError(fn)


def _cumulative_ts(fn: str, series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if fn == "cummax":
        return _ticker_transform(series, lambda s: s.cummax(), df)
    if fn == "cummin":
        return _ticker_transform(series, lambda s: s.cummin(), df)
    if fn == "cum_sum":
        return _ticker_transform(series, lambda s: s.cumsum(), df)
    if fn == "cumret":
        first = _ticker_transform(series, _first_valid_reference, df)
        return _safe_div(series, first) - 1.0
    if fn == "expanding_drawdown":
        return _safe_div(series, _cumulative_ts("cummax", series, df)) - 1.0
    if fn == "expanding_runup":
        return _safe_div(series, _cumulative_ts("cummin", series, df)) - 1.0
    if fn == "days_since_high":
        return _days_since_extreme(series, "high", df)
    if fn == "days_since_low":
        return _days_since_extreme(series, "low", df)
    if fn == "up_streak":
        return _streak(series, "up", df)
    if fn == "down_streak":
        return _streak(series, "down", df)
    raise ValueError(fn)


def _cumulative_noarg(fn: str, df: pd.DataFrame) -> pd.Series:
    if fn == "cum_obv":
        signed_volume = _finance_noarg("signed_volume", df).fillna(0.0)
        return _ticker_transform(signed_volume, lambda s: s.cumsum(), df)
    if fn == "cum_adl":
        ad_flow = (_money_flow_multiplier(df) * _resolve("volume", df)).fillna(0.0)
        return _ticker_transform(ad_flow, lambda s: s.cumsum(), df)
    if fn == "cum_pvt":
        close = _resolve("close", df)
        volume = _resolve("volume", df)
        ret = _safe_div(close, _ts_apply(close, "shift", 1, df)) - 1.0
        pvt_flow = (ret * volume).fillna(0.0)
        return _ticker_transform(pvt_flow, lambda s: s.cumsum(), df)
    raise ValueError(fn)


def _ticker_transform(series: pd.Series, func, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="ticker", group_keys=False).transform(func)
    return func(series)


def _date_transform(series: pd.Series, func, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="date", group_keys=False).transform(func)
    value = func(series)
    return pd.Series(value, index=series.index)


def _date_sum(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="date").transform("sum")
    return pd.Series(float(series.sum()), index=series.index)


def _sector_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for sector_name, tickers in getattr(settings, "SECTORS", {}).items():
        for ticker in tickers:
            mapping[str(ticker).upper()] = str(sector_name)
    return mapping


def _sector_code_map() -> dict[str, float]:
    return {
        str(sector_name): float(i + 1)
        for i, sector_name in enumerate(getattr(settings, "SECTORS", {}).keys())
    }


def _sector_series(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.MultiIndex) or "ticker" not in df.index.names:
        return pd.Series("Unknown", index=df.index)
    mapping = _sector_mapping()
    tickers = df.index.get_level_values("ticker")
    sectors = [mapping.get(str(ticker).upper(), "Unknown") for ticker in tickers]
    return pd.Series(sectors, index=df.index)


def _sector_date_transform(series: pd.Series, func, df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.MultiIndex):
        value = func(series)
        if isinstance(value, pd.Series):
            return value.reindex(series.index)
        return pd.Series(value, index=series.index)
    dates = df.index.get_level_values("date")
    sectors = _sector_series(df)
    return series.groupby([dates, sectors], group_keys=False).transform(func)


def _sector_date_sum(series: pd.Series, df: pd.DataFrame) -> pd.Series:
    if not isinstance(df.index, pd.MultiIndex):
        return pd.Series(float(series.sum()), index=series.index)
    dates = df.index.get_level_values("date")
    sectors = _sector_series(df)
    return series.groupby([dates, sectors]).transform("sum")


def _sector_one_day_ret(df: pd.DataFrame) -> pd.Series:
    return _sector_date_transform(_stock_one_day_ret(df), lambda s: s.mean(), df)


def _sector_index(df: pd.DataFrame) -> pd.Series:
    sector_daily_ret = _sector_one_day_ret(df).fillna(0.0)
    return _ticker_transform(1.0 + sector_daily_ret, lambda s: s.cumprod(), df)


def _stock_one_day_ret(df: pd.DataFrame) -> pd.Series:
    close = _resolve("close", df)
    return _safe_div(close, _ts_apply(close, "shift", 1, df)) - 1.0


def _first_valid_reference(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    value = valid.iloc[0] if len(valid) else np.nan
    return pd.Series(value, index=series.index)


def _days_since_extreme(series: pd.Series, mode: str, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="ticker", group_keys=False).transform(
            lambda s: _days_since_extreme_1d(s, mode)
        )
    return _days_since_extreme_1d(series, mode)


def _days_since_extreme_1d(series: pd.Series, mode: str) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    best = -np.inf if mode == "high" else np.inf
    last_pos = -1

    for i, value in enumerate(values):
        if np.isnan(value):
            continue
        is_new = value >= best if mode == "high" else value <= best
        if is_new or last_pos == -1:
            best = value
            last_pos = i
        out[i] = i - last_pos

    return pd.Series(out, index=series.index)


def _streak(series: pd.Series, mode: str, df: pd.DataFrame) -> pd.Series:
    if isinstance(df.index, pd.MultiIndex):
        return series.groupby(level="ticker", group_keys=False).transform(
            lambda s: _streak_1d(s, mode)
        )
    return _streak_1d(series, mode)


def _streak_1d(series: pd.Series, mode: str) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    count = 0.0

    for i, value in enumerate(values):
        if np.isnan(value):
            count = 0.0
            continue
        if i == 0 or np.isnan(values[i - 1]):
            count = 0.0
        else:
            delta = value - values[i - 1]
            if (mode == "up" and delta > 0) or (mode == "down" and delta < 0):
                count += 1.0
            else:
                count = 0.0
        out[i] = count

    return pd.Series(out, index=series.index)


def _compare(
    fn: str,
    left: pd.Series,
    right: pd.Series,
    df: pd.DataFrame,
) -> pd.Series:
    if fn == "gt":
        return (left > right).astype(float)
    if fn == "lt":
        return (left < right).astype(float)

    prev_left = _ts_apply(left, "shift", 1, df)
    prev_right = _ts_apply(right, "shift", 1, df)
    if fn == "cross_above":
        return ((left > right) & (prev_left <= prev_right)).astype(float)
    if fn == "cross_below":
        return ((left < right) & (prev_left >= prev_right)).astype(float)
    raise ValueError(fn)


def _conditional(fn: str, args: tuple[str, str, str], df: pd.DataFrame) -> pd.Series:
    if fn == "where":
        cond = evaluate(args[0], df)
        true_value = evaluate(args[1], df)
        false_value = evaluate(args[2], df)
        return true_value.where(cond > 0, false_value)

    if fn == "rule_signal":
        expr = evaluate(args[0], df)
        low = evaluate(args[1], df)
        high = evaluate(args[2], df)
        out = pd.Series(0.0, index=df.index)
        out = out.where(~(expr < low), 1.0)
        out = out.where(~(expr > high), -1.0)
        return out

    raise ValueError(fn)


def _binary(op: str, lv: pd.Series, rv: pd.Series) -> pd.Series:
    if op == "+": return lv + rv
    if op == "-": return lv - rv
    if op == "*": return lv * rv
    if op == "/": return _safe_div(lv, rv)
    raise ValueError(op)


def _finance_ts_zero_count(
    fn: str,
    series: pd.Series,
    w: int,
    df: pd.DataFrame,
) -> int:
    if fn in ("ret", "logret"):
        return int(_zero_denominator_mask(_ts_apply(series, "shift", w, df)).sum())
    if fn == "vol":
        return int(_zero_denominator_mask(_ts_apply(series, "shift", 1, df)).sum())
    if fn == "drawdown":
        return int(_zero_denominator_mask(_ts_apply(series, "max", w, df)).sum())
    if fn == "breakout":
        prior_max = _ts_apply(_ts_apply(series, "max", w, df), "shift", 1, df)
        return int(_zero_denominator_mask(prior_max).sum())
    if fn == "ma_ratio":
        return int(_zero_denominator_mask(_ts_apply(series, "mean", w, df)).sum())
    if fn == "bb_pos":
        return int(_zero_denominator_mask(_ts_apply(series, "std", w, df)).sum())
    if fn == "bb_width":
        return int(_zero_denominator_mask(_ts_apply(series, "mean", w, df)).sum())
    if fn == "vol_scale":
        return int(_zero_denominator_mask(_ts_apply(series, "std", w, df)).sum())
    if fn == "efficiency_ratio":
        one_step_move = (series - _ts_apply(series, "shift", 1, df)).abs()
        return int(_zero_denominator_mask(_ts_apply(one_step_move, "sum", w, df)).sum())
    if fn == "ulcer_index":
        return int(_zero_denominator_mask(_ts_apply(series, "max", w, df)).sum())
    return 0


def _finance_window_zero_count(fn: str, w: int, df: pd.DataFrame) -> int:
    open_ = _resolve("open", df)
    high = _resolve("high", df)
    low = _resolve("low", df)
    close = _resolve("close", df)
    volume = _resolve("volume", df)

    if fn in ("pos", "stoch", "willr"):
        denom = _ts_apply(high, "max", w, df) - _ts_apply(low, "min", w, df)
        return int(_zero_denominator_mask(denom).sum())
    if fn == "volume_ratio":
        return int(_zero_denominator_mask(_ts_apply(volume, "mean", w, df)).sum())
    if fn == "atr":
        return int(_zero_denominator_mask(close).sum())
    if fn == "cmf":
        high_low = high - low
        vol_sum = _ts_apply(volume, "sum", w, df)
        return int(
            _zero_denominator_mask(high_low).sum()
            + _zero_denominator_mask(vol_sum).sum()
        )
    if fn == "adx":
        tr_sum = _ts_apply(_atr_abs(1, df), "sum", w, df)
        adx_series = _adx(w, df)
        return int(_zero_denominator_mask(tr_sum).sum()) + int(
            np.isinf(adx_series).sum()
        )
    if fn == "cci":
        typical = _typical_price(df)
        mad = _ts_rolling_apply(
            typical, w, lambda x: float(np.mean(np.abs(x - np.mean(x)))), df
        )
        return int(_zero_denominator_mask(mad).sum())
    if fn == "keltner_pos":
        return int(_zero_denominator_mask(_atr_abs(w, df)).sum())
    if fn == "keltner_width":
        center = _ts_apply(close, "ema", w, df)
        return int(_zero_denominator_mask(center).sum())
    if fn == "donchian_width":
        return int(_zero_denominator_mask(close).sum())
    if fn == "vwap":
        return int(_zero_denominator_mask(_ts_apply(volume, "sum", w, df)).sum())
    if fn == "vwap_pos":
        vwap = _finance_window("vwap", w, df)
        return int(
            _zero_denominator_mask(_ts_apply(volume, "sum", w, df)).sum()
            + _zero_denominator_mask(vwap).sum()
        )
    if fn == "amihud":
        dollar_volume = close * volume
        return int(
            _zero_denominator_mask(_ts_apply(close, "shift", 1, df)).sum()
            + _zero_denominator_mask(dollar_volume).sum()
        )
    if fn == "parkinson_vol":
        return int(_zero_denominator_mask(low).sum())
    if fn == "gk_vol":
        return int(
            _zero_denominator_mask(low).sum()
            + _zero_denominator_mask(open_).sum()
        )
    if fn == "rs_vol":
        return int(
            _zero_denominator_mask(open_).sum()
            + _zero_denominator_mask(close).sum()
        )
    if fn == "choppiness":
        price_range = _ts_apply(high, "max", w, df) - _ts_apply(low, "min", w, df)
        return int(_zero_denominator_mask(price_range).sum())
    return 0


def _market_window_zero_count(fn: str, w: int, df: pd.DataFrame) -> int:
    close = _resolve("close", df)
    market_close = _resolve("market_close", df)
    market_high = _resolve("market_high", df)
    market_low = _resolve("market_low", df)
    market_volume = _resolve("market_volume", df)
    market_ret = _safe_div(
        market_close,
        _ts_apply(market_close, "shift", 1, df),
    ) - 1.0

    if fn in ("market_ret", "market_vol"):
        return int(_zero_denominator_mask(_ts_apply(market_close, "shift", 1, df)).sum())
    if fn == "market_drawdown":
        return int(_zero_denominator_mask(_ts_apply(market_close, "max", w, df)).sum())
    if fn == "market_ma_ratio":
        return int(_zero_denominator_mask(_ts_apply(market_close, "mean", w, df)).sum())
    if fn == "market_pos":
        denom = _ts_apply(market_high, "max", w, df) - _ts_apply(market_low, "min", w, df)
        return int(_zero_denominator_mask(denom).sum())
    if fn == "market_volume_ratio":
        return int(_zero_denominator_mask(_ts_apply(market_volume, "mean", w, df)).sum())
    if fn == "rel_ret":
        return int(
            _zero_denominator_mask(_ts_apply(close, "shift", w, df)).sum()
            + _zero_denominator_mask(_ts_apply(market_close, "shift", w, df)).sum()
        )
    if fn == "rel_strength":
        market_ratio = _safe_div(market_close, _ts_apply(market_close, "shift", w, df))
        return int(
            _zero_denominator_mask(_ts_apply(close, "shift", w, df)).sum()
            + _zero_denominator_mask(_ts_apply(market_close, "shift", w, df)).sum()
            + _zero_denominator_mask(market_ratio).sum()
        )
    if fn in ("market_beta", "market_alpha", "idiosyncratic_vol"):
        denom = _ts_apply(market_ret, "std", w, df) ** 2
        return int(_zero_denominator_mask(denom).sum())
    if fn == "up_capture":
        denom = _ts_apply(market_ret.where(market_ret > 0), "sum", w, df)
        return int(_zero_denominator_mask(denom).sum())
    if fn == "down_capture":
        denom = _ts_apply(market_ret.where(market_ret < 0), "sum", w, df)
        return int(_zero_denominator_mask(denom).sum())
    return 0


def _breadth_noarg_zero_count(fn: str, df: pd.DataFrame) -> int:
    if fn in ("advance_ratio", "decline_ratio", "advance_decline_net_pct"):
        stock_ret = _stock_one_day_ret(df)
        total = _date_sum(stock_ret.notna().astype(float), df).replace(0.0, np.nan)
        return int(_zero_denominator_mask(total).sum())
    return 0


def _breadth_window_zero_count(fn: str, w: int, df: pd.DataFrame) -> int:
    if fn == "pct_above_ma":
        close = _resolve("close", df)
        ma = _ts_apply(close, "mean", w, df)
        valid = close.notna() & ma.notna()
        total = _date_sum(valid.astype(float), df).replace(0.0, np.nan)
        return int(_zero_denominator_mask(total).sum())
    return 0


def _sector_noarg_zero_count(fn: str, df: pd.DataFrame) -> int:
    if fn in (
        "sector_advance_ratio",
        "sector_decline_ratio",
        "sector_advance_decline_net_pct",
    ):
        stock_ret = _stock_one_day_ret(df)
        total = _sector_date_sum(stock_ret.notna().astype(float), df).replace(0.0, np.nan)
        return int(_zero_denominator_mask(total).sum())
    return 0


def _sector_window_zero_count(fn: str, w: int, df: pd.DataFrame) -> int:
    close = _resolve("close", df)
    volume = _resolve("volume", df)
    sector_index = _sector_index(df)
    sector_ret = _sector_one_day_ret(df)

    if fn in ("sector_ret", "sector_vol"):
        return int(_zero_denominator_mask(_ts_apply(sector_index, "shift", 1, df)).sum())
    if fn == "sector_drawdown":
        return int(_zero_denominator_mask(_ts_apply(sector_index, "max", w, df)).sum())
    if fn == "sector_ma_ratio":
        return int(_zero_denominator_mask(_ts_apply(sector_index, "mean", w, df)).sum())
    if fn == "sector_pos":
        return 0
    if fn == "sector_volume_ratio":
        sector_volume = _sector_date_sum(volume, df)
        return int(_zero_denominator_mask(_ts_apply(sector_volume, "mean", w, df)).sum())
    if fn == "rel_sector_ret":
        return int(
            _zero_denominator_mask(_ts_apply(close, "shift", w, df)).sum()
            + _zero_denominator_mask(_ts_apply(sector_index, "shift", w, df)).sum()
        )
    if fn == "sector_rel_strength":
        sector_ratio = _safe_div(sector_index, _ts_apply(sector_index, "shift", w, df))
        return int(
            _zero_denominator_mask(_ts_apply(close, "shift", w, df)).sum()
            + _zero_denominator_mask(_ts_apply(sector_index, "shift", w, df)).sum()
            + _zero_denominator_mask(sector_ratio).sum()
        )
    if fn in ("sector_beta", "sector_alpha", "sector_idiosyncratic_vol"):
        denom = _ts_apply(sector_ret, "std", w, df) ** 2
        return int(_zero_denominator_mask(denom).sum())
    if fn == "sector_up_capture":
        denom = _ts_apply(sector_ret.where(sector_ret > 0), "sum", w, df)
        return int(_zero_denominator_mask(denom).sum())
    if fn == "sector_down_capture":
        denom = _ts_apply(sector_ret.where(sector_ret < 0), "sum", w, df)
        return int(_zero_denominator_mask(denom).sum())
    if fn == "sector_pct_above_ma":
        ma = _ts_apply(close, "mean", w, df)
        valid = close.notna() & ma.notna()
        total = _sector_date_sum(valid.astype(float), df).replace(0.0, np.nan)
        return int(_zero_denominator_mask(total).sum())
    return 0


def _finance_two_window_zero_count(
    fn: str,
    w1: int,
    w2: int,
    df: pd.DataFrame,
) -> int:
    if fn == "stoch_d":
        return _finance_window_zero_count("stoch", w1, df)
    return 0


def _finance_noarg_zero_count(fn: str, df: pd.DataFrame) -> int:
    open_ = _resolve("open", df)
    close = _resolve("close", df)

    if fn in ("body", "range", "upper_wick", "lower_wick"):
        return int(_zero_denominator_mask(open_).sum())
    if fn == "gap":
        return int(_zero_denominator_mask(_ts_apply(close, "shift", 1, df)).sum())
    return 0


def _cumulative_ts_zero_count(
    fn: str,
    series: pd.Series,
    df: pd.DataFrame,
) -> int:
    if fn == "cumret":
        first = _ticker_transform(series, _first_valid_reference, df)
        return int(_zero_denominator_mask(first).sum())
    if fn == "expanding_drawdown":
        return int(_zero_denominator_mask(_cumulative_ts("cummax", series, df)).sum())
    if fn == "expanding_runup":
        return int(_zero_denominator_mask(_cumulative_ts("cummin", series, df)).sum())
    return 0


def _cumulative_noarg_zero_count(fn: str, df: pd.DataFrame) -> int:
    close = _resolve("close", df)
    high = _resolve("high", df)
    low = _resolve("low", df)

    if fn == "cum_obv":
        return 0
    if fn == "cum_adl":
        return int(_zero_denominator_mask(high - low).sum())
    if fn == "cum_pvt":
        return int(_zero_denominator_mask(_ts_apply(close, "shift", 1, df)).sum())
    return 0


def _safe_div(lv: pd.Series, rv: pd.Series) -> pd.Series:
    return lv / rv.mask(_zero_denominator_mask(rv), np.nan)


def _zero_denominator_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & np.isclose(
        values, 0.0, atol=DIV_ZERO_EPS, rtol=0.0
    )


def _resolve(base: str, df: pd.DataFrame) -> pd.Series:
    col = _COL_ALIASES.get(base.lower())
    if col is None or col not in df.columns:
        raise ValueError(f"Unknown base: {base!r}")
    return df[col]
