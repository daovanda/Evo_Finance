"""
Formula safety checks used before admitting evolved features.

The main guard here prevents absolute constant thresholds on raw price/volume
scale features, for example ``lt(low_60, const(48))``. Constants are still
allowed for arithmetic scaling and for thresholds on normalized/relative
features such as RSI, rank, returns, z-scores, breadth ratios, and similar
dimensionless indicators.
"""

from __future__ import annotations

import re

from mutator.evaluator import (
    _balanced,
    _parse_float_arg,
    _parse_fn,
    _parse_pair_ts_args,
    _parse_three_expr_args,
    _parse_two_expr_args,
    _parse_window_tag,
    _split_binary,
    _split_top_level,
    _split_ts_args,
)


_TERMINAL_RE = re.compile(r"^([a-z_]+)_(\d+)$")

_COMPARE_OPS = {"gt", "lt", "cross_above", "cross_below"}
_NORMALIZER_OPS = {"rank", "zscore", "sector_rank", "sector_zscore", "ts_rank", "ts_zscore"}
_BOUNDED_ANY_OPS = {"sign"}
_PRESERVE_IF_NORMALIZED_OPS = {
    "abs", "signed_log", "clip", "pos_part", "neg_part",
    "winsorize", "neutralize", "sector_neutralize",
}
_TS_PRESERVE_IF_NORMALIZED_OPS = {
    "std", "max", "min", "shift", "sum", "ema",
    "median", "q25", "q75", "iqr", "skew", "kurt",
    "decay_linear", "slope",
}
_FINANCE_TS_NORMALIZED_OPS = {
    "ret", "logret", "vol", "drawdown", "breakout", "ma_ratio",
    "bb_pos", "bb_width", "rsi", "efficiency_ratio", "ulcer_index",
}
_FINANCE_TS_IF_INPUT_NORMALIZED_OPS = {"vol_scale"}
_FINANCE_WINDOW_NORMALIZED_OPS = {
    "pos", "volume_ratio", "atr", "stoch", "mfi", "cmf", "adx", "cci",
    "willr", "keltner_pos", "keltner_width", "donchian_width", "vwap_pos",
    "amihud", "parkinson_vol", "gk_vol", "rs_vol", "aroon_up",
    "aroon_down", "aroon_osc", "choppiness",
}
_MARKET_WINDOW_NORMALIZED_OPS = {
    "market_ret", "market_vol", "market_drawdown", "market_ma_ratio",
    "market_rsi", "market_pos", "market_volume_ratio", "rel_ret",
    "rel_strength", "market_corr", "market_beta", "market_alpha",
    "idiosyncratic_vol", "up_capture", "down_capture",
}
_BREADTH_NORMALIZED_OPS = {
    "advance_ratio", "decline_ratio", "advance_decline_ratio",
    "advance_decline_spread", "advance_decline_net_pct", "cs_dispersion",
    "pct_above_ma", "breadth_momentum",
}
_SECTOR_NORMALIZED_OPS = {
    "sector_advance_ratio", "sector_decline_ratio",
    "sector_advance_decline_ratio", "sector_advance_decline_spread",
    "sector_advance_decline_net_pct", "sector_dispersion", "sector_ret",
    "sector_vol", "sector_drawdown", "sector_ma_ratio", "sector_rsi",
    "sector_pos", "sector_volume_ratio", "rel_sector_ret",
    "sector_rel_strength", "sector_corr", "sector_beta", "sector_alpha",
    "sector_idiosyncratic_vol", "sector_up_capture",
    "sector_down_capture", "sector_pct_above_ma",
    "sector_breadth_momentum",
}
_CUMULATIVE_NORMALIZED_OPS = {"cumret", "expanding_drawdown", "expanding_runup"}

_RAW_PRICE_BASES = {"open", "high", "low", "close", "o", "h", "l", "c"}
_RAW_VOLUME_BASES = {"volume", "v"}
_MARKET_PRICE_BASES = {
    "market_open", "market_high", "market_low", "market_close",
    "m_open", "m_high", "m_low", "m_close",
    "vnindex_open", "vnindex_high", "vnindex_low", "vnindex_close",
}
_MARKET_VOLUME_BASES = {"market_volume", "m_volume", "vnindex_volume"}


def is_const_threshold_safe(formula: str) -> bool:
    """Return True when every const threshold is applied to a normalized expr."""
    return const_threshold_violation(formula) is None


def const_threshold_violation(formula: str) -> str | None:
    """Return a human-readable violation reason, or None if the formula is safe."""
    return _violation(formula.strip())


def is_normalized_for_const_threshold(expr: str) -> bool:
    """Return True if ``expr`` can be compared to a fixed numeric threshold."""
    return _is_normalized_expr(expr.strip())


def _violation(formula: str) -> str | None:
    formula = _canonical(formula)
    if not formula:
        return None

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _violation(inner)

    op, left, right = _split_binary(formula)
    if op is not None:
        return _violation(left) or _violation(right)

    fn, args = _parse_fn(formula)
    if fn in _COMPARE_OPS:
        parsed = _parse_two_expr_args(args)
        if parsed is None:
            return None
        left, right = parsed
        nested = _violation(left) or _violation(right)
        if nested is not None:
            return nested

        left_const = _is_constant_only(left)
        right_const = _is_constant_only(right)
        if left_const and not _is_normalized_expr(right):
            return _format_threshold_violation(formula, right)
        if right_const and not _is_normalized_expr(left):
            return _format_threshold_violation(formula, left)
        return None

    if fn == "rule_signal":
        parsed = _parse_three_expr_args(args)
        if parsed is None:
            return None
        expr, low, high = parsed
        nested = _violation(expr) or _violation(low) or _violation(high)
        if nested is not None:
            return nested
        if (_contains_const(low) or _contains_const(high)) and not _is_normalized_expr(expr):
            return _format_threshold_violation(formula, expr)
        return None

    if fn is not None:
        for part in _split_top_level(args, ","):
            if _is_numeric_literal(part):
                continue
            nested = _violation(part)
            if nested is not None:
                return nested
        return None

    return None


def _is_normalized_expr(expr: str) -> bool:
    expr = _canonical(expr)
    if not expr or _is_constant_only(expr):
        return False

    parsed_tag = _parse_window_tag(expr)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _is_normalized_expr(inner)

    if _TERMINAL_RE.match(expr):
        return False

    op, left, right = _split_binary(expr)
    if op is not None:
        return _is_normalized_binary(op, left, right)

    fn, args = _parse_fn(expr)
    if fn is None:
        return False
    if fn == "const":
        return False
    if fn in _COMPARE_OPS:
        return _violation(expr) is None
    if fn == "where":
        parsed = _parse_three_expr_args(args)
        if parsed is None:
            return False
        return (
            _violation(parsed[0]) is None
            and _is_normalized_expr(parsed[1])
            and _is_normalized_expr(parsed[2])
        )
    if fn == "rule_signal":
        return _violation(expr) is None
    if fn in _NORMALIZER_OPS or fn in _BOUNDED_ANY_OPS:
        return True
    if fn in _PRESERVE_IF_NORMALIZED_OPS:
        return _is_normalized_expr(args)
    if fn in _TS_PRESERVE_IF_NORMALIZED_OPS:
        parsed = _split_ts_args(args)
        return parsed[0] is not None and _is_normalized_expr(parsed[0])
    if fn in _FINANCE_TS_NORMALIZED_OPS:
        return True
    if fn in _FINANCE_TS_IF_INPUT_NORMALIZED_OPS:
        parsed = _split_ts_args(args)
        return parsed[0] is not None and _is_normalized_expr(parsed[0])
    if fn in _FINANCE_WINDOW_NORMALIZED_OPS:
        return True
    if fn in _MARKET_WINDOW_NORMALIZED_OPS:
        return True
    if fn in _BREADTH_NORMALIZED_OPS:
        return True
    if fn in _SECTOR_NORMALIZED_OPS:
        return True
    if fn in _CUMULATIVE_NORMALIZED_OPS:
        return True
    if fn == "stoch_d":
        return True
    if fn == "ts_corr":
        return True
    if fn in {"ts_beta", "ts_cov"}:
        parsed = _parse_pair_ts_args(args)
        return (
            parsed is not None
            and _is_normalized_expr(parsed[0])
            and _is_normalized_expr(parsed[1])
        )

    return False


def _is_normalized_binary(op: str, left: str, right: str) -> bool:
    left_const = _is_constant_only(left)
    right_const = _is_constant_only(right)
    left_norm = _is_normalized_expr(left)
    right_norm = _is_normalized_expr(right)

    if op in {"+", "-"}:
        if left_const and right_norm:
            return True
        if right_const and left_norm:
            return True
        return left_norm and right_norm

    if op == "*":
        if left_const and right_norm:
            return True
        if right_const and left_norm:
            return True
        return left_norm and right_norm

    if op == "/":
        if left_const and right_norm:
            return True
        if right_const and left_norm:
            return True
        if left_norm and right_norm:
            return True
        left_scale = _scale_kind(left)
        right_scale = _scale_kind(right)
        return left_scale is not None and left_scale == right_scale

    return False


def _is_constant_only(expr: str) -> bool:
    expr = _canonical(expr)
    if not expr:
        return False

    fn, args = _parse_fn(expr)
    if fn == "const":
        return _parse_float_arg(args) is not None

    op, left, right = _split_binary(expr)
    if op is not None:
        return _is_constant_only(left) and _is_constant_only(right)

    parsed_tag = _parse_window_tag(expr)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _is_constant_only(inner)

    return False


def _contains_const(expr: str) -> bool:
    expr = _canonical(expr)
    if not expr:
        return False

    fn, args = _parse_fn(expr)
    if fn == "const":
        return True

    parsed_tag = _parse_window_tag(expr)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _contains_const(inner)

    op, left, right = _split_binary(expr)
    if op is not None:
        return _contains_const(left) or _contains_const(right)

    if fn is not None:
        return any(
            _contains_const(part)
            for part in _split_top_level(args, ",")
            if not _is_numeric_literal(part)
        )

    return False


def _scale_kind(expr: str) -> str | None:
    expr = _canonical(expr)

    match = _TERMINAL_RE.match(expr)
    if match:
        base = match.group(1)
        if base in _RAW_PRICE_BASES:
            return "price"
        if base in _RAW_VOLUME_BASES:
            return "volume"
        if base in _MARKET_PRICE_BASES:
            return "market_price"
        if base in _MARKET_VOLUME_BASES:
            return "market_volume"
        return None

    parsed_tag = _parse_window_tag(expr)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _scale_kind(inner)

    op, left, right = _split_binary(expr)
    if op in {"+", "-"}:
        left_scale = _scale_kind(left)
        right_scale = _scale_kind(right)
        return left_scale if left_scale == right_scale else None
    if op == "*":
        if _is_constant_only(left):
            return _scale_kind(right)
        if _is_constant_only(right):
            return _scale_kind(left)
        return None
    if op == "/":
        if _is_constant_only(right):
            return _scale_kind(left)
        return None

    fn, args = _parse_fn(expr)
    if fn is None:
        return None
    if fn in {"ema", "max", "min", "shift", "median", "q25", "q75", "decay_linear"}:
        parsed = _split_ts_args(args)
        return _scale_kind(parsed[0]) if parsed[0] is not None else None
    if fn in {"std", "iqr"}:
        parsed = _split_ts_args(args)
        return _scale_kind(parsed[0]) if parsed[0] is not None else None
    if fn in {"vwap", "typical_price"}:
        return "price"

    return None


def _canonical(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")") and _balanced(expr):
        expr = expr[1:-1].strip()
    return expr


def _is_numeric_literal(text: str) -> bool:
    try:
        float(text.strip())
        return True
    except (TypeError, ValueError):
        return False


def _format_threshold_violation(formula: str, expr: str) -> str:
    return (
        "const threshold requires normalized/relative feature: "
        f"{formula!r} uses {expr!r}"
    )
