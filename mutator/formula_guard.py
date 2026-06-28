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
_CS_NORMALIZER_OPS = {"rank", "zscore", "sector_rank", "sector_zscore"}
_TS_NORMALIZER_OPS = {"ts_rank", "ts_zscore"}
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
    "parkinson_vol", "gk_vol", "rs_vol", "aroon_up", "aroon_down",
    "aroon_osc", "choppiness",
}
_MARKET_WINDOW_NORMALIZED_OPS = {
    "market_ret", "market_vol", "market_drawdown", "market_ma_ratio",
    "market_rsi", "market_pos", "market_volume_ratio", "rel_ret",
    "rel_strength", "market_corr", "market_beta", "market_alpha",
    "idiosyncratic_vol", "up_capture", "down_capture",
}
_BREADTH_NORMALIZED_OPS = {
    "advance_ratio", "decline_ratio", "advance_decline_ratio",
    "advance_decline_net_pct", "cs_dispersion",
    "pct_above_ma", "breadth_momentum",
}
_SECTOR_NORMALIZED_OPS = {
    "sector_advance_ratio", "sector_decline_ratio",
    "sector_advance_decline_ratio",
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
BLOCKED_EVOLUTION_PRIMITIVES = frozenset({
    "advance_count", "decline_count", "unchanged_count",
    "advance_decline_spread",
    "sector_code", "sector_size",
    "sector_advance_count", "sector_decline_count", "sector_unchanged_count",
    "sector_advance_decline_spread",
})
_BLOCKED_ID_COUNT_OPS = BLOCKED_EVOLUTION_PRIMITIVES


def is_const_threshold_safe(formula: str) -> bool:
    """Return True when every const threshold is applied to a normalized expr."""
    return const_threshold_violation(formula) is None


def const_threshold_violation(formula: str) -> str | None:
    """Return a human-readable violation reason, or None if the formula is safe."""
    return _violation(formula.strip())


def is_normalized_for_const_threshold(expr: str) -> bool:
    """Return True if ``expr`` can be compared to a fixed numeric threshold."""
    return _is_normalized_expr(expr.strip())


def is_raw_scale_safe(formula: str) -> bool:
    """Return True when the formula output is not raw price/volume scale."""
    return raw_scale_violation(formula) is None


def raw_scale_violation(formula: str) -> str | None:
    """
    Return a violation when a selected feature still carries raw scale.

    Raw OHLCV and market index levels are useful as ingredients, but allowing
    them as final model features lets tree splits learn absolute price levels
    or time/regime proxies. Relative/normalized transforms remain allowed.
    """
    formula = formula.strip()
    if _is_constant_only(formula):
        return f"constant-only feature is not allowed: {formula!r}"

    blocked_primitive = _blocked_id_count_violation(formula)
    if blocked_primitive is not None:
        return blocked_primitive

    normalizer_violation = _cs_normalizer_input_violation(formula)
    if normalizer_violation is not None:
        return normalizer_violation

    bounded_violation = _bounded_raw_input_violation(formula)
    if bounded_violation is not None:
        return bounded_violation

    comparison_violation = _raw_scale_comparison_violation(formula)
    if comparison_violation is not None:
        return comparison_violation

    scale = _output_scale_kind(formula)
    if scale in {
        "price", "volume", "market_price", "market_volume",
        "dollar_volume", "raw_mixed", "inverse_raw",
    }:
        return f"raw-scale feature is not allowed: {formula!r} has scale {scale!r}"
    return None


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
    if fn in _CS_NORMALIZER_OPS:
        return _is_normalized_expr(args)
    if fn in _TS_NORMALIZER_OPS or fn in _BOUNDED_ANY_OPS:
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
    if fn in {"ts_corr", "ts_beta", "ts_cov"}:
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


def _output_scale_kind(expr: str) -> str | None:
    expr = _canonical(expr)
    if not expr:
        return None

    if _is_constant_only(expr):
        return "dimensionless"

    match = _TERMINAL_RE.match(expr)
    if match:
        return _terminal_scale(match.group(1))

    parsed_tag = _parse_window_tag(expr)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _output_scale_kind(inner)

    op, left, right = _split_binary(expr)
    if op is not None:
        return _binary_output_scale(op, left, right)

    fn, args = _parse_fn(expr)
    if fn is None:
        return None
    if fn == "const":
        return "dimensionless"
    if fn in _COMPARE_OPS or fn == "rule_signal":
        return "dimensionless"
    if fn == "where":
        parsed = _parse_three_expr_args(args)
        if parsed is None:
            return None
        return _combine_branch_scales(
            _output_scale_kind(parsed[1]),
            _output_scale_kind(parsed[2]),
        )
    if fn in _CS_NORMALIZER_OPS:
        return "dimensionless" if _is_normalized_expr(args) else "raw_mixed"
    if fn in _TS_NORMALIZER_OPS or fn in _BOUNDED_ANY_OPS:
        return "dimensionless"
    if fn in _PRESERVE_IF_NORMALIZED_OPS:
        return _output_scale_kind(args)
    if fn in _TS_PRESERVE_IF_NORMALIZED_OPS:
        parsed = _split_ts_args(args)
        return _output_scale_kind(parsed[0]) if parsed[0] is not None else None
    if fn in _FINANCE_TS_NORMALIZED_OPS:
        return "dimensionless"
    if fn in _FINANCE_TS_IF_INPUT_NORMALIZED_OPS:
        parsed = _split_ts_args(args)
        if parsed[0] is None:
            return None
        return "dimensionless" if _is_normalized_expr(parsed[0]) else "raw_mixed"
    if fn == "delta":
        parsed = _split_ts_args(args)
        return _output_scale_kind(parsed[0]) if parsed[0] is not None else None
    if fn in _FINANCE_WINDOW_NORMALIZED_OPS:
        return "dimensionless"
    if fn == "amihud":
        return "inverse_raw"
    if fn in {"body", "range", "gap", "upper_wick", "lower_wick"}:
        return "dimensionless"
    if fn in {"liquidity", "dollar_volume", "money_flow"}:
        return "dollar_volume"
    if fn in {"obv", "signed_volume"}:
        return "volume"
    if fn in {"vwap", "typical_price"}:
        return "price"
    if fn in _MARKET_WINDOW_NORMALIZED_OPS:
        return "dimensionless"
    if fn in _BREADTH_NORMALIZED_OPS:
        return "dimensionless"
    if fn in _BLOCKED_ID_COUNT_OPS:
        return "id_or_count"
    if fn in _SECTOR_NORMALIZED_OPS:
        return "dimensionless"
    if fn in _CUMULATIVE_NORMALIZED_OPS:
        return "dimensionless"
    if fn in {"cummax", "cummin", "cum_sum"}:
        return _output_scale_kind(args)
    if fn in {"days_since_high", "days_since_low", "up_streak", "down_streak"}:
        return "dimensionless"
    if fn in {"cum_obv", "cum_adl", "cum_pvt"}:
        return "volume"
    if fn == "stoch_d":
        return "dimensionless"
    if fn in {"ts_corr", "ts_beta", "ts_cov"}:
        parsed = _parse_pair_ts_args(args)
        if parsed is None:
            return None
        if not (
            _is_normalized_expr(parsed[0])
            and _is_normalized_expr(parsed[1])
        ):
            return "raw_mixed"
        return "dimensionless"

    return None


def _binary_output_scale(op: str, left: str, right: str) -> str | None:
    left_scale = _output_scale_kind(left)
    right_scale = _output_scale_kind(right)
    left_raw = _is_raw_scale(left_scale)
    right_raw = _is_raw_scale(right_scale)

    if op in {"+", "-"}:
        if left_raw and right_raw:
            return left_scale if left_scale == right_scale else "raw_mixed"
        if left_raw:
            return left_scale
        if right_raw:
            return right_scale
        if left_scale == "dimensionless" and right_scale == "dimensionless":
            return "dimensionless"
        return None

    if op == "*":
        if left_raw and right_raw:
            return "raw_mixed"
        if left_raw:
            return left_scale
        if right_raw:
            return right_scale
        if left_scale == "dimensionless" and right_scale == "dimensionless":
            return "dimensionless"
        return None

    if op == "/":
        if left_raw and right_raw:
            return "dimensionless" if left_scale == right_scale else "raw_mixed"
        if left_raw:
            return left_scale
        if right_raw:
            return "inverse_raw"
        if left_scale == "dimensionless" and right_scale == "dimensionless":
            return "dimensionless"
        return None

    return None


def _combine_branch_scales(left_scale: str | None, right_scale: str | None) -> str | None:
    if _is_raw_scale(left_scale) or _is_raw_scale(right_scale):
        if left_scale == right_scale:
            return left_scale
        return "raw_mixed"
    if left_scale == "dimensionless" and right_scale == "dimensionless":
        return "dimensionless"
    return None


def _terminal_scale(base: str) -> str | None:
    if base in _RAW_PRICE_BASES:
        return "price"
    if base in _RAW_VOLUME_BASES:
        return "volume"
    if base in _MARKET_PRICE_BASES:
        return "market_price"
    if base in _MARKET_VOLUME_BASES:
        return "market_volume"
    return None


def _is_raw_scale(scale: str | None) -> bool:
    return scale in {
        "price", "volume", "market_price", "market_volume",
        "dollar_volume", "raw_mixed", "inverse_raw",
    }


def _blocked_id_count_violation(formula: str) -> str | None:
    formula = _canonical(formula)
    if not formula:
        return None

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _blocked_id_count_violation(inner)

    op, left, right = _split_binary(formula)
    if op is not None:
        return (
            _blocked_id_count_violation(left)
            or _blocked_id_count_violation(right)
        )

    fn, args = _parse_fn(formula)
    if fn is None:
        return None

    if fn in _BLOCKED_ID_COUNT_OPS:
        return (
            "non-economic ID/count primitive is not allowed as a selected feature: "
            f"{formula!r}"
        )

    for part in _split_top_level(args, ","):
        if _is_numeric_literal(part):
            continue
        nested = _blocked_id_count_violation(part)
        if nested is not None:
            return nested

    return None


def _raw_scale_comparison_violation(formula: str) -> str | None:
    formula = _canonical(formula)
    if not formula:
        return None

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _raw_scale_comparison_violation(inner)

    op, left, right = _split_binary(formula)
    if op is not None:
        return (
            _raw_scale_comparison_violation(left)
            or _raw_scale_comparison_violation(right)
        )

    fn, args = _parse_fn(formula)
    if fn is None:
        return None

    if fn in _COMPARE_OPS:
        parsed = _parse_two_expr_args(args)
        if parsed is None:
            return None
        left, right = parsed
        nested = (
            _raw_scale_comparison_violation(left)
            or _raw_scale_comparison_violation(right)
        )
        if nested is not None:
            return nested
        return _format_raw_scale_comparison_violation(formula, left, right)

    if fn == "rule_signal":
        parsed = _parse_three_expr_args(args)
        if parsed is None:
            return None
        expr, low, high = parsed
        nested = (
            _raw_scale_comparison_violation(expr)
            or _raw_scale_comparison_violation(low)
            or _raw_scale_comparison_violation(high)
        )
        if nested is not None:
            return nested
        return (
            _format_raw_scale_comparison_violation(formula, expr, low)
            or _format_raw_scale_comparison_violation(formula, expr, high)
        )

    for part in _split_top_level(args, ","):
        if _is_numeric_literal(part):
            continue
        nested = _raw_scale_comparison_violation(part)
        if nested is not None:
            return nested

    return None


def _bounded_raw_input_violation(formula: str) -> str | None:
    formula = _canonical(formula)
    if not formula:
        return None

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _bounded_raw_input_violation(inner)

    op, left, right = _split_binary(formula)
    if op is not None:
        return (
            _bounded_raw_input_violation(left)
            or _bounded_raw_input_violation(right)
        )

    fn, args = _parse_fn(formula)
    if fn is None:
        return None

    if fn in _BOUNDED_ANY_OPS:
        nested = _bounded_raw_input_violation(args)
        if nested is not None:
            return nested
        arg_scale = _output_scale_kind(args)
        if _contains_const(args) and not _is_normalized_expr(args):
            return (
                "bounded transform cannot hide an absolute raw-scale threshold: "
                f"{formula!r} uses {args!r}"
            )
        if arg_scale in {"raw_mixed", "inverse_raw", "dollar_volume"}:
            return (
                "bounded transform cannot hide incompatible raw-scale input: "
                f"{formula!r} uses {args!r} ({arg_scale})"
            )
        return None

    for part in _split_top_level(args, ","):
        if _is_numeric_literal(part):
            continue
        nested = _bounded_raw_input_violation(part)
        if nested is not None:
            return nested

    return None


def _format_raw_scale_comparison_violation(
    formula: str,
    left: str,
    right: str,
) -> str | None:
    left_scale = _output_scale_kind(left)
    right_scale = _output_scale_kind(right)
    left_raw = _is_raw_scale(left_scale)
    right_raw = _is_raw_scale(right_scale)
    if left_raw != right_raw:
        return (
            "comparison cannot mix raw-scale and normalized/relative inputs: "
            f"{formula!r} compares {left!r} ({left_scale}) with "
            f"{right!r} ({right_scale})"
        )
    if not (left_raw and right_raw):
        return None
    if left_scale == right_scale and left_scale not in {"raw_mixed", "inverse_raw"}:
        return None
    return (
        "comparison between incompatible raw-scale systems is not allowed: "
        f"{formula!r} compares {left!r} ({left_scale}) with {right!r} ({right_scale})"
    )


def _cs_normalizer_input_violation(formula: str) -> str | None:
    formula = _canonical(formula)
    if not formula:
        return None

    parsed_tag = _parse_window_tag(formula)
    if parsed_tag is not None:
        inner, _ = parsed_tag
        return _cs_normalizer_input_violation(inner)

    op, left, right = _split_binary(formula)
    if op is not None:
        return (
            _cs_normalizer_input_violation(left)
            or _cs_normalizer_input_violation(right)
        )

    fn, args = _parse_fn(formula)
    if fn is None:
        return None

    if fn in _CS_NORMALIZER_OPS:
        nested = _cs_normalizer_input_violation(args)
        if nested is not None:
            return nested
        if not _is_normalized_expr(args):
            return (
                "cross-sectional rank/zscore requires normalized/relative input: "
                f"{formula!r} uses {args!r}"
            )
        return None

    for part in _split_top_level(args, ","):
        if _is_numeric_literal(part):
            continue
        nested = _cs_normalizer_input_violation(part)
        if nested is not None:
            return nested

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
