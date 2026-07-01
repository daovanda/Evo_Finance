"""Expression evaluation and safety guards for crypto features."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto import config


UNARY_OPS = ("abs", "signed_log", "sign", "clip", "pos_part", "neg_part")
ROLLING_OPS = (
    "mean",
    "std",
    "min",
    "max",
    "sum",
    "ema",
    "median",
    "q25",
    "q75",
    "iqr",
    "ts_zscore",
    "ts_rank",
    "delta",
    "shift",
    "slope",
    "decay_linear",
)
PAIR_TS_OPS = ("ts_corr", "ts_beta", "ts_cov")
COMPARE_OPS = ("gt", "lt", "cross_above", "cross_below")
CONDITIONAL_OPS = ("where", "rule_signal")
BINARY_OPS = ("+", "-", "*", "/")
CONSTANT_VALUES = (
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    -0.2,
    -0.1,
    -0.05,
    0.0,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    70.0,
)
_CONST_RE = re.compile(r"^const\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)$")


@dataclass(frozen=True)
class FeatureQuality:
    ok: bool
    reason: str = ""


class CryptoFeatureSpace:
    """
    Evaluate generated formulas over a safe base feature frame.

    Raw OHLCV/volume/trade-count columns are never exposed here. Any generated
    expression must be built from safe base features supplied by
    crypto.features.selectable_features().
    """

    def __init__(
        self,
        base_df: pd.DataFrame,
        base_features: list[str],
    ):
        self.base_df = base_df.sort_index()
        self.base_features = list(dict.fromkeys(base_features))
        missing = [name for name in self.base_features if name not in self.base_df.columns]
        if missing:
            raise ValueError(f"Base feature frame missing features: {missing[:5]}")
        self._base_set = set(self.base_features)
        self._cache: dict[str, pd.Series] = {}

    def evaluate(self, formula: str) -> pd.Series:
        formula = str(formula).strip()
        cached = self._cache.get(formula)
        if cached is not None:
            return cached

        series = self._evaluate_uncached(formula)
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        series = series.reindex(self.base_df.index)
        self._cache[formula] = series
        return series

    def matrix(self, formulas: list[str], index: pd.Index | None = None) -> pd.DataFrame:
        cols = {}
        for formula in formulas:
            name = _sanitize_feature_name(formula)
            if name in cols:
                name = f"{name}_{len(cols)}"
            cols[name] = self.evaluate(formula)
        matrix = pd.DataFrame(cols, index=self.base_df.index)
        if index is not None:
            matrix = matrix.loc[index]
        return matrix.replace([np.inf, -np.inf], np.nan)

    def quality(
        self,
        formula: str,
        index: pd.Index,
        min_valid_ratio: float = config.FEATURE_MIN_VALID_RATIO,
        max_dominant_value_ratio: float = config.FEATURE_MAX_DOMINANT_VALUE_RATIO,
    ) -> FeatureQuality:
        if len(formula) > config.EXPR_MAX_LENGTH:
            return FeatureQuality(False, "formula too long")
        if formula_depth(formula, self._base_set) > config.EXPR_MAX_DEPTH:
            return FeatureQuality(False, "formula too deep")
        try:
            series = self.evaluate(formula).loc[index]
        except Exception as exc:
            return FeatureQuality(False, f"eval failed: {exc}")

        valid_ratio = float(series.notna().mean()) if len(series) else 0.0
        if valid_ratio < min_valid_ratio:
            return FeatureQuality(False, "too many missing values")
        valid = series.dropna()
        if len(valid) < 20:
            return FeatureQuality(False, "too few valid values")
        if valid.nunique(dropna=True) < 2:
            return FeatureQuality(False, "constant feature")
        dominant = float(valid.value_counts(normalize=True, dropna=True).iloc[0])
        if dominant > max_dominant_value_ratio:
            return FeatureQuality(False, "dominant value too frequent")
        abs_q = float(valid.abs().quantile(0.995))
        if not np.isfinite(abs_q) or abs_q > config.EXPR_MAX_ABS_QUANTILE:
            return FeatureQuality(False, "extreme values")
        return FeatureQuality(True)

    def _evaluate_uncached(self, formula: str) -> pd.Series:
        if formula in self._base_set:
            return self.base_df[formula]

        const_match = _CONST_RE.match(formula)
        if const_match:
            return pd.Series(float(const_match.group(1)), index=self.base_df.index)

        fn, args = _parse_fn(formula)
        if fn in UNARY_OPS:
            return _unary(fn, self.evaluate(args))
        if fn in ROLLING_OPS:
            parsed = _split_window_args(args)
            if parsed is not None:
                expr, window = parsed
                return _rolling(fn, self.evaluate(expr), window)
        if fn in PAIR_TS_OPS:
            parsed_pair = _split_pair_window_args(args)
            if parsed_pair is not None:
                left, right, window = parsed_pair
                return _pair_rolling(fn, self.evaluate(left), self.evaluate(right), window)
        if fn in COMPARE_OPS:
            parsed_two = _split_expr_args(args, expected=2)
            if parsed_two is not None:
                return _compare(fn, self.evaluate(parsed_two[0]), self.evaluate(parsed_two[1]))
        if fn in CONDITIONAL_OPS:
            parsed_three = _split_expr_args(args, expected=3)
            if parsed_three is not None:
                if fn == "where":
                    cond = self.evaluate(parsed_three[0])
                    return pd.Series(
                        np.where(cond > 0, self.evaluate(parsed_three[1]), self.evaluate(parsed_three[2])),
                        index=self.base_df.index,
                    )
                return _rule_signal(
                    self.evaluate(parsed_three[0]),
                    self.evaluate(parsed_three[1]),
                    self.evaluate(parsed_three[2]),
                )

        op, left, right = _split_binary(formula)
        if op is not None:
            return _binary(op, self.evaluate(left), self.evaluate(right))

        if formula.startswith("(") and formula.endswith(")") and _balanced(formula):
            return self.evaluate(formula[1:-1])

        raise ValueError(f"Cannot evaluate crypto formula: {formula!r}")


def formula_depth(formula: str, base_features: set[str]) -> int:
    formula = formula.strip()
    if formula in base_features or _CONST_RE.match(formula):
        return 1
    op, left, right = _split_binary(formula)
    if op is not None:
        return 1 + max(formula_depth(left, base_features), formula_depth(right, base_features))
    fn, args = _parse_fn(formula)
    if fn:
        parts = _split_top_level(args, ",")
        child_depths = [
            formula_depth(part, base_features)
            for part in parts
            if part and not part.strip().isdigit()
        ]
        return 1 + (max(child_depths) if child_depths else 0)
    if formula.startswith("(") and formula.endswith(")") and _balanced(formula):
        return formula_depth(formula[1:-1], base_features)
    return config.EXPR_MAX_DEPTH + 1


def constant_formula(value: float) -> str:
    return f"const({_format_number(value)})"


def _unary(fn: str, series: pd.Series) -> pd.Series:
    if fn == "abs":
        return series.abs()
    if fn == "signed_log":
        return np.sign(series) * np.log1p(series.abs())
    if fn == "sign":
        return np.sign(series)
    if fn == "clip":
        return series.clip(-5.0, 5.0)
    if fn == "pos_part":
        return series.clip(lower=0.0)
    if fn == "neg_part":
        return series.clip(upper=0.0)
    raise ValueError(f"Unknown unary op: {fn}")


def _rolling(fn: str, series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=window)
    if fn == "mean":
        return rolling.mean()
    if fn == "std":
        return rolling.std()
    if fn == "min":
        return rolling.min()
    if fn == "max":
        return rolling.max()
    if fn == "sum":
        return rolling.sum()
    if fn == "ema":
        return series.ewm(span=window, min_periods=window, adjust=False).mean()
    if fn == "median":
        return rolling.median()
    if fn == "q25":
        return rolling.quantile(0.25)
    if fn == "q75":
        return rolling.quantile(0.75)
    if fn == "iqr":
        return rolling.quantile(0.75) - rolling.quantile(0.25)
    if fn == "ts_zscore":
        return _safe_div(series - rolling.mean(), rolling.std())
    if fn == "ts_rank":
        return rolling.apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)
    if fn == "delta":
        return series.diff(window)
    if fn == "shift":
        return series.shift(window)
    if fn == "slope":
        x = np.arange(window, dtype=float)

        def slope(values) -> float:
            y = np.asarray(values, dtype=float)
            if np.isnan(y).any():
                return np.nan
            x_center = x - x.mean()
            denom = float((x_center ** 2).sum())
            return float(((y - y.mean()) * x_center).sum() / denom) if denom else np.nan

        return rolling.apply(slope, raw=True)
    if fn == "decay_linear":
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        return rolling.apply(lambda values: float(np.dot(values, weights)), raw=True)
    raise ValueError(f"Unknown rolling op: {fn}")


def _pair_rolling(fn: str, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    if fn == "ts_corr":
        return left.rolling(window, min_periods=window).corr(right)
    if fn == "ts_cov":
        return left.rolling(window, min_periods=window).cov(right)
    if fn == "ts_beta":
        cov = left.rolling(window, min_periods=window).cov(right)
        var = right.rolling(window, min_periods=window).var()
        return _safe_div(cov, var)
    raise ValueError(f"Unknown pair rolling op: {fn}")


def _compare(fn: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if fn == "gt":
        return (left > right).astype(float)
    if fn == "lt":
        return (left < right).astype(float)
    if fn == "cross_above":
        return ((left > right) & (left.shift(1) <= right.shift(1))).astype(float)
    if fn == "cross_below":
        return ((left < right) & (left.shift(1) >= right.shift(1))).astype(float)
    raise ValueError(f"Unknown compare op: {fn}")


def _rule_signal(expr: pd.Series, low: pd.Series, high: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(expr < low, 1.0, np.where(expr > high, -1.0, 0.0)),
        index=expr.index,
    )


def _binary(op: str, left: pd.Series, right: pd.Series) -> pd.Series:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return _safe_div(left, right)
    raise ValueError(f"Unknown binary op: {op}")


def _safe_div(left, right) -> pd.Series:
    left_s = pd.Series(left)
    right_s = pd.Series(right).where(pd.Series(right).abs() > 1e-12)
    return (left_s / right_s).replace([np.inf, -np.inf], np.nan)


def _parse_fn(formula: str) -> tuple[str | None, str | None]:
    for fn in UNARY_OPS + ROLLING_OPS + PAIR_TS_OPS + COMPARE_OPS + CONDITIONAL_OPS:
        prefix = fn + "("
        if not formula.startswith(prefix):
            continue
        depth = 0
        for idx, ch in enumerate(formula[len(fn):]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = len(fn) + idx
                    if end == len(formula) - 1:
                        return fn, formula[len(prefix):end]
                    break
    return None, None


def _split_window_args(args: str | None) -> tuple[str, int] | None:
    if args is None:
        return None
    parts = _split_top_level(args, ",")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    return parts[0], int(parts[1])


def _split_pair_window_args(args: str | None) -> tuple[str, str, int] | None:
    if args is None:
        return None
    parts = _split_top_level(args, ",")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return parts[0], parts[1], int(parts[2])


def _split_expr_args(args: str | None, expected: int) -> list[str] | None:
    if args is None:
        return None
    parts = _split_top_level(args, ",")
    return parts if len(parts) == expected else None


def _split_top_level(text: str, sep: str) -> list[str]:
    depth = 0
    start = 0
    parts = []
    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def _split_binary(formula: str) -> tuple[str | None, str | None, str | None]:
    for op_tok in (" + ", " - ", " * ", " / "):
        found = _find_op(formula, op_tok)
        if found is not None:
            return op_tok.strip(), found[0], found[1]
    return None, None, None


def _find_op(formula: str, op_tok: str) -> tuple[str, str] | None:
    depth = 0
    last = -1
    for idx, ch in enumerate(formula):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and formula[idx:].startswith(op_tok):
            last = idx
    if last == -1:
        return None
    return formula[:last].strip(), formula[last + len(op_tok):].strip()


def _balanced(formula: str) -> bool:
    if not (formula.startswith("(") and formula.endswith(")")):
        return False
    depth = 0
    for idx, ch in enumerate(formula):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and idx < len(formula) - 1:
                return False
    return depth == 0


def _sanitize_feature_name(formula: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", formula)[:180]


def _format_number(value: float) -> str:
    text = f"{float(value):.6g}"
    return "0" if text == "-0" else text

