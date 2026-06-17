"""
Evo_Finance — Gene & Individual (v2)
──────────────────────────────────────
A Gene is an expression stored as a canonical formula string.
An Individual is an ordered list of Genes used as model features.

Formula examples
  "close_1"                          raw close (rolling mean w=1)
  "close_20"                         20-day rolling mean of close
  "std(close_1, 20)"                 20-day rolling std of close   ← NEW
  "max(close_1, 20)"                 20-day rolling max            ← NEW
  "min(low_1, 14)"                   14-day rolling min of low     ← NEW
  "shift(close_1, 5)"                close lagged 5 days           ← NEW
  "rank(close_10)"                   cross-sectional rank
  "zscore(volume_3)"                 cross-sectional z-score
  "(close_1 - shift(close_1, 20))"   20-day price change           ← NEW combo
  "rank((close_1 - shift(close_1, 20)) / shift(close_1, 20))"   momentum20 ← NEW
  "std(close_1, 20) / close_20"      normalised volatility         ← NEW combo
"""

from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Constants ────────────────────────────────────────────────────────────────

# Unary ops (no window)
SECTOR_CS_OPS = ("sector_rank", "sector_zscore", "sector_neutralize")

CS_UNARY_OPS = (
    "rank", "zscore", "abs", "signed_log",
    "sign", "clip", "pos_part", "neg_part",
    "winsorize", "neutralize",
) + SECTOR_CS_OPS

# Rolling time-series ops (require a window argument)
TS_ROLLING_OPS = (
    "std", "max", "min", "shift", "sum", "ema",
    "median", "q25", "q75", "iqr", "skew", "kurt",
    "ts_rank", "ts_zscore", "decay_linear", "slope",
    "days_since_rolling_high", "days_since_rolling_low",
)
PAIR_TS_OPS = ("ts_corr", "ts_cov", "ts_beta")

# Finance transforms applied to an existing expression (require a window).
FINANCE_TS_OPS = (
    "ret", "logret", "delta", "vol", "drawdown", "breakout", "ma_ratio",
    "bb_pos", "bb_width", "rsi", "vol_scale", "efficiency_ratio", "ulcer_index",
)

# Finance features computed directly from OHLCV context.
FINANCE_WINDOW_OPS = (
    "pos", "volume_ratio", "liquidity", "atr", "stoch", "obv", "mfi", "cmf",
    "adx", "cci", "willr", "keltner_pos", "keltner_width", "donchian_width",
    "vwap", "vwap_pos", "amihud", "parkinson_vol", "gk_vol", "rs_vol",
    "aroon_up", "aroon_down", "aroon_osc", "choppiness",
)
MARKET_WINDOW_OPS = (
    "market_ret", "market_vol", "market_drawdown", "market_ma_ratio",
    "market_rsi", "market_pos", "market_volume_ratio",
    "rel_ret", "rel_strength", "market_corr", "market_beta", "market_alpha",
    "idiosyncratic_vol", "up_capture", "down_capture",
)
BREADTH_NOARG_OPS = (
    "advance_count", "decline_count", "unchanged_count",
    "advance_ratio", "decline_ratio", "advance_decline_ratio",
    "advance_decline_spread", "advance_decline_net_pct", "cs_dispersion",
)
BREADTH_WINDOW_OPS = ("pct_above_ma", "breadth_momentum")
SECTOR_NOARG_OPS = (
    "sector_code", "sector_size",
    "sector_advance_count", "sector_decline_count", "sector_unchanged_count",
    "sector_advance_ratio", "sector_decline_ratio",
    "sector_advance_decline_ratio", "sector_advance_decline_spread",
    "sector_advance_decline_net_pct", "sector_dispersion",
)
SECTOR_WINDOW_OPS = (
    "sector_ret", "sector_vol", "sector_drawdown", "sector_ma_ratio",
    "sector_rsi", "sector_pos", "sector_volume_ratio",
    "rel_sector_ret", "sector_rel_strength", "sector_corr", "sector_beta",
    "sector_alpha", "sector_idiosyncratic_vol",
    "sector_up_capture", "sector_down_capture",
    "sector_pct_above_ma", "sector_breadth_momentum",
)
FINANCE_TWO_WINDOW_OPS = ("stoch_d",)
FINANCE_NOARG_OPS = (
    "body", "range", "gap", "upper_wick", "lower_wick", "dollar_volume",
    "signed_volume", "typical_price", "money_flow",
)
CUMULATIVE_TS_OPS = (
    "cummax", "cummin", "cumret", "expanding_drawdown", "expanding_runup",
    "days_since_high", "days_since_low", "cum_sum", "up_streak", "down_streak",
)
CUMULATIVE_NOARG_OPS = ("cum_obv", "cum_adl", "cum_pvt")
COMPARISON_OPS = ("gt", "lt", "cross_above", "cross_below")
CONDITIONAL_OPS = ("where", "rule_signal")
CONSTANT_VALUES = (
    -2.0, -1.0, -0.5, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5,
    1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0,
    30.0, 50.0, 70.0,
)
CONSTANT_FORMULAS = tuple(f"const({value:g})" for value in CONSTANT_VALUES)

# All unary ops visible to the mutator
ALL_UNARY_OPS = CS_UNARY_OPS + TS_ROLLING_OPS + FINANCE_TS_OPS + CUMULATIVE_TS_OPS

# Binary element-wise ops
BINARY_OPS = ("+", "-", "*", "/")
_CONST_RE = re.compile(r'const\((-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)')


# ─── Gene ─────────────────────────────────────────────────────────────────────

@dataclass
class Gene:
    formula: str
    history: List[str] = field(default_factory=list)

    # ── Construction helpers ──────────────────────────────────────────────────

    @staticmethod
    def raw(base: str, window: int) -> "Gene":
        """Gene.raw('close', 5) → formula='close_5' (rolling mean)."""
        return Gene(formula=f"{base}_{window}")

    @staticmethod
    def constant(value: float) -> "Gene":
        return Gene(formula=f"const({_format_number(value)})")

    @staticmethod
    def transform(gene: "Gene", op: str, window: int = 20) -> "Gene":
        """
        Apply a unary operator to a gene.

        Cross-sectional (no window): rank, zscore, abs, signed_log
            → op(gene.formula)

        Rolling time-series (needs window): std, max, min, shift
            → op(gene.formula, window)
        """
        if op in CS_UNARY_OPS:
            new_formula = f"{op}({gene.formula})"
            tag = f"cs_transform:{op}"
        elif op in CUMULATIVE_TS_OPS:
            new_formula = f"{op}({gene.formula})"
            tag = f"cumulative_transform:{op}"
        elif op in TS_ROLLING_OPS or op in FINANCE_TS_OPS:
            new_formula = f"{op}({gene.formula}, {window})"
            tag = f"window_transform:{op}_{window}"
        else:
            raise ValueError(f"Unknown op: {op!r}")

        return Gene(
            formula=new_formula,
            history=gene.history + [f"{tag}({gene.formula})"],
        )

    @staticmethod
    def combine(gene_x: "Gene", gene_y: "Gene", op: str) -> "Gene":
        """Binary combination: (gene_x OP gene_y)."""
        if op not in BINARY_OPS:
            raise ValueError(f"Unknown binary op: {op!r}")
        new_formula = f"({gene_x.formula} {op} {gene_y.formula})"
        return Gene(
            formula=new_formula,
            history=gene_x.history + [
                f"combine:{gene_x.formula}{op}{gene_y.formula}"
            ],
        )

    @staticmethod
    def compare(gene_x: "Gene", gene_y: "Gene", op: str) -> "Gene":
        if op not in COMPARISON_OPS:
            raise ValueError(f"Unknown comparison op: {op!r}")
        new_formula = f"{op}({gene_x.formula}, {gene_y.formula})"
        return Gene(
            formula=new_formula,
            history=gene_x.history + [
                f"compare:{op}({gene_x.formula},{gene_y.formula})"
            ],
        )

    @staticmethod
    def pair_ts(gene_x: "Gene", gene_y: "Gene", op: str, window: int) -> "Gene":
        if op not in PAIR_TS_OPS:
            raise ValueError(f"Unknown pair time-series op: {op!r}")
        new_formula = f"{op}({gene_x.formula}, {gene_y.formula}, {window})"
        return Gene(
            formula=new_formula,
            history=gene_x.history + [
                f"pair_ts:{op}_{window}({gene_x.formula},{gene_y.formula})"
            ],
        )

    @staticmethod
    def where(cond: "Gene", true_gene: "Gene", false_gene: "Gene") -> "Gene":
        new_formula = (
            f"where({cond.formula}, {true_gene.formula}, {false_gene.formula})"
        )
        return Gene(
            formula=new_formula,
            history=cond.history + [f"where:{new_formula}"],
        )

    @staticmethod
    def rule_signal(expr: "Gene", low: "Gene", high: "Gene") -> "Gene":
        new_formula = f"rule_signal({expr.formula}, {low.formula}, {high.formula})"
        return Gene(
            formula=new_formula,
            history=expr.history + [f"rule_signal:{new_formula}"],
        )

    @staticmethod
    def mutate_constant(gene: "Gene", rng) -> Optional["Gene"]:
        matches = list(_CONST_RE.finditer(gene.formula))
        if not matches:
            return None

        match = matches[int(rng.integers(len(matches)))]
        old_value = float(match.group(1))
        scale = max(abs(old_value) * 0.15, 0.005 if abs(old_value) < 0.1 else 0.05)
        new_value = old_value + float(rng.normal(0.0, scale))
        if abs(new_value) < 1e-9:
            new_value = 0.0
        new_const = f"const({_format_number(new_value)})"
        new_formula = gene.formula[:match.start()] + new_const + gene.formula[match.end():]
        return Gene(
            formula=new_formula,
            history=gene.history + [
                f"mutate_const:{old_value:g}->{_format_number(new_value)}"
            ],
        )

    @staticmethod
    def change_window(gene: "Gene", new_window: int) -> "Gene":
        """
        Change the outermost window of a gene.

        - Simple terminal 'close_5' → 'close_{new_window}'
        - Rolling fn 'std(close_1, 14)' → 'std(close_1, {new_window})'
        - Complex expression → wrap as '(expr)_w{new_window}' (rolling mean)
        """
        import re

        # Case 1: simple terminal  base_N
        parts = gene.formula.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            new_formula = f"{parts[0]}_{new_window}"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}→{new_formula}"],
            )

        # Case 2: rolling fn  op(inner, N)
        window_fns = "|".join(TS_ROLLING_OPS + FINANCE_TS_OPS)
        m = re.match(rf'^({window_fns})\((.+),\s*\d+\)$', gene.formula)
        if m:
            fn, inner = m.group(1), m.group(2)
            new_formula = f"{fn}({inner}, {new_window})"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}→{new_formula}"],
            )

        # Case 2b: already window-tagged  (expr)_wN → replace N
        context_fns = "|".join(
            FINANCE_WINDOW_OPS + MARKET_WINDOW_OPS
            + BREADTH_WINDOW_OPS + SECTOR_WINDOW_OPS
        )
        m_ctx = re.match(rf'^({context_fns})\(\s*\d+\s*\)$', gene.formula)
        if m_ctx:
            fn = m_ctx.group(1)
            new_formula = f"{fn}({new_window})"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}â†’{new_formula}"],
            )

        two_window_fns = "|".join(FINANCE_TWO_WINDOW_OPS)
        m_two_ctx = re.match(
            rf'^({two_window_fns})\(\s*\d+\s*,\s*(\d+)\s*\)$',
            gene.formula,
        )
        if m_two_ctx:
            fn, smooth_window = m_two_ctx.group(1), m_two_ctx.group(2)
            new_formula = f"{fn}({new_window}, {smooth_window})"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}->{new_formula}"],
            )

        pair_window_fns = "|".join(PAIR_TS_OPS)
        m_pair = re.match(rf'^({pair_window_fns})\((.+),\s*\d+\)$', gene.formula)
        if m_pair:
            fn, inner = m_pair.group(1), m_pair.group(2)
            new_formula = f"{fn}({inner}, {new_window})"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}->{new_formula}"],
            )

        m2 = re.match(r'^\((.+)\)_w\d+$', gene.formula)
        if m2:
            inner = m2.group(1)
            new_formula = f"({inner})_w{new_window}"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}→{new_formula}"],
            )

        # Case 3: complex expression → wrap with rolling mean tag
        # Strip existing outer parens to avoid double-wrapping: ((expr))_wN
        inner = gene.formula
        if inner.startswith("(") and inner.endswith(")") and _is_fully_wrapped(inner):
            inner = inner[1:-1]
        new_formula = f"({inner})_w{new_window}"
        return Gene(
            formula=new_formula,
            history=gene.history + [f"window:{gene.formula}→{new_formula}"],
        )


def _is_fully_wrapped(formula: str) -> bool:
    """True if the outermost ( ) enclose the entire expression."""
    if not (formula.startswith("(") and formula.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(formula):
        if ch == "(": depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and i < len(formula) - 1:
                return False
    return depth == 0


def _format_number(value: float) -> str:
    if abs(value) < 1e-12:
        value = 0.0
    return f"{value:.6g}"



# ─── Individual ───────────────────────────────────────────────────────────────

@dataclass
class Individual:
    genes: List[Gene]
    score: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    @staticmethod
    def seed(bases=("open", "high", "close", "low", "volume"),
             window: int = 1) -> "Individual":
        """Seed individual: raw OHLCV at default window."""
        genes = [Gene.raw(b, window) for b in bases]
        return Individual(genes=genes, generation=0)

    def clone(self) -> "Individual":
        return Individual(
            genes=[copy.deepcopy(g) for g in self.genes],
            score=None,
            metrics={},
            generation=self.generation + 1,
        )

    def add_gene(self, gene: Gene) -> None:
        self.genes.append(gene)

    def remove_gene(self, gene: Gene) -> None:
        """Remove by formula match (ignores history differences)."""
        for i, g in enumerate(self.genes):
            if g.formula == gene.formula:
                del self.genes[i]
                return
        raise ValueError(f"Gene {gene.formula!r} not found in individual")

    def replace_gene(self, old: Gene, new: Gene) -> None:
        """Replace by formula match (ignores history differences)."""
        for i, g in enumerate(self.genes):
            if g.formula == old.formula:
                self.genes[i] = new
                return
        raise ValueError(f"Gene {old.formula!r} not found in individual")

    @property
    def formulas(self) -> List[str]:
        return [g.formula for g in self.genes]

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        genes_str = ", ".join(self.formulas)
        score_str = f"{self.score:.4f}" if self.score is not None else "None"
        return f"Individual(score={score_str}, genes=[{genes_str}])"
