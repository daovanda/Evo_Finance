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
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ─── Constants ────────────────────────────────────────────────────────────────

# Unary cross-sectional ops (no window)
CS_UNARY_OPS = ("rank", "zscore", "abs", "signed_log")

# Rolling time-series ops (require a window argument)
TS_ROLLING_OPS = ("std", "max", "min", "shift")

# All unary ops visible to the mutator
ALL_UNARY_OPS = CS_UNARY_OPS + TS_ROLLING_OPS

# Binary element-wise ops
BINARY_OPS = ("+", "-", "*", "/")


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
        elif op in TS_ROLLING_OPS:
            new_formula = f"{op}({gene.formula}, {window})"
            tag = f"ts_transform:{op}_{window}"
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
        m = re.match(r'^(std|max|min|shift)\((.+),\s*\d+\)$', gene.formula)
        if m:
            fn, inner = m.group(1), m.group(2)
            new_formula = f"{fn}({inner}, {new_window})"
            return Gene(
                formula=new_formula,
                history=gene.history + [f"window:{gene.formula}→{new_formula}"],
            )

        # Case 2b: already window-tagged  (expr)_wN → replace N
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

    # ── Utils ─────────────────────────────────────────────────────────────────

    def __eq__(self, other):
        return isinstance(other, Gene) and self.formula == other.formula

    def __hash__(self):
        return hash(self.formula)

    def __repr__(self):
        return f"Gene({self.formula!r})"


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