"""
Evo_Finance — Gene & Individual
────────────────────────────────
A Gene is an expression stored as a formula string.
An Individual is an ordered list of Genes.

Formula string examples
  "close_1"                       raw close, window=1
  "open_5"                        open, rolling mean window=5
  "rank(close_10)"
  "zscore(volume_3)"
  "abs(high_1)"
  "signed_log(low_14)"
  "(close_10 + volume_5)"
  "rank((close_10 + volume_5))"   nested — allowed
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ─── Gene ─────────────────────────────────────────────────────────────────────

@dataclass
class Gene:
    formula: str                          # canonical expression string
    history: List[str] = field(default_factory=list)

    # ---- construction helpers ------------------------------------------------

    @staticmethod
    def raw(base: str, window: int) -> "Gene":
        """e.g. Gene.raw('close', 5)  →  formula='close_5'"""
        return Gene(formula=f"{base}_{window}")

    @staticmethod
    def transform(gene: "Gene", op: str) -> "Gene":
        """Unary transforms: rank / zscore / abs / signed_log"""
        assert op in ("rank", "zscore", "abs", "signed_log"), f"Unknown op: {op}"
        new_formula = f"{op}({gene.formula})"
        new_gene = Gene(formula=new_formula,
                        history=gene.history + [f"transform:{op}({gene.formula})"])
        return new_gene

    @staticmethod
    def combine(gene_x: "Gene", gene_y: "Gene", op: str) -> "Gene":
        """Binary ops: +, -, *, /"""
        assert op in ("+", "-", "*", "/"), f"Unknown op: {op}"
        new_formula = f"({gene_x.formula} {op} {gene_y.formula})"
        new_gene = Gene(
            formula=new_formula,
            history=gene_x.history + [
                f"combine:{gene_x.formula}{op}{gene_y.formula}"
            ],
        )
        return new_gene

    @staticmethod
    def change_window(gene: "Gene", new_window: int) -> "Gene":
        """
        Replace the outermost window suffix if the gene is a raw feature
        like 'close_5'.  For complex expressions the whole string is kept and
        a '_w{new_window}' tag is appended so callers can handle it.
        """
        parts = gene.formula.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            new_formula = f"{parts[0]}_{new_window}"
        else:
            # complex formula — tag it; evaluator should handle
            new_formula = f"({gene.formula})_w{new_window}"
        new_gene = Gene(
            formula=new_formula,
            history=gene.history + [
                f"window_change:{gene.formula}→{new_formula}"
            ],
        )
        return new_gene

    # ---- utils ---------------------------------------------------------------

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
    score: Optional[float] = None                 # set after fitness eval
    metrics: Dict[str, float] = field(default_factory=dict)
    generation: int = 0                           # how many mutations from seed

    # ---- construction --------------------------------------------------------

    @staticmethod
    def seed(bases=("open", "high", "close", "low", "volume"),
             window: int = 1) -> "Individual":
        """Initial individual: raw OHLCV at default window."""
        genes = [Gene.raw(b, window) for b in bases]
        return Individual(genes=genes, generation=0)

    # ---- mutation support ----------------------------------------------------

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
        self.genes.remove(gene)

    def replace_gene(self, old: Gene, new: Gene) -> None:
        idx = self.genes.index(old)
        self.genes[idx] = new

    # ---- utils ---------------------------------------------------------------

    @property
    def formulas(self) -> List[str]:
        return [g.formula for g in self.genes]

    def __len__(self):
        return len(self.genes)

    def __repr__(self):
        genes_str = ", ".join(self.formulas)
        return f"Individual(score={self.score:.4f}, genes=[{genes_str}])" \
               if self.score is not None \
               else f"Individual(score=None, genes=[{genes_str}])"