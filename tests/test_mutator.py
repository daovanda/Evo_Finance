import unittest
from unittest.mock import patch

import pandas as pd

from mutator.domain import Domain
from mutator.gene import Gene, Individual
from mutator.mutator import Mutator


class NoopMutator(Mutator):
    def _c1(self, ind: Individual, train_df: pd.DataFrame) -> None:
        return None


class NoFallbackMutator(Mutator):
    def _c3(self, ind: Individual, train_df: pd.DataFrame) -> None:
        return None


class RejectingDomain(Domain):
    def __init__(self, formulas=None):
        super().__init__()
        self._formulas = list(formulas or [])
        self.try_add_calls = 0

    def try_add(self, gene: Gene, train_df: pd.DataFrame, force: bool = False) -> bool:
        self.try_add_calls += 1
        return False


class SequentialDomain(Domain):
    def __init__(self, formulas):
        super().__init__()
        self._formulas = list(formulas)
        self._pos = 0

    def random_gene(self, rng) -> Gene:
        formula = self._formulas[self._pos % len(self._formulas)]
        self._pos += 1
        return Gene(formula)


def _sample_train_df(periods=40):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=periods, freq="B"), ["AAA"]],
        names=["date", "ticker"],
    )
    seq = pd.Series(range(len(idx)), index=idx, dtype=float)
    df = pd.DataFrame(index=idx)
    df["open"] = 10.0 + seq
    df["high"] = df["open"] + 1.0
    df["low"] = df["open"] - 1.0
    df["close"] = df["open"] + 0.5
    df["volume"] = 1000.0 + seq
    return df


class MutatorTests(unittest.TestCase):
    def test_mutate_rejects_unchanged_child(self):
        domain = Domain()
        domain.seed()
        mutator = NoopMutator(domain, seed=7)
        parent = Individual(genes=[Gene("ret(close_1, 20)")])

        with patch.dict(
            "mutator.mutator.MUTATOR_PROBS",
            {"c1": 1.0, "c2": 0.0, "c3": 0.0},
        ):
            with self.assertRaisesRegex(ValueError, "unchanged individual"):
                mutator.mutate(parent, pd.DataFrame())

        self.assertEqual(parent.formulas, ["ret(close_1, 20)"])

    def test_try_replace_rejects_new_gene_when_domain_rejects_it(self):
        domain = RejectingDomain()
        mutator = Mutator(domain, seed=7)
        old_gene = Gene("ret(close_1, 3)")
        new_gene = Gene("ret(close_1, 1)")
        individual = Individual(genes=[old_gene])

        replaced = mutator._try_replace(
            individual,
            old_gene,
            new_gene,
            _sample_train_df(),
            "test",
        )

        self.assertFalse(replaced)
        self.assertEqual(individual.formulas, [old_gene.formula])
        self.assertEqual(domain.try_add_calls, 1)

    def test_c2_rejects_new_gene_when_domain_rejects_it(self):
        domain = RejectingDomain()
        mutator = NoFallbackMutator(domain, seed=7)
        old_gene = Gene("ret(close_1, 3)")
        individual = Individual(genes=[old_gene])

        with (
            patch("mutator.mutator.WINDOWS", [1]),
            patch("mutator.mutator.MAX_RETRY", 1),
        ):
            mutator._c2(individual, _sample_train_df())

        self.assertEqual(individual.formulas, [old_gene.formula])
        self.assertEqual(domain.try_add_calls, 1)

    def test_try_replace_allows_gene_already_in_domain(self):
        old_gene = Gene("ret(close_1, 3)")
        new_gene = Gene("ret(close_1, 1)")
        domain = RejectingDomain(formulas=[new_gene.formula])
        mutator = Mutator(domain, seed=7)
        individual = Individual(genes=[old_gene])

        replaced = mutator._try_replace(
            individual,
            old_gene,
            new_gene,
            _sample_train_df(),
            "test",
        )

        self.assertTrue(replaced)
        self.assertEqual(individual.formulas, [new_gene.formula])
        self.assertEqual(domain.try_add_calls, 0)

    def test_random_domain_gene_skips_unselectable_formulas(self):
        domain = SequentialDomain(
            [
                "close_1",
                "sector_code()",
                "const(1)",
                "vol_scale(close_1, 20)",
                "ret(close_1, 1)",
            ]
        )
        mutator = Mutator(domain, seed=7)

        gene = mutator._random_selectable_domain_gene()

        self.assertEqual(gene.formula, "ret(close_1, 1)")


if __name__ == "__main__":
    unittest.main()
