"""Individuals, archive, and mutation for crypto feature evolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto import config
from crypto.expression import (
    BINARY_OPS,
    COMPARE_OPS,
    CONSTANT_VALUES,
    PAIR_TS_OPS,
    ROLLING_OPS,
    UNARY_OPS,
    CryptoFeatureSpace,
    constant_formula,
)


@dataclass
class CryptoIndividual:
    features: list[str]
    generation: int = 0
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def clone(self) -> "CryptoIndividual":
        return CryptoIndividual(
            features=list(self.features),
            generation=self.generation + 1,
            score=self.score,
            metrics=dict(self.metrics),
        )


class CryptoArchive:
    def __init__(self, max_size: int = config.ARCHIVE_SIZE):
        self.max_size = int(max_size)
        self._entries: list[CryptoIndividual] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[CryptoIndividual]:
        return list(self._entries)

    @property
    def best(self) -> CryptoIndividual | None:
        return self._entries[0] if self._entries else None

    @property
    def worst_score(self) -> float:
        return float(self._entries[-1].score) if self._entries else float("-inf")

    def is_empty(self) -> bool:
        return not self._entries

    def random_individual(self, rng: np.random.Generator) -> CryptoIndividual:
        if not self._entries:
            raise ValueError("Cannot sample from empty archive.")
        return self._entries[int(rng.integers(len(self._entries)))]

    def try_add(self, individual: CryptoIndividual) -> bool:
        if individual.score is None or not np.isfinite(float(individual.score)):
            return False

        duplicate = self._duplicate_index(individual)
        if duplicate is not None:
            if float(individual.score) <= float(self._entries[duplicate].score):
                return False
            self._entries[duplicate] = individual
            self._sort()
            return True

        if len(self._entries) < self.max_size:
            self._entries.append(individual)
            self._sort()
            return True

        if float(individual.score) > self.worst_score:
            self._entries.pop()
            self._entries.append(individual)
            self._sort()
            return True
        return False

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for rank, individual in enumerate(self._entries, start=1):
            rows.append(
                {
                    "rank": rank,
                    "score": individual.score,
                    "n_features": len(individual.features),
                    "generation": individual.generation,
                    "metrics": individual.metrics,
                    "features": individual.features,
                }
            )
        return rows

    def save(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata or {},
            "entries": self.summary(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, max_size: int = config.ARCHIVE_SIZE) -> "CryptoArchive":
        archive = cls(max_size=max_size)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = payload.get("entries", payload if isinstance(payload, list) else [])
        for row in entries:
            individual = CryptoIndividual(
                features=list(row.get("features", [])),
                generation=int(row.get("generation", 0) or 0),
                score=float(row.get("score", float("-inf"))),
                metrics=dict(row.get("metrics", {})),
            )
            archive.try_add(individual)
        return archive

    def _duplicate_index(self, individual: CryptoIndividual) -> int | None:
        signature = _signature(individual.features)
        for idx, entry in enumerate(self._entries):
            if _signature(entry.features) == signature:
                return idx
        return None

    def _sort(self) -> None:
        self._entries.sort(key=lambda item: float(item.score), reverse=True)


class CryptoMutator:
    def __init__(
        self,
        feature_pool: list[str],
        feature_space: CryptoFeatureSpace | pd.DataFrame,
        train_index: pd.Index,
        seed: int = 42,
    ):
        self.feature_pool = list(dict.fromkeys(feature_pool))
        if not self.feature_pool:
            raise ValueError("CryptoMutator requires a non-empty feature pool.")
        self.feature_space = (
            feature_space
            if isinstance(feature_space, CryptoFeatureSpace)
            else CryptoFeatureSpace(feature_space, self.feature_pool)
        )
        self.train_index = train_index
        self.domain_pool = list(self.feature_pool)
        self.rng = np.random.default_rng(seed)

    def seed_individual(self) -> CryptoIndividual:
        preferred = [
            "ret_close_1",
            "ret_close_3",
            "logret_close_10",
            "ma_ratio_close_20",
            "volatility_20",
            "rsi_14",
            "range_pct",
            "buy_pressure_base",
            "volume_log_z_20",
            "trade_count_log_z_20",
            "taker_ratio_z_20",
            "ret_x_buy_pressure_20",
        ]
        selected = [name for name in preferred if name in self.feature_pool]
        if len(selected) < config.FEATURE_MIN:
            selected.extend(
                name for name in self.feature_pool
                if name not in selected
            )
        return CryptoIndividual(features=selected[: config.FEATURE_MIN])

    def mutate(self, individual: CryptoIndividual) -> CryptoIndividual:
        child = individual.clone()
        action = self.rng.choice(
            ["add", "remove", "replace", "transform"],
            p=[0.30, 0.12, 0.28, 0.30],
        )
        if len(child.features) <= config.FEATURE_MIN:
            action = "add"
        if len(child.features) >= config.FEATURE_MAX:
            action = "remove"

        if action == "remove":
            remove_idx = int(self.rng.integers(len(child.features)))
            child.features.pop(remove_idx)
            return child

        if action == "add":
            candidate = self._candidate_not_in(child.features)
            if candidate is not None:
                child.features.append(candidate)
            return child

        if action == "replace":
            old_idx = int(self.rng.integers(len(child.features)))
            base = [feature for idx, feature in enumerate(child.features) if idx != old_idx]
            candidate = self._candidate_not_in(base)
            if candidate is not None:
                child.features[old_idx] = candidate
            return child

        old_idx = int(self.rng.integers(len(child.features)))
        base = [feature for idx, feature in enumerate(child.features) if idx != old_idx]
        candidate = self._generated_candidate(base, anchor=child.features[old_idx])
        if candidate is not None:
            child.features[old_idx] = candidate
        return child

    def _candidate_not_in(self, current: list[str]) -> str | None:
        current_set = set(current)
        for _ in range(100):
            if self.rng.random() < 0.55:
                candidate = self._generated_candidate(current)
            else:
                candidate = str(self.rng.choice(self.domain_pool))
            if candidate is None:
                continue
            if candidate in current_set:
                continue
            if self._admit_candidate(candidate, current):
                return candidate
        fallback = [
            name for name in self.domain_pool
            if name not in current_set and self._admit_candidate(name, current)
        ]
        return str(self.rng.choice(fallback)) if fallback else None

    def _generated_candidate(
        self,
        current: list[str],
        anchor: str | None = None,
    ) -> str | None:
        mode = str(
            self.rng.choice(
                ["unary", "rolling", "binary", "pair_ts", "compare", "where", "rule"],
                p=[0.14, 0.24, 0.24, 0.10, 0.11, 0.09, 0.08],
            )
        )
        anchor_formula = anchor or self._random_operand(prefer_current=current)
        if anchor_formula is None:
            return None

        for _ in range(30):
            if mode == "unary":
                formula = f"{self.rng.choice(UNARY_OPS)}({anchor_formula})"
            elif mode == "rolling":
                op = str(self.rng.choice(ROLLING_OPS))
                window = int(self.rng.choice(config.WINDOWS))
                formula = f"{op}({anchor_formula}, {window})"
            elif mode == "binary":
                other = self._random_operand(prefer_current=current)
                if other is None:
                    continue
                op = str(self.rng.choice(BINARY_OPS))
                formula = f"({anchor_formula} {op} {other})"
            elif mode == "pair_ts":
                other = self._random_operand(prefer_current=current)
                if other is None:
                    continue
                op = str(self.rng.choice(PAIR_TS_OPS))
                window = int(self.rng.choice(config.WINDOWS))
                formula = f"{op}({anchor_formula}, {other}, {window})"
            elif mode == "compare":
                op = str(self.rng.choice(COMPARE_OPS))
                other = self._threshold_or_operand(anchor_formula)
                formula = f"{op}({anchor_formula}, {other})"
            elif mode == "where":
                op = str(self.rng.choice(["gt", "lt"]))
                threshold = self._threshold_or_operand(anchor_formula, prefer_constant=True)
                cond = f"{op}({anchor_formula}, {threshold})"
                true_value = constant_formula(1.0 if op == "gt" else -1.0)
                false_value = constant_formula(0.0)
                formula = f"where({cond}, {true_value}, {false_value})"
            else:
                low, high = self._rule_threshold_pair(anchor_formula)
                formula = f"rule_signal({anchor_formula}, {low}, {high})"

            if self._admit_candidate(formula, current):
                return formula
        return None

    def _random_operand(self, prefer_current: list[str] | None = None) -> str | None:
        pool: list[str] = []
        if prefer_current and self.rng.random() < 0.45:
            pool = list(prefer_current)
        if not pool:
            pool = self.domain_pool
        if not pool:
            return None
        return str(self.rng.choice(pool))

    def _threshold_or_operand(
        self,
        anchor: str,
        prefer_constant: bool = False,
    ) -> str:
        if prefer_constant or self.rng.random() < 0.70:
            return constant_formula(self._jitter_constant(float(self.rng.choice(CONSTANT_VALUES))))
        other = self._random_operand(prefer_current=[anchor])
        return other if other is not None else constant_formula(0.0)

    def _rule_threshold_pair(self, anchor: str) -> tuple[str, str]:
        lower_upper = [
            (-1.0, 1.0),
            (-0.5, 0.5),
            (-0.2, 0.2),
            (-0.05, 0.05),
            (0.2, 0.8),
            (30.0, 70.0),
        ]
        low, high = lower_upper[int(self.rng.integers(len(lower_upper)))]
        low = self._jitter_constant(low)
        high = self._jitter_constant(high)
        if low > high:
            low, high = high, low
        return constant_formula(low), constant_formula(high)

    def _jitter_constant(self, value: float) -> float:
        scale = max(abs(value) * 0.15, 0.005 if abs(value) < 0.1 else 0.05)
        new_value = value + float(self.rng.normal(0.0, scale))
        return 0.0 if abs(new_value) < 1e-9 else new_value

    def _admit_candidate(self, candidate: str, current: list[str]) -> bool:
        if candidate in current:
            return False
        quality = self.feature_space.quality(candidate, self.train_index)
        if not quality.ok:
            return False
        if not self._passes_corr_guard(candidate, current):
            return False
        if candidate not in self.domain_pool:
            self.domain_pool.append(candidate)
        return True

    def _passes_corr_guard(self, candidate: str, current: list[str]) -> bool:
        if not current:
            return True
        candidate_series = self.feature_space.evaluate(candidate).loc[self.train_index]
        for feature in current:
            corr = _spearman_abs_corr(
                candidate_series,
                self.feature_space.evaluate(feature).loc[self.train_index],
            )
            if corr >= config.FEATURE_CORR_THRESHOLD:
                return False
        return True


def _spearman_abs_corr(left: pd.Series, right: pd.Series) -> float:
    data = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        return 0.0
    if data.iloc[:, 0].nunique(dropna=True) < 2 or data.iloc[:, 1].nunique(dropna=True) < 2:
        return 0.0
    corr = data.iloc[:, 0].corr(data.iloc[:, 1], method="spearman")
    return abs(float(corr)) if np.isfinite(corr) else 0.0


def _signature(features: list[str]) -> tuple[str, ...]:
    return tuple(sorted({str(feature) for feature in features}))
