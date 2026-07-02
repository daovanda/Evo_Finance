"""Individuals, archive, and mutation for crypto feature evolution."""

from __future__ import annotations

import json
import re
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


_WINDOW_ARG_RE = re.compile(r",\s*(\d+)(?=\))")
_WINDOW_SUFFIX_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*_(\d+)\b")


@dataclass
class CryptoIndividual:
    features: list[str]
    generation: int = 0
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

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
            metrics, final_eval, final_val_metrics, test_metrics = _split_display_metrics(
                individual.metrics
            )
            rows.append(
                {
                    "rank": rank,
                    "score": individual.score,
                    "n_features": len(individual.features),
                    "generation": individual.generation,
                    "metrics": metrics,
                    "final_eval": final_eval,
                    "final_val_metrics": final_val_metrics,
                    "test_metrics": test_metrics,
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
            metrics = _flatten_loaded_metrics(row)
            individual = CryptoIndividual(
                features=list(row.get("features", [])),
                generation=int(row.get("generation", 0) or 0),
                score=float(row.get("score", float("-inf"))),
                metrics=metrics,
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
        if len(self.feature_pool) < config.FEATURE_MIN:
            raise ValueError(
                "CryptoMutator feature_pool is smaller than FEATURE_MIN."
            )
        selected_idx = self.rng.choice(
            len(self.feature_pool),
            size=config.FEATURE_MIN,
            replace=False,
        )
        selected = [self.feature_pool[int(idx)] for idx in selected_idx]
        return CryptoIndividual(features=selected)

    def mutate(self, individual: CryptoIndividual) -> CryptoIndividual:
        child = individual.clone()
        strategy = str(
            self.rng.choice(
                ["c1", "c2", "c3"],
                p=[
                    config.MUTATOR_PROBS["c1"],
                    config.MUTATOR_PROBS["c2"],
                    config.MUTATOR_PROBS["c3"],
                ],
            )
        )

        if strategy == "c1":
            changed = self._c1(child)
            if not changed:
                self._c3(child)
        elif strategy == "c2":
            changed = self._c2(child)
            if not changed:
                self._c3(child)
        else:
            self._c3(child)
        return child

    def _c1(self, individual: CryptoIndividual) -> bool:
        action = str(self.rng.choice(["add", "remove"]))
        if len(individual.features) >= config.FEATURE_MAX:
            action = "remove"
        if len(individual.features) <= config.FEATURE_MIN:
            action = "add"

        if action == "remove":
            if len(individual.features) <= config.FEATURE_MIN:
                return False
            remove_idx = int(self.rng.integers(len(individual.features)))
            individual.features.pop(remove_idx)
            return True

        candidate = self._domain_candidate_not_in(individual.features)
        if candidate is None:
            return False
        individual.features.append(candidate)
        return True

    def _c2(self, individual: CryptoIndividual) -> bool:
        if not individual.features:
            return False
        for _ in range(config.MAX_RETRY):
            old_idx = int(self.rng.integers(len(individual.features)))
            old_formula = individual.features[old_idx]
            candidate = self._change_window_formula(old_formula)
            if candidate is None:
                continue
            if self._try_replace(individual, old_idx, candidate):
                return True
        return False

    def _c3(self, individual: CryptoIndividual) -> bool:
        if not individual.features:
            return False
        for _ in range(config.MAX_RETRY):
            old_idx = int(self.rng.integers(len(individual.features)))
            base = [
                feature for idx, feature in enumerate(individual.features)
                if idx != old_idx
            ]
            candidate = self._generated_candidate(
                base,
                anchor=individual.features[old_idx],
            )
            if candidate is None:
                continue
            individual.features[old_idx] = candidate
            return True
        return False

    def _try_replace(
        self,
        individual: CryptoIndividual,
        old_idx: int,
        candidate: str,
    ) -> bool:
        old_formula = individual.features[old_idx]
        if candidate == old_formula:
            return False
        base = [
            feature for idx, feature in enumerate(individual.features)
            if idx != old_idx
        ]
        if candidate in base:
            return False
        if not self._admit_candidate(candidate, base):
            return False
        individual.features[old_idx] = candidate
        return True

    def _domain_candidate_not_in(self, current: list[str]) -> str | None:
        current_set = set(current)
        for _ in range(config.MAX_RETRY):
            candidate = str(self.rng.choice(self.domain_pool))
            if candidate in current_set:
                continue
            if self._admit_candidate(candidate, current):
                return candidate
        return None

    def _change_window_formula(self, formula: str) -> str | None:
        spans = _window_spans(formula)
        if not spans:
            return None
        start, end, old_window = spans[int(self.rng.integers(len(spans)))]
        choices = [int(window) for window in config.WINDOWS if int(window) != old_window]
        if not choices:
            return None
        new_window = int(self.rng.choice(choices))
        return f"{formula[:start]}{new_window}{formula[end:]}"

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


def _window_spans(formula: str) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for match in _WINDOW_ARG_RE.finditer(formula):
        start, end = match.span(1)
        seen.add((start, end))
        spans.append((start, end, int(match.group(1))))
    for match in _WINDOW_SUFFIX_RE.finditer(formula):
        start, end = match.span(1)
        if (start, end) in seen:
            continue
        seen.add((start, end))
        spans.append((start, end, int(match.group(1))))
    return spans


def _split_display_metrics(
    metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    wf_metrics: dict[str, Any] = {}
    final_eval: dict[str, Any] = {}
    final_val_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}

    for key, value in metrics.items():
        if key == "final_n_horizon_scores":
            final_eval["n_horizon_scores"] = value
        elif key == "final_train_rows":
            final_eval["train_rows"] = value
        elif key == "final_val_rows":
            final_val_metrics["rows"] = value
        elif key == "final_test_rows":
            test_metrics["rows"] = value
        elif key == "final_val_overfit_gap":
            final_val_metrics["overfit_gap"] = value
        elif key == "final_test_overfit_gap":
            test_metrics["overfit_gap"] = value
        elif key.startswith("final_val_"):
            final_val_metrics[key.removeprefix("final_val_")] = value
        elif key.startswith("final_test_"):
            test_metrics[key.removeprefix("final_test_")] = value
        elif key.startswith("final_h"):
            _add_horizon_metric(final_val_metrics, test_metrics, key, value)
        else:
            wf_metrics[key] = value

    return wf_metrics, final_eval, final_val_metrics, test_metrics


def _add_horizon_metric(
    final_val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    match = re.match(r"^final_h(\d+)_(val|test)_(.+)$", key)
    if match is None:
        return
    horizon, split, metric_name = match.groups()
    target = final_val_metrics if split == "val" else test_metrics
    horizons = target.setdefault("horizons", {})
    horizon_metrics = horizons.setdefault(f"h{horizon}", {})
    horizon_metrics[metric_name] = value


def _flatten_loaded_metrics(row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics", {}))
    final_eval = row.get("final_eval") or {}
    final_val = row.get("final_val_metrics") or {}
    test = row.get("test_metrics") or {}

    if "n_horizon_scores" in final_eval:
        metrics["final_n_horizon_scores"] = final_eval["n_horizon_scores"]
    if "train_rows" in final_eval:
        metrics["final_train_rows"] = final_eval["train_rows"]
    _flatten_split_metrics(metrics, final_val, split="val")
    _flatten_split_metrics(metrics, test, split="test")
    return metrics


def _flatten_split_metrics(
    metrics: dict[str, Any],
    split_metrics: dict[str, Any],
    split: str,
) -> None:
    prefix = "final_val" if split == "val" else "final_test"
    for key, value in split_metrics.items():
        if key == "rows":
            metrics[f"{prefix}_rows"] = value
        elif key == "overfit_gap":
            metrics[f"{prefix}_overfit_gap"] = value
        elif key == "horizons":
            for horizon_key, horizon_metrics in dict(value).items():
                horizon = str(horizon_key).removeprefix("h")
                for metric_name, metric_value in dict(horizon_metrics).items():
                    metrics[f"final_h{horizon}_{split}_{metric_name}"] = metric_value
        else:
            metrics[f"{prefix}_{key}"] = value
