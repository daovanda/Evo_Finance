"""
Evo_Finance — Domain
─────────────────────
The Domain is the ever-growing pool of formula strings that the Mutator
can draw from. A new formula is admitted only if its absolute Spearman rank
correlation with every existing formula in the domain is < CORR_THRESHOLD
(computed on the training set).

Thread-safety: single-process assumed; no locking needed.
"""

from __future__ import annotations
import logging
import time
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    CORR_THRESHOLD,
    FEATURE_MAX_DOMINANT_VALUE_RATIO,
    FEATURE_MIN_VALID_RATIO,
    WINDOWS,
)
from mutator.gene import (
    Gene, BREADTH_NOARG_OPS, BREADTH_WINDOW_OPS,
    CONSTANT_FORMULAS, CUMULATIVE_NOARG_OPS, CUMULATIVE_TS_OPS,
    FINANCE_NOARG_OPS, FINANCE_TS_OPS, FINANCE_TWO_WINDOW_OPS,
    FINANCE_WINDOW_OPS, MARKET_WINDOW_OPS, PAIR_TS_OPS,
    SECTOR_CS_OPS, SECTOR_NOARG_OPS, SECTOR_WINDOW_OPS, TS_ROLLING_OPS,
)
from mutator.evaluator import evaluate, has_division_by_zero
from mutator.formula_guard import (
    BLOCKED_EVOLUTION_PRIMITIVES,
    const_threshold_violation,
    raw_scale_violation,
)

logger = logging.getLogger(__name__)

# Raw base columns (order preserved)
_RAW_BASES = ["open", "high", "close", "low", "volume"]

class Domain:
    """
    Stores formula strings (not computed arrays).
    Computed arrays are materialised on-demand for correlation checks.
    """

    def __init__(self):
        self._formulas: List[str] = []
        # Cache: formula -> computed Series for the current train context.
        self._cache: dict[str, pd.Series] = {}
        self._cache_context_key: tuple | None = None

    # ── Initialisation ────────────────────────────────────────────────────────

    def seed(self, window: int = 1) -> List[Gene]:
        """
        Populate the selectable domain with guard-safe finance primitives.

        Raw OHLCV anchor genes are still returned for backwards compatibility,
        but they are not inserted into ``self._formulas``.  They are raw scale
        and should only appear inside normalized primitives such as returns,
        ratios, candlestick percentages, or market-relative transforms.
        """
        genes = [Gene.raw(b, window) for b in _RAW_BASES]

        finance_formulas = []
        skipped_unsafe = 0
        for formula in _finance_seed_formulas():
            violation = const_threshold_violation(formula) or raw_scale_violation(formula)
            if violation is not None:
                skipped_unsafe += 1
                logger.debug("Domain.seed: skip unsafe formula %r - %s", formula, violation)
                continue
            finance_formulas.append(formula)
        for formula in finance_formulas:
            if formula not in self._formulas:
                self._formulas.append(formula)

        logger.info(
            "Domain seeded with %d raw anchors excluded + %d guard-safe finance primitives%s.",
            len(genes),
            len(finance_formulas),
            f" ({skipped_unsafe} unsafe skipped)" if skipped_unsafe else "",
        )
        return genes

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def formulas(self) -> List[str]:
        return list(self._formulas)

    def __len__(self):
        return len(self._formulas)

    def random_gene(self, rng: np.random.Generator) -> Gene:
        """Draw a uniformly random Gene from the domain."""
        formula = rng.choice(self._formulas)
        return Gene(formula=str(formula))

    def try_add(
        self,
        gene: Gene,
        train_df: pd.DataFrame,
        force: bool = False,
    ) -> bool:
        """
        Attempt to add *gene* to the domain.

        A gene is added only if:
          - Its formula is not already in the domain, AND
          - Its absolute correlation with every existing domain member is
            below CORR_THRESHOLD (evaluated on train_df).

        Parameters
        ----------
        gene      : Gene to test.
        train_df  : Training DataFrame (features must be computable).
        force     : If True, skip correlation check (used for seed genes).

        Returns True if the gene was added.
        """
        if gene.formula in self._formulas:
            return False

        violation = const_threshold_violation(gene.formula)
        if violation is not None:
            logger.debug(
                "Domain.try_add: reject unsafe const threshold %r - %s",
                gene.formula,
                violation,
            )
            return False
        violation = raw_scale_violation(gene.formula)
        if violation is not None:
            logger.debug(
                "Domain.try_add: reject raw-scale feature %r - %s",
                gene.formula,
                violation,
            )
            return False

        if not force:
            try:
                if has_division_by_zero(gene.formula, train_df):
                    logger.debug(
                        "Domain.try_add: reject division-by-zero gene %r",
                        gene.formula,
                    )
                    return False
            except Exception as exc:
                logger.warning(
                    "Domain.try_add: division check failed for %r — %s",
                    gene.formula, exc,
                )
                return False

            try:
                full_series = _evaluate_clean(gene.formula, train_df)
            except Exception as exc:
                logger.warning("Domain.try_add: cannot evaluate %r — %s",
                               gene.formula, exc)
                return False

            quality_violation = _feature_quality_violation(
                gene.formula,
                full_series,
                train_df,
            )
            if quality_violation is not None:
                logger.debug(
                    "Domain.try_add: reject low-quality feature %r - %s",
                    gene.formula,
                    quality_violation,
                )
                return False
            new_series = full_series.dropna()

            for existing_formula in self._formulas:
                ex_series = self._cache.get(existing_formula)
                if ex_series is None:
                    try:
                        ex_series = self._compute(existing_formula, train_df)
                    except Exception:
                        continue
                corr = _safe_corr(new_series, ex_series)
                if abs(corr) >= CORR_THRESHOLD:
                    return False

            # passed — cache it
            self._cache[gene.formula] = new_series

        self._formulas.append(gene.formula)
        logger.debug("Domain ← %r  (size=%d)", gene.formula, len(self._formulas))
        return True

    def precompute(
        self,
        train_df: pd.DataFrame,
        progress_every: int = 50,
        slow_sec: float = 1.0,
    ) -> None:
        """Materialise all domain formulas on train_df with progress logging."""
        self._ensure_cache_context(train_df)
        total = len(self._formulas)
        start = time.perf_counter()
        computed = skipped = failed = 0
        logger.info(
            "Domain.precompute: start %d formulas on %d rows.",
            total,
            len(train_df),
        )

        for i, f in enumerate(self._formulas, 1):
            t0 = time.perf_counter()
            if f not in self._cache:
                try:
                    self._cache[f] = self._compute(f, train_df)
                    computed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning("Domain.precompute: %r failed — %s", f, exc)

            else:
                skipped += 1

            elapsed_formula = time.perf_counter() - t0
            if slow_sec > 0 and elapsed_formula >= slow_sec:
                logger.info(
                    "Domain.precompute: slow %.2fs | %s",
                    elapsed_formula,
                    f,
                )

            if progress_every > 0 and i % progress_every == 0:
                logger.info(
                    "Domain.precompute: %d/%d | computed=%d skipped=%d "
                    "failed=%d | elapsed=%.1fs",
                    i,
                    total,
                    computed,
                    skipped,
                    failed,
                    time.perf_counter() - start,
                )

        logger.info(
            "Domain.precompute: done | computed=%d skipped=%d failed=%d "
            "elapsed=%.1fs",
            computed,
            skipped,
            failed,
            time.perf_counter() - start,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, formula: str, df: pd.DataFrame) -> pd.Series:
        self._ensure_cache_context(df)
        if formula in self._cache:
            return self._cache[formula]
        series = _evaluate_clean(formula, df).dropna()
        self._cache[formula] = series
        return series

    def _ensure_cache_context(self, df: pd.DataFrame) -> None:
        key = _dataframe_cache_key(df)
        if self._cache_context_key == key:
            return
        if self._cache:
            logger.debug(
                "Domain cache context changed; clearing %d cached formulas.",
                len(self._cache),
            )
            self._cache.clear()
        self._cache_context_key = key


# ─── Utility ──────────────────────────────────────────────────────────────────

def _dataframe_cache_key(df: pd.DataFrame) -> tuple:
    if len(df.index) == 0:
        first = last = None
    else:
        first = df.index[0]
        last = df.index[-1]
    return (
        id(df),
        id(df.index),
        len(df),
        tuple(df.index.names),
        tuple(df.columns),
        first,
        last,
    )


def _finance_seed_formulas() -> List[str]:
    formulas: List[str] = []
    formulas.extend(CONSTANT_FORMULAS)

    for op in FINANCE_NOARG_OPS:
        base = f"{op}()"
        formulas.append(base)
        formulas.extend(f"({base})_w{w}" for w in WINDOWS)

    for op in FINANCE_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)

    formulas.extend(
        f"{op}()"
        for op in BREADTH_NOARG_OPS
        if op not in BLOCKED_EVOLUTION_PRIMITIVES
    )
    for op in BREADTH_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)

    for op in SECTOR_NOARG_OPS:
        if op in BLOCKED_EVOLUTION_PRIMITIVES:
            continue
        formulas.append(f"{op}()")
    for op in SECTOR_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)
    for op in SECTOR_CS_OPS:
        formulas.extend(
            f"{op}({expr})"
            for expr in (
                "ret(close_1, 1)", "volume_ratio(20)",
                "rsi(close_1, 14)", "rel_sector_ret(20)",
            )
        )

    market_bases = [
        "market_open_1", "market_high_1", "market_low_1",
        "market_close_1", "market_volume_1",
    ]
    formulas.extend(market_bases)
    for op in MARKET_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)

    for op in FINANCE_TWO_WINDOW_OPS:
        if op == "stoch_d":
            formulas.extend(f"{op}({w}, {smooth})" for w in WINDOWS for smooth in (3, 5))

    close_base = "close_1"
    for w in WINDOWS:
        formulas.extend(f"{op}({close_base}, {w})" for op in FINANCE_TS_OPS)
        formulas.extend(
            f"{op}({close_base}, {w})"
            for op in TS_ROLLING_OPS
            if op in ("sum", "ema")
        )
        formulas.append(f"ema(volume_1, {w})")
        formulas.append(f"sum(volume_1, {w})")
        for op in (
            "std", "max", "min", "shift",
            "median", "q25", "q75", "iqr", "skew", "kurt",
            "ts_rank", "ts_zscore", "decay_linear", "slope",
            "days_since_rolling_high", "days_since_rolling_low",
        ):
            formulas.append(f"{op}({close_base}, {w})")
        for op in (
            "std", "max", "min", "shift",
            "median", "q25", "q75", "iqr", "skew", "kurt",
            "ts_rank", "ts_zscore", "decay_linear", "slope",
            "days_since_rolling_high", "days_since_rolling_low",
        ):
            formulas.append(f"{op}(volume_1, {w})")

    for w in WINDOWS:
        formulas.append(f"ma_ratio(volume_1, {w})")
        formulas.append(f"sum(signed_volume(), {w})")
        for op in PAIR_TS_OPS:
            formulas.append(f"{op}(ret(close_1, 1), volume_ratio({w}), {w})")
            formulas.append(f"{op}(ret(close_1, 1), ret(market_close_1, 1), {w})")

    formulas.extend(f"{op}(close_1)" for op in CUMULATIVE_TS_OPS)
    formulas.extend(f"{op}()" for op in CUMULATIVE_NOARG_OPS)
    formulas.extend(
        [
            "cummax(high_1)",
            "cummin(low_1)",
            "cum_sum(volume_1)",
        ]
    )

    formulas.extend(
        [
            "(ema(close_1, 12) - ema(close_1, 26))",
            "((ema(close_1, 12) / ema(close_1, 26)) - const(1))",
            "bb_pos(close_1, 20)",
            "bb_width(close_1, 20)",
            "rsi(close_1, 14)",
            "stoch(14)",
            "stoch_d(14, 3)",
            "mfi(14)",
            "cmf(20)",
            "adx(14)",
            "cci(20)",
            "willr(14)",
            "keltner_pos(20)",
            "keltner_width(20)",
            "donchian_width(20)",
            "gt(rsi(close_1, 14), const(70))",
            "lt(rsi(close_1, 14), const(30))",
            "cross_above(ema(close_1, 12), ema(close_1, 26))",
            "cross_below(ema(close_1, 12), ema(close_1, 26))",
            "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0))",
            "where(lt(rsi(close_1, 14), const(30)), const(1), const(0))",
            (
                "where(lt(rsi(close_1, 14), const(30)), const(1), "
                "where(gt(rsi(close_1, 14), const(70)), const(-1), const(0)))"
            ),
            "rule_signal(rsi(close_1, 14), const(30), const(70))",
            "winsorize(ret(close_1, 10))",
            "neutralize(ret(close_1, 10))",
            "vol_scale(ret(close_1, 1), 20)",
            "ts_rank(close_1, 20)",
            "ts_zscore(ret(close_1, 1), 20)",
            "decay_linear(ret(close_1, 1), 20)",
            "slope(close_1, 20)",
            "iqr(ret(close_1, 1), 20)",
            "ts_corr(ret(close_1, 1), volume_ratio(20), 20)",
            "ts_beta(ret(close_1, 1), volume_ratio(20), 20)",
            "vwap(20)",
            "vwap_pos(20)",
            "amihud(20)",
            "parkinson_vol(20)",
            "gk_vol(20)",
            "rs_vol(20)",
            "efficiency_ratio(close_1, 20)",
            "ulcer_index(close_1, 20)",
            "days_since_rolling_high(close_1, 20)",
            "days_since_rolling_low(close_1, 20)",
            "aroon_up(20)",
            "aroon_down(20)",
            "aroon_osc(20)",
            "choppiness(20)",
            "up_streak(close_1)",
            "down_streak(close_1)",
            "cum_adl()",
            "cum_pvt()",
            "market_ret(20)",
            "market_vol(20)",
            "market_drawdown(20)",
            "market_ma_ratio(20)",
            "market_rsi(14)",
            "market_pos(20)",
            "market_volume_ratio(20)",
            "rel_ret(20)",
            "rel_strength(20)",
            "market_corr(20)",
            "market_beta(20)",
            "market_alpha(20)",
            "idiosyncratic_vol(20)",
            "up_capture(20)",
            "down_capture(20)",
            "advance_ratio()",
            "decline_ratio()",
            "advance_decline_ratio()",
            "advance_decline_net_pct()",
            "cs_dispersion()",
            "pct_above_ma(20)",
            "pct_above_ma(60)",
            "breadth_momentum(20)",
            "sector_advance_ratio()",
            "sector_decline_ratio()",
            "sector_advance_decline_ratio()",
            "sector_advance_decline_net_pct()",
            "sector_dispersion()",
            "sector_ret(20)",
            "sector_vol(20)",
            "sector_drawdown(20)",
            "sector_ma_ratio(20)",
            "sector_rsi(14)",
            "sector_pos(20)",
            "sector_volume_ratio(20)",
            "rel_sector_ret(20)",
            "sector_rel_strength(20)",
            "sector_corr(20)",
            "sector_beta(20)",
            "sector_alpha(20)",
            "sector_idiosyncratic_vol(20)",
            "sector_up_capture(20)",
            "sector_down_capture(20)",
            "sector_pct_above_ma(20)",
            "sector_pct_above_ma(60)",
            "sector_breadth_momentum(20)",
            "sector_rank(ret(close_1, 1))",
            "sector_zscore(ret(close_1, 1))",
            "sector_neutralize(ret(close_1, 1))",
        ]
    )

    return list(dict.fromkeys(formulas))


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    """
    Spearman rank correlation for feature de-duplication.

    Fitness is rank-based, so admission should reject monotonic copies of an
    existing feature, not only linear Pearson duplicates. Return 1.0 for an
    unusable new signal so callers reject it.
    """
    try:
        aligned = (
            pd.concat([a, b], axis=1)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(aligned) < 10:
            return 1.0  # quá ít data → coi như duplicate, reject
        # Hằng số → corr = NaN → nếu không chặn sẽ pass threshold
        std_a = aligned.iloc[:, 0].std()
        std_b = aligned.iloc[:, 1].std()
        if std_a < 1e-8:
            return 1.0  # reject: gene không có thông tin
        if std_b < 1e-8:
            return 0.0
        corr, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(corr) if np.isfinite(corr) else 0.0
    except Exception:
        return 0.0


def _evaluate_clean(formula: str, df: pd.DataFrame) -> pd.Series:
    return evaluate(formula, df).replace([np.inf, -np.inf], np.nan)


def _feature_quality_violation(
    formula: str,
    series: pd.Series,
    train_df: pd.DataFrame,
    min_valid_ratio: float = FEATURE_MIN_VALID_RATIO,
    max_dominant_value_ratio: float = FEATURE_MAX_DOMINANT_VALUE_RATIO,
) -> str | None:
    """Reject features whose signal exists on too little of the train context."""
    if len(train_df) == 0:
        return f"no rows available for feature quality check: {formula!r}"

    aligned = series.reindex(train_df.index)
    clean = aligned.dropna()
    if len(clean) == 0:
        return f"feature has no valid values: {formula!r}"

    if min_valid_ratio > 0.0:
        valid_ratio = float(aligned.notna().mean())
        if valid_ratio < float(min_valid_ratio):
            return (
                f"valid ratio {valid_ratio:.3f} is below "
                f"FEATURE_MIN_VALID_RATIO={float(min_valid_ratio):.3f}"
            )

    if max_dominant_value_ratio < 1.0:
        dominant_ratio = float(clean.value_counts(normalize=True).iloc[0])
        if dominant_ratio > float(max_dominant_value_ratio):
            return (
                f"dominant value ratio {dominant_ratio:.3f} is above "
                "FEATURE_MAX_DOMINANT_VALUE_RATIO="
                f"{float(max_dominant_value_ratio):.3f}"
            )
    return None


def individual_corr_check(
    new_gene: Gene,
    existing_genes: List[Gene],
    train_df: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> bool:
    """
    Returns True nếu new_gene có corr < threshold với TẤT CẢ existing_genes
    VÀ new_gene không phải hằng số (std > 1e-8).
    """
    violation = const_threshold_violation(new_gene.formula)
    if violation is not None:
        logger.debug(
            "corr_check: reject unsafe const threshold %r - %s",
            new_gene.formula,
            violation,
        )
        return False
    violation = raw_scale_violation(new_gene.formula)
    if violation is not None:
        logger.debug(
            "corr_check: reject raw-scale feature %r - %s",
            new_gene.formula,
            violation,
        )
        return False

    try:
        if has_division_by_zero(new_gene.formula, train_df):
            logger.debug(
                "corr_check: reject division-by-zero gene %r",
                new_gene.formula,
            )
            return False
    except Exception as exc:
        logger.warning(
            "corr_check division check failed for %r: %s",
            new_gene.formula, exc,
        )
        return False

    try:
        full_series = _evaluate_clean(new_gene.formula, train_df)
    except Exception as exc:
        logger.warning("corr_check eval failed for %r: %s", new_gene.formula, exc)
        return False

    quality_violation = _feature_quality_violation(
        new_gene.formula,
        full_series,
        train_df,
    )
    if quality_violation is not None:
        logger.debug(
            "corr_check: reject low-quality feature %r - %s",
            new_gene.formula,
            quality_violation,
        )
        return False

    new_series = full_series.dropna()

    # Loại gene hằng số ngay tại đây
    if new_series.std() < 1e-8:
        logger.debug("corr_check: reject constant gene %r", new_gene.formula)
        return False

    for g in existing_genes:
        try:
            ex_series = (
                evaluate(g.formula, train_df)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
        except Exception:
            continue
        corr = _safe_corr(new_series, ex_series)
        if abs(corr) >= threshold:
            return False
    return True
