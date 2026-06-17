"""
Evo_Finance — Domain
─────────────────────
The Domain is the ever-growing pool of formula strings that the Mutator
can draw from.  A new formula is admitted only if its Pearson correlation
with every existing formula in the domain is < CORR_THRESHOLD (computed on
the training set).

Thread-safety: single-process assumed; no locking needed.
"""

from __future__ import annotations
import logging
from typing import List, Optional
import numpy as np
import pandas as pd

from config.settings import CORR_THRESHOLD, WINDOWS
from mutator.gene import (
    Gene, BREADTH_NOARG_OPS, BREADTH_WINDOW_OPS,
    CONSTANT_FORMULAS, CUMULATIVE_NOARG_OPS, CUMULATIVE_TS_OPS,
    FINANCE_NOARG_OPS, FINANCE_TS_OPS, FINANCE_TWO_WINDOW_OPS,
    FINANCE_WINDOW_OPS, MARKET_WINDOW_OPS, PAIR_TS_OPS,
    SECTOR_CS_OPS, SECTOR_NOARG_OPS, SECTOR_WINDOW_OPS, TS_ROLLING_OPS,
)
from mutator.evaluator import evaluate, has_division_by_zero

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
        # Cache: formula → computed Series on train set
        self._cache: dict[str, pd.Series] = {}

    # ── Initialisation ────────────────────────────────────────────────────────

    def seed(self, window: int = 1) -> List[Gene]:
        """Populate with raw OHLCV genes; return them for the first individual."""
        genes = [Gene.raw(b, window) for b in _RAW_BASES]
        for g in genes:
            self._formulas.append(g.formula)

        finance_formulas = _finance_seed_formulas()
        for formula in finance_formulas:
            if formula not in self._formulas:
                self._formulas.append(formula)

        logger.info(
            "Domain seeded with %d raw features + %d finance primitives.",
            len(genes), len(finance_formulas),
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
          - Its correlation with every existing domain member < CORR_THRESHOLD
            (evaluated on train_df).

        Parameters
        ----------
        gene      : Gene to test.
        train_df  : Training DataFrame (features must be computable).
        force     : If True, skip correlation check (used for seed genes).

        Returns True if the gene was added.
        """
        if gene.formula in self._formulas:
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
                new_series = self._compute(gene.formula, train_df)
            except Exception as exc:
                logger.warning("Domain.try_add: cannot evaluate %r — %s",
                               gene.formula, exc)
                return False

            for existing_formula in self._formulas:
                ex_series = self._cache.get(existing_formula)
                if ex_series is None:
                    continue
                corr = _safe_corr(new_series, ex_series)
                if corr >= CORR_THRESHOLD:
                    return False

            # passed — cache it
            self._cache[gene.formula] = new_series

        self._formulas.append(gene.formula)
        logger.debug("Domain ← %r  (size=%d)", gene.formula, len(self._formulas))
        return True

    def precompute(self, train_df: pd.DataFrame) -> None:
        """Materialise all domain formulas on train_df (call once after seeding)."""
        for f in self._formulas:
            if f not in self._cache:
                try:
                    self._cache[f] = self._compute(f, train_df)
                except Exception as exc:
                    logger.warning("Domain.precompute: %r failed — %s", f, exc)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, formula: str, df: pd.DataFrame) -> pd.Series:
        if formula in self._cache:
            return self._cache[formula]
        series = evaluate(formula, df).dropna()
        self._cache[formula] = series
        return series


# ─── Utility ──────────────────────────────────────────────────────────────────

def _finance_seed_formulas() -> List[str]:
    formulas: List[str] = []
    formulas.extend(CONSTANT_FORMULAS)

    for op in FINANCE_NOARG_OPS:
        base = f"{op}()"
        formulas.append(base)
        formulas.extend(f"({base})_w{w}" for w in WINDOWS)

    for op in FINANCE_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)

    formulas.extend(f"{op}()" for op in BREADTH_NOARG_OPS)
    for op in BREADTH_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)

    for op in SECTOR_NOARG_OPS:
        formulas.append(f"{op}()")
    for op in SECTOR_WINDOW_OPS:
        formulas.extend(f"{op}({w})" for w in WINDOWS)
    for op in SECTOR_CS_OPS:
        formulas.extend(
            f"{op}({expr})"
            for expr in (
                "close_1", "ret(close_1, 1)", "volume_ratio(20)",
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
            "advance_count()",
            "decline_count()",
            "unchanged_count()",
            "advance_ratio()",
            "decline_ratio()",
            "advance_decline_ratio()",
            "advance_decline_spread()",
            "advance_decline_net_pct()",
            "cs_dispersion()",
            "pct_above_ma(20)",
            "pct_above_ma(60)",
            "breadth_momentum(20)",
            "sector_code()",
            "sector_size()",
            "sector_advance_count()",
            "sector_decline_count()",
            "sector_unchanged_count()",
            "sector_advance_ratio()",
            "sector_decline_ratio()",
            "sector_advance_decline_ratio()",
            "sector_advance_decline_spread()",
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
    Pearson correlation; trả về 1.0 (reject) nếu:
      - Không đủ dữ liệu
      - Một trong hai series là hằng số (std ≈ 0) → gene vô nghĩa như x/x
    Trả về 0.0 nếu có exception khác.
    """
    try:
        aligned = pd.concat([a, b], axis=1).dropna()
        if len(aligned) < 10:
            return 1.0  # quá ít data → coi như duplicate, reject
        # Hằng số → corr = NaN → nếu không chặn sẽ pass threshold
        std_a = aligned.iloc[:, 0].std()
        std_b = aligned.iloc[:, 1].std()
        if std_a < 1e-8:
            return 1.0  # reject: gene không có thông tin
        if std_b < 1e-8:
            return 0.0
        return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
    except Exception:
        return 0.0


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
        new_series = evaluate(new_gene.formula, train_df).dropna()
    except Exception as exc:
        logger.warning("corr_check eval failed for %r: %s", new_gene.formula, exc)
        return False

    # Loại gene hằng số ngay tại đây
    if new_series.std() < 1e-8:
        logger.debug("corr_check: reject constant gene %r", new_gene.formula)
        return False

    for g in existing_genes:
        try:
            ex_series = evaluate(g.formula, train_df).dropna()
        except Exception:
            continue
        corr = _safe_corr(new_series, ex_series)
        if abs(corr) >= threshold:
            return False
    return True
