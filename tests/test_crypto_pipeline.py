import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from crypto import config
from crypto.backtest import (
    BASE_FRACTION_BAND_MAX,
    BASE_FRACTION_BAND_STEP,
    BASE_SIGNAL_HIGH_GROUP,
    BASE_SIGNAL_LOW_GROUP,
    BundleSignals,
    ModelSpec,
    SplitSignals,
    _base_fraction_band_indices,
    _base_fraction_band_tp_rows,
    _base_overview,
    _apply_score_band_entry_filter,
    _backtest_name,
    _dynamic_tp_arrays,
    _draw_table,
    _fraction_band_sweep_table,
    _fraction_band_two_sided_table,
    _overview_table,
    _optimize_two_sided_tp_pair,
    _parse_spec,
    _score_band_two_sided_tp_strategy,
    _score_band_fixed_h5_metrics,
    _score_band_sweep_thresholds,
    _score_band_strategy_table,
    _simulate_score_band_fixed_h5_frame,
    _simulate_two_sided_tp_frame,
    _simulate_dynamic_tp_arrays,
    _summarize_split,
    _sweep_table,
    _tp_sweep_rows,
    _two_sided_score_band_trade_path,
    _two_sided_tp_grid,
    _two_sided_sweep_thresholds,
    _two_sided_score_band_table,
    _two_sided_tp_metrics,
    _train_spec_bundle,
)
import matplotlib.pyplot as plt
from crypto.main import _validate_resume_metadata
from crypto.data import CryptoFold, add_binary_labels, split_labeled_by_dates
from crypto.evolution import CryptoArchive, CryptoIndividual, CryptoMutator
from crypto.expression import CryptoFeatureSpace
from crypto.features import (
    RAW_SCALE_COLUMNS,
    _rolling_rank_pct,
    build_feature_frame,
    selectable_features,
)
from crypto.fitness import (
    CryptoFitnessEvaluator,
    _classification_trade_metrics,
    _internal_early_stop_split,
)
from crypto.prod.live_backend import (
    SCORE_BAND_COUNT,
    _prediction_score_band_index,
    _score_band_name,
    _score_band_percent_range,
)
from crypto.prod.train_model import SCORE_BAND_FRACTIONS, _score_band_cutoffs


class CryptoPipelineTests(unittest.TestCase):
    def test_prod_score_bands_cover_full_val_distribution(self):
        pred = pd.Series(np.arange(1, 101, dtype=float))
        cutoffs = _score_band_cutoffs(pred)

        self.assertEqual(SCORE_BAND_COUNT, 20)
        self.assertEqual(len(SCORE_BAND_FRACTIONS), 20)
        self.assertAlmostEqual(SCORE_BAND_FRACTIONS[0], 0.05)
        self.assertAlmostEqual(SCORE_BAND_FRACTIONS[-1], 1.0)
        self.assertEqual(set(cutoffs), {f"q{index}" for index in range(1, 21)})
        self.assertEqual(_prediction_score_band_index(100.0, cutoffs), 1)
        self.assertEqual(_prediction_score_band_index(1.0, cutoffs), 20)
        self.assertEqual(_prediction_score_band_index(0.0, cutoffs), 20)
        self.assertEqual(_score_band_name(1), "Q1")
        self.assertEqual(_score_band_name(20), "Q20")
        self.assertEqual(_score_band_percent_range(1), "0%-5%")
        self.assertEqual(_score_band_percent_range(20), "95%-100%")

    def test_prod_score_band_rejects_legacy_six_cutoff_manifest(self):
        legacy = {f"q{index}": float(7 - index) for index in range(1, 7)}
        self.assertIsNone(_prediction_score_band_index(10.0, legacy))

    def test_score_band_ands_horizon_top_sets_before_building_band(self):
        idx = pd.date_range("2024-01-01", periods=1000, freq="15min")
        pred_h1 = pd.Series(np.linspace(1.0, 0.0, len(idx)), index=idx)
        pred_h2 = pd.Series(np.linspace(0.20, 0.0, len(idx)), index=idx)
        pred_h2.iloc[:25] = np.linspace(1.0, 0.96, 25)
        pred_h2.iloc[50:75] = np.linspace(0.95, 0.91, 25)

        def horizon_split(split, pred):
            data = pd.DataFrame({"label": 0.0, "pred": pred}, index=idx)
            selected = data.nlargest(50, "pred").index
            return SplitSignals(split, data, selected, float(pred.loc[selected].min()), 0.05)

        val_h1 = horizon_split("val", pred_h1)
        val_h2 = horizon_split("val", pred_h2)
        test_h1 = horizon_split("test", pred_h1)
        test_h2 = horizon_split("test", pred_h2)
        expected = val_h1.selected_index.intersection(val_h2.selected_index)
        combined_data = pd.DataFrame(
            {
                "label": 0.0,
                "pred": pd.concat([pred_h1, pred_h2], axis=1).mean(axis=1),
            },
            index=idx,
        )
        bundle = BundleSignals(
            label="h-ensemble",
            val=SplitSignals("val", combined_data, expected, 0.0, 0.05),
            test=SplitSignals("test", combined_data, expected, 0.0, 0.05),
            val_horizons=(val_h1, val_h2),
            test_horizons=(test_h1, test_h2),
        )

        bands = _base_fraction_band_indices(
            [bundle],
            selection="and",
            max_fraction=0.05,
        )

        self.assertEqual(len(expected), 25)
        self.assertTrue(bands["val"][0][2].equals(expected))
        self.assertTrue(bands["test"][0][2].equals(expected))

    def test_backtest_base_fraction_bands_are_disjoint(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min")
        pred = pd.Series(np.linspace(1.0, 0.01, len(idx)), index=idx)
        data = pd.DataFrame({"label": 0.0, "pred": pred}, index=idx)
        split_val = SplitSignals("val", data, idx[:20], 0.80, 0.20)
        split_test = SplitSignals("test", data, idx[:20], 0.80, 0.20)
        bundle = BundleSignals("base", split_val, split_test)
        path_returns = pd.DataFrame(index=idx)
        for horizon in range(1, 25):
            path_returns[f"high_h{horizon}"] = 0.0
            path_returns[f"low_h{horizon}"] = 0.0
            path_returns[f"close_h{horizon}"] = 0.0
        path_returns["high_h24"] = 0.002
        path_returns["low_h24"] = -0.0015

        rows = _base_fraction_band_tp_rows(
            base_bundles=[bundle],
            selection="and",
            raw_path_returns=path_returns,
            thresholds=[0.001],
        )
        raw = pd.DataFrame(rows)
        band_count = int(
            round(BASE_FRACTION_BAND_MAX / BASE_FRACTION_BAND_STEP)
        )

        for split in ("val", "test"):
            split_rows = raw[
                (raw["split"] == split) & (raw["group"] == BASE_SIGNAL_HIGH_GROUP)
            ]
            self.assertEqual(len(split_rows), band_count)
            self.assertEqual(
                int(split_rows["sample_count"].sum()),
                int(round(BASE_FRACTION_BAND_MAX * len(idx))),
            )
            self.assertTrue((split_rows["sample_count"] == 5).all())
            self.assertTrue((split_rows["hit_rate"] == 1.0).all())

        rendered = _fraction_band_sweep_table(
            raw,
            group_name=BASE_SIGNAL_HIGH_GROUP,
            thresholds=[0.001],
        )
        self.assertEqual(len(rendered), band_count * 2)
        self.assertIn("val top 0%-5% n=5", rendered["split / score band"].tolist())
        self.assertIn(
            "test top 65%-70% n=5",
            rendered["split / score band"].tolist(),
        )
        low_h1_rendered = _fraction_band_sweep_table(
            raw,
            group_name=BASE_SIGNAL_LOW_GROUP,
            thresholds=[-0.001],
        )
        self.assertEqual(len(low_h1_rendered), band_count * 2)
        self.assertTrue((low_h1_rendered["-0.10%"] == "100.00%").all())
        two_sided_rendered = _fraction_band_two_sided_table(raw)
        self.assertEqual(len(_two_sided_sweep_thresholds()), 20)
        self.assertEqual(len(_score_band_sweep_thresholds()), 21)
        self.assertEqual(_score_band_sweep_thresholds()[0], 0.0)
        self.assertEqual(_score_band_sweep_thresholds()[-1], 0.01)
        self.assertEqual(len(two_sided_rendered), band_count * 2)
        self.assertTrue((two_sided_rendered["0.10%"] == "100.00%").all())
        self.assertTrue((two_sided_rendered["0.15%"] == "0.00%").all())
        self.assertIn("1.00%", two_sided_rendered.columns)

        # Score-band excursion tables always use raw price geometry, even when
        # the archive evaluation direction is Short.
        short_rows = _base_fraction_band_tp_rows(
            base_bundles=[bundle],
            selection="and",
            raw_path_returns=path_returns,
            thresholds=[0.001],
        )
        short_low = _fraction_band_sweep_table(
            pd.DataFrame(short_rows),
            group_name=BASE_SIGNAL_LOW_GROUP,
            thresholds=[-0.001],
        )
        self.assertTrue((short_low["-0.10%"] == "100.00%").all())

        overview = _base_overview(bundle, path_returns, tp_threshold=0.004)
        rendered_overview = _overview_table(overview)
        self.assertEqual(
            rendered_overview.columns.tolist(),
            ["split", "base signal", "avg trades/day", "TP_threshold"],
        )
        self.assertEqual(rendered_overview["base signal"].tolist(), ["20", "20"])
        self.assertEqual(rendered_overview["avg trades/day"].tolist(), ["10.00", "10.00"])

        trade_path = _two_sided_score_band_trade_path(
            base_bundle=bundle,
            raw_path_returns=path_returns,
            tp_long=0.003,
            tp_short=0.002,
        )
        self.assertEqual(len(trade_path), 40)
        for split in ("val", "test"):
            split_path = trade_path[trade_path["split"] == split]
            self.assertEqual(len(split_path), 20)
            self.assertTrue(split_path.index.is_monotonic_increasing)
            self.assertAlmostEqual(
                float(split_path["cumulative_net_return"].iloc[-1]),
                float(split_path["net_return"].sum()),
            )

    def test_backtest_trains_each_spec_with_its_own_direction(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="15min")
        frame = pd.DataFrame(index=idx)
        spec = ModelSpec(
            archive_path=Path("short.json"),
            rank=1,
            label_mode="payoff",
            label_threshold=0.002,
            top_fraction=0.20,
            label_direction="short",
        )
        val = SplitSignals("val", frame, idx[:1], 0.5, 0.20)
        test = SplitSignals("test", frame, idx[1:2], 0.5, 0.20)

        with (
            patch(
                "crypto.backtest.add_binary_labels", return_value=frame
            ) as add_labels,
            patch(
                "crypto.backtest.split_labeled_by_dates",
                return_value=(frame, frame, frame),
            ),
            patch("crypto.backtest._entry_to_individual", return_value=object()),
            patch(
                "crypto.backtest._train_one_horizon_signal",
                return_value=(5, val, test),
            ),
        ):
            bundle = _train_spec_bundle(
                spec=spec,
                entry={},
                raw_df=frame,
                feature_space=object(),
                horizons=[5],
                val_start="2024-01-01",
                test_start="2024-02-01",
                test_end=None,
                purge_bars=5,
            )

        self.assertEqual(add_labels.call_args.kwargs["label_direction"], "short")
        self.assertIn("payoff short", bundle.label)

    def test_backtest_base_sweep_reports_two_sided_moves(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="15min")
        raw_high = np.array([0.003, 0.003, 0.0005, 0.001])
        raw_low = np.array([-0.002, -0.0005, -0.003, -0.001])

        for direction, sign in (("long", 1.0), ("short", -1.0)):
            path_returns = pd.DataFrame(
                {
                    "high_h1": raw_high * sign,
                    "low_h1": raw_low * sign,
                    "close_h1": np.zeros(len(idx)),
                },
                index=idx,
            )
            rows = _tp_sweep_rows(
                split="val",
                group="p1_base_signal",
                selected_base_index=idx,
                path_returns=path_returns,
                thresholds=[-0.001, 0.001],
                min_h=1,
                label_direction=direction,
                include_two_sided_move=True,
            )

            self.assertTrue(np.isnan(rows[0]["two_sided_rate"]))
            self.assertEqual(rows[1]["two_sided_count"], 1)
            self.assertAlmostEqual(rows[1]["two_sided_rate"], 0.25)

            rendered = _sweep_table(pd.DataFrame(rows), group_name="p1_base_signal")
            two_sided = rendered[rendered["split"] == "val high>thr & low<-thr"].iloc[0]
            self.assertEqual(two_sided["-0.10%"], "")
            self.assertEqual(two_sided["0.10%"], "25.00%")

    def test_backtest_draw_table_handles_empty_results(self):
        fig, ax = plt.subplots()
        try:
            _draw_table(
                ax,
                pd.DataFrame(columns=["rank", "val E[net]"]),
                title="empty optimizer",
                font_size=7.0,
            )
            self.assertEqual(len(ax.tables), 0)
            self.assertTrue(any("No samples" in text.get_text() for text in ax.texts))
        finally:
            plt.close(fig)

    def test_backtest_name_stays_within_windows_safe_length(self):
        base_specs = [
            ModelSpec(
                archive_path=Path(f"crypto_btc_archive_with_a_long_name_{i}.json"),
                rank=1,
                label_mode="mfe",
                label_threshold=0.003,
                top_fraction=0.15,
            )
            for i in range(5)
        ]
        exit1 = ModelSpec(
            Path("crypto_btc_exit_k1_h5_tp04_seed1_12h.json"),
            1,
            "exit_after_k",
            0.004,
            0.20,
            exit_after_k=1,
        )
        exit2 = ModelSpec(
            Path("crypto_btc_exit_k2_h5_tp04_seed1_12h.json"),
            1,
            "exit_after_k",
            0.004,
            0.10,
            exit_after_k=2,
        )

        name = _backtest_name(base_specs, exit1, exit2, "and", 0.005)

        self.assertLessEqual(len(name), 150)
        self.assertTrue(name.startswith("strategy_e1top_20p000pct_e2top_10p000pct"))
        self.assertIn("_bases5_", name)

    def test_dynamic_tp_simulator_routes_all_five_terminal_branches(self):
        frame = pd.DataFrame(
            {
                "high_h1": [0.006, 0.0, 0.0, 0.0, 0.0],
                "high_h2": [0.0, 0.0, 0.005, 0.0, 0.0],
                "max_high_h2_plus": [0.0, 0.005, 0.005, 0.005, 0.004],
                "max_high_h3_plus": [0.0, 0.0, 0.0, 0.005, 0.004],
                "close_final": [-0.01] * 5,
                "exit1_selected": [False, True, False, False, False],
                "exit2_selected": [False, False, False, True, False],
            }
        )
        thresholds = (0.005, 0.0045, 0.004, 0.004, 0.003)

        metrics = _simulate_dynamic_tp_arrays(
            _dynamic_tp_arrays(frame),
            thresholds,
        )

        expected_gross = sum(thresholds) / len(thresholds)
        self.assertAlmostEqual(
            metrics["e_net"],
            expected_gross - config.TRADE_COST,
        )
        self.assertEqual(metrics["hit_rate"], 1.0)
        self.assertEqual(metrics["n_trades"], 5.0)

    def test_score_band_fixed_strategy_routes_h1_through_h5_and_close_h5(self):
        frame = pd.DataFrame(
            {
                "high_h1": [0.007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "high_h2": [0.0, 0.007, 0.0, 0.0, 0.0, 0.0, 0.0],
                "high_h3": [0.0, 0.0, 0.007, 0.0, 0.0, 0.0, 0.0],
                "high_h4": [0.0, 0.0, 0.0, 0.007, 0.0, 0.0, 0.0],
                "open_h5": [0.0, 0.0, 0.0, 0.0, -0.001, -0.003, -0.003],
                "high_h5": [0.0, 0.0, 0.0, 0.0, 0.0, -0.002, -0.0021],
                "adverse_h1": [0.0] * 7,
                "adverse_h2": [0.0] * 7,
                "adverse_h3": [0.0] * 7,
                "adverse_h4": [0.0] * 7,
                "adverse_h5": [0.0] * 7,
                "close_h2": [-0.004, -0.003, -0.002, -0.001, 0.002, 0.003, 0.004],
                "close_h5": [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.006],
            }
        )
        for horizon in range(1, 6):
            frame[f"raw_low_h{horizon}"] = [
                -0.003,
                -0.003,
                -0.0005,
                -0.0005,
                -0.0005,
                -0.0005,
                -0.003,
            ]
        frame.loc[frame.index[2], "raw_low_h3"] = -0.003
        for horizon in range(1, 6):
            frame.loc[frame.index[4], f"raw_low_h{horizon}"] = 0.002
        frame["close_h3"] = [-0.004, -0.003, -0.002, -0.001, 0.002, 0.003, 0.004]
        frame["close_h4"] = [-0.004, -0.003, -0.002, -0.001, 0.002, 0.003, 0.004]
        frame.loc[frame.index[4], "close_h5"] = 0.003

        with (
            patch("crypto.backtest.SCORE_BAND_CUTLOSS", -0.008),
            patch("crypto.backtest.SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD", -1.0),
        ):
            detailed = _simulate_score_band_fixed_h5_frame(frame)

        self.assertEqual(
            detailed["outcome"].tolist(),
            [
                "tp_h1",
                "tp_h2",
                "tp_h3",
                "tp_h4",
                "open_h5",
                "tp_h5",
                "close_h5",
            ],
        )
        self.assertTrue(
            np.allclose(
                detailed["realized"].to_numpy(),
                [0.007, 0.007, 0.007, 0.007, -0.001, -0.002, -0.006],
            )
        )

        with patch("crypto.backtest.SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD", 0.001):
            metrics = _score_band_fixed_h5_metrics(
                detailed,
                split="val",
                band_start=0.0,
                band_end=0.05,
                signals_before_filter=10,
            )
            rendered = _score_band_strategy_table(pd.DataFrame([metrics]))
        self.assertEqual(metrics["signals_before_filter"], 10)
        self.assertEqual(metrics["trades"], 7)
        self.assertEqual(rendered.iloc[0]["n"], "10")
        self.assertEqual(rendered.iloc[0]["n_after_filter"], "7")
        self.assertEqual(metrics["hit_h1"], 1)
        self.assertEqual(metrics["hit_h2"], 1)
        self.assertEqual(metrics["hit_h3"], 1)
        self.assertEqual(metrics["hit_h4"], 1)
        self.assertEqual(metrics["hit_open_h5"], 1)
        self.assertEqual(metrics["hit_h5"], 1)
        self.assertEqual(metrics["cutloss"], 0)
        self.assertEqual(metrics["close_h5"], 1)
        self.assertEqual(metrics["high_h1_below_threshold"], 6)
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_rate"], 6 / 7
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_high_h2_mean"],
            0.007 / 6,
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_close_h2_mean"],
            0.0005,
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_high_h3_mean"],
            0.007 / 6,
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_close_h3_mean"],
            0.0005,
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_high_h4_mean"],
            0.007 / 6,
        )
        self.assertAlmostEqual(
            metrics["high_h1_below_threshold_close_h4_mean"],
            0.0005,
        )
        self.assertAlmostEqual(
            metrics["e_net"],
            np.mean([0.007, 0.007, 0.007, 0.007, -0.001, -0.002, -0.006])
            - config.TRADE_COST,
        )
        self.assertEqual(rendered.iloc[0]["hit H1"], "1 (14.3%) | +0.70%")
        self.assertEqual(rendered.iloc[0]["hit H2"], "1 (14.3%) | +0.70%")
        self.assertEqual(rendered.iloc[0]["hit H3"], "1 (14.3%) | +0.70%")
        self.assertEqual(rendered.iloc[0]["hit H4"], "1 (14.3%) | +0.70%")
        self.assertEqual(rendered.iloc[0]["hit open H5"], "1 (14.3%) | -0.10%")
        self.assertEqual(rendered.iloc[0]["hit H5"], "1 (14.3%) | -0.20%")
        self.assertEqual(rendered.iloc[0]["cutloss"], "0 (0.0%) | n/a")
        self.assertEqual(rendered.iloc[0]["close H5"], "1 (14.3%) | -0.60%")
        self.assertEqual(
            rendered.iloc[0]["weak H1 TP H2-H4"],
            "0 (0.0%) | n/a",
        )
        self.assertEqual(
            rendered.iloc[0]["high H1<0.10% / all"],
            "6 (85.7%)",
        )
        self.assertEqual(
            rendered.iloc[0]["high H1<0.10% mean high H2 | mean close H2"],
            "+0.12% | +0.05%",
        )
        self.assertEqual(
            rendered.iloc[0]["high H1<0.10% mean high H3 | mean close H3"],
            "+0.12% | +0.05%",
        )
        self.assertEqual(
            rendered.iloc[0]["high H1<0.10% mean high H4 | mean close H4"],
            "+0.12% | +0.05%",
        )

    def test_score_band_weak_h1_reprices_tp_for_h2_through_h4(self):
        frame = pd.DataFrame(
            {
                "high_h1": [0.001, 0.001, 0.001, 0.003],
                "high_h2": [-0.001, -0.002, -0.002, 0.007],
                "high_h3": [-0.002, -0.001, -0.002, 0.0],
                "high_h4": [-0.002, -0.002, -0.002, 0.0],
                "open_h5": [-0.003, -0.003, -0.003, -0.003],
                "high_h5": [-0.002, -0.002, -0.002, -0.002],
                "adverse_h1": [0.0] * 4,
                "adverse_h2": [0.0] * 4,
                "adverse_h3": [0.0] * 4,
                "adverse_h4": [0.0] * 4,
                "adverse_h5": [0.0] * 4,
                "close_h5": [-0.004] * 4,
            }
        )

        with (
            patch("crypto.backtest.SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD", 0.002),
            patch("crypto.backtest.SCORE_BAND_WEAK_H1_TP_H2_H4", -0.001),
            patch("crypto.backtest.SCORE_BAND_CUTLOSS", -1.0),
        ):
            detailed = _simulate_score_band_fixed_h5_frame(frame)

        self.assertEqual(
            detailed["outcome"].tolist(),
            ["weak_tp_h2", "weak_tp_h3", "tp_h5", "tp_h2"],
        )
        self.assertTrue(
            np.allclose(detailed["realized"], [-0.001, -0.001, -0.002, 0.007])
        )
        metrics = _score_band_fixed_h5_metrics(
            detailed,
            split="val",
            band_start=0.0,
            band_end=0.05,
        )
        self.assertEqual(metrics["weak_h1_tp_h2_h4"], 2)
        self.assertEqual(metrics["hit_h2"], 2)
        self.assertEqual(metrics["hit_h3"], 1)

    def test_score_band_entry_filter_uses_only_h0_low(self):
        frame = pd.DataFrame(
            {
                "entry_filter_low_h0": [-0.001, -0.003, -0.002],
                "marker": ["pass", "below", "equal"],
            }
        )

        with patch("crypto.backtest.SCORE_BAND_ENTRY_MIN_LOW_THRESHOLD", -0.002):
            filtered = _apply_score_band_entry_filter(frame)

        self.assertEqual(filtered["marker"].tolist(), ["pass"])

    def test_two_sided_score_band_settles_each_leg_independently(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="15min")
        frame = pd.DataFrame(index=idx)
        for horizon in range(1, 6):
            frame[f"high_h{horizon}"] = [0.004, 0.004, 0.001, 0.001]
            frame[f"low_h{horizon}"] = [-0.004, -0.001, -0.004, -0.001]
        frame["close_h5"] = [0.001, -0.002, 0.002, -0.001]

        detailed = _simulate_two_sided_tp_frame(
            frame,
            tp_long=0.003,
            tp_short=0.002,
        )

        self.assertEqual(detailed["long_hit"].tolist(), [True, True, False, False])
        self.assertEqual(detailed["short_hit"].tolist(), [True, False, True, False])
        self.assertTrue(
            np.allclose(detailed["gross_return"], [0.005, 0.005, 0.004, 0.0])
        )
        metrics = _two_sided_tp_metrics(
            detailed,
            split="val",
            band_start=0.0,
            band_end=0.05,
        )
        self.assertEqual(metrics["both_hit"], 1)
        self.assertEqual(metrics["long_only"], 1)
        self.assertEqual(metrics["short_only"], 1)
        self.assertEqual(metrics["neither_hit"], 1)
        self.assertAlmostEqual(metrics["gross_mean"], 0.0035)
        self.assertAlmostEqual(metrics["e_net"], 0.0035 - config.TRADE_COST)

        rendered = _two_sided_score_band_table(pd.DataFrame([metrics]))
        self.assertEqual(rendered.iloc[0]["both TP"], "1 (25.0%)")
        self.assertEqual(rendered.iloc[0]["gross mean"], "+0.35%")

    def test_two_sided_tp_optimizer_includes_zero_and_both_maxima(self):
        frame = pd.DataFrame(index=pd.RangeIndex(4))
        for horizon in range(1, 25):
            frame[f"high_h{horizon}"] = 0.005
            frame[f"low_h{horizon}"] = -0.002
        frame["close_h24"] = 0.0

        long_grid = _two_sided_tp_grid(0.01)
        short_grid = _two_sided_tp_grid(0.01)
        selected = _optimize_two_sided_tp_pair(
            frame,
            max_tp_long=0.01,
            max_tp_short=0.01,
            exit_horizon=24,
        )

        self.assertEqual(long_grid[0], 0.0)
        self.assertEqual(short_grid[0], 0.0)
        self.assertAlmostEqual(long_grid[-1], 0.01)
        self.assertAlmostEqual(short_grid[-1], 0.01)
        self.assertAlmostEqual(selected[0], 0.005)
        self.assertAlmostEqual(selected[1], 0.002)

    def test_two_sided_score_band_applies_val_optimized_pair_to_test(self):
        val_idx = pd.date_range("2024-01-01", periods=1000, freq="15min")
        test_idx = pd.date_range("2025-01-01", periods=1000, freq="15min")
        pred = np.linspace(1.0, 0.0, 1000)

        def split(name, index):
            data = pd.DataFrame({"label": 0.0, "pred": pred}, index=index)
            return SplitSignals(name, data, index[:50], 0.95, 0.05)

        bundle = BundleSignals("base", split("val", val_idx), split("test", test_idx))
        path = pd.DataFrame(index=val_idx.append(test_idx))
        for horizon in range(1, 25):
            path[f"high_h{horizon}"] = 0.0
            path[f"low_h{horizon}"] = 0.0
            path.loc[val_idx, f"high_h{horizon}"] = 0.005
            path.loc[val_idx, f"low_h{horizon}"] = -0.002
            path.loc[test_idx, f"high_h{horizon}"] = 0.009
            path.loc[test_idx, f"low_h{horizon}"] = -0.009
            path[f"close_h{horizon}"] = 0.0

        results = _score_band_two_sided_tp_strategy(
            [bundle],
            selection="and",
            raw_path_returns=path,
            tp_long=0.01,
            tp_short=0.01,
        )
        val_first = results[
            (results["split"] == "val") & (results["score_band"] == "top 0%-5%")
        ].iloc[0]
        test_first = results[
            (results["split"] == "test") & (results["score_band"] == "top 0%-5%")
        ].iloc[0]

        self.assertAlmostEqual(val_first["tp_long"], 0.005)
        self.assertAlmostEqual(val_first["tp_short"], 0.002)
        self.assertAlmostEqual(test_first["tp_long"], 0.005)
        self.assertAlmostEqual(test_first["tp_short"], 0.002)

    def test_two_sided_score_band_can_use_fixed_configured_pair(self):
        val_idx = pd.date_range("2024-01-01", periods=1000, freq="15min")
        test_idx = pd.date_range("2025-01-01", periods=1000, freq="15min")
        pred = np.linspace(1.0, 0.0, 1000)

        def split(name, index):
            data = pd.DataFrame({"label": 0.0, "pred": pred}, index=index)
            return SplitSignals(name, data, index[:50], 0.95, 0.05)

        bundle = BundleSignals("base", split("val", val_idx), split("test", test_idx))
        path = pd.DataFrame(index=val_idx.append(test_idx))
        for horizon in range(1, 25):
            path[f"high_h{horizon}"] = 0.009
            path[f"low_h{horizon}"] = -0.009
            path[f"close_h{horizon}"] = 0.0

        results = _score_band_two_sided_tp_strategy(
            [bundle],
            selection="and",
            raw_path_returns=path,
            tp_long=0.007,
            tp_short=0.004,
            optimize_tp=False,
        )
        band_rows = results.dropna(subset=["tp_long", "tp_short"])

        self.assertTrue(np.allclose(band_rows["tp_long"], 0.007))
        self.assertTrue(np.allclose(band_rows["tp_short"], 0.004))
        trade_path = _two_sided_score_band_trade_path(
            base_bundle=bundle,
            raw_path_returns=path,
            tp_long=0.007,
            tp_short=0.004,
            optimize_tp=False,
        )
        self.assertTrue(np.allclose(trade_path["tp_long"], 0.007))
        self.assertTrue(np.allclose(trade_path["tp_short"], 0.004))

    def test_two_sided_trade_path_uses_each_bands_val_optimized_pair(self):
        val_idx = pd.date_range("2024-01-01", periods=1000, freq="15min")
        test_idx = pd.date_range("2025-01-01", periods=1000, freq="15min")
        pred = np.linspace(1.0, 0.0, 1000)

        def split(name, index):
            data = pd.DataFrame({"label": 0.0, "pred": pred}, index=index)
            return SplitSignals(name, data, index[:100], 0.90, 0.10)

        bundle = BundleSignals("base", split("val", val_idx), split("test", test_idx))
        path = pd.DataFrame(index=val_idx.append(test_idx))
        for horizon in range(1, 25):
            path[f"high_h{horizon}"] = 0.0
            path[f"low_h{horizon}"] = 0.0
            path[f"close_h{horizon}"] = 0.0
        for index in (val_idx, test_idx):
            path.loc[index[:50], "high_h1"] = 0.005
            path.loc[index[:50], "low_h1"] = -0.002
            path.loc[index[50:100], "high_h1"] = 0.008
            path.loc[index[50:100], "low_h1"] = -0.004

        strategy = _score_band_two_sided_tp_strategy(
            [bundle],
            selection="and",
            raw_path_returns=path,
            tp_long=0.01,
            tp_short=0.01,
            optimize_tp=True,
        )
        trade_path = _two_sided_score_band_trade_path(
            base_bundle=bundle,
            raw_path_returns=path,
            tp_long=0.01,
            tp_short=0.01,
            optimize_tp=True,
            base_bundles=[bundle],
            selection="and",
            score_band_strategy=strategy,
        )

        for split_name in ("val", "test"):
            split_path = trade_path[trade_path["split"] == split_name]
            first = split_path[split_path["score_band"] == "top 0%-5%"]
            second = split_path[split_path["score_band"] == "top 5%-10%"]
            self.assertEqual(len(first), 50)
            self.assertEqual(len(second), 50)
            self.assertTrue(np.allclose(first["tp_long"], 0.005))
            self.assertTrue(np.allclose(first["tp_short"], 0.002))
            self.assertTrue(np.allclose(second["tp_long"], 0.008))
            self.assertTrue(np.allclose(second["tp_short"], 0.004))

    def test_two_sided_score_band_uses_h24_and_closes_unfilled_leg_at_h24(self):
        frame = pd.DataFrame(index=pd.RangeIndex(2))
        for horizon in range(1, 25):
            frame[f"high_h{horizon}"] = 0.0
            frame[f"low_h{horizon}"] = 0.0
        frame.loc[0, "high_h24"] = 0.004
        frame.loc[1, "low_h24"] = -0.003
        frame["close_h24"] = [-0.001, 0.002]

        detailed = _simulate_two_sided_tp_frame(
            frame,
            tp_long=0.003,
            tp_short=0.002,
            exit_horizon=24,
        )

        self.assertEqual(detailed["long_hit"].tolist(), [True, False])
        self.assertEqual(detailed["short_hit"].tolist(), [False, True])
        self.assertTrue(np.allclose(detailed["long_return"], [0.003, 0.002]))
        self.assertTrue(np.allclose(detailed["short_return"], [0.001, 0.002]))

    def test_score_band_cutloss_wins_same_candle_tp_ties(self):
        frame = pd.DataFrame(
            {
                "high_h1": [0.007, 0.0],
                "high_h2": [0.0, 0.0],
                "high_h3": [0.0, 0.0],
                "high_h4": [0.0, 0.0],
                "open_h5": [0.0, -0.003],
                "high_h5": [0.0, -0.002],
                "adverse_h1": [-0.008, 0.0],
                "adverse_h2": [0.0, 0.0],
                "adverse_h3": [0.0, 0.0],
                "adverse_h4": [0.0, 0.0],
                "adverse_h5": [0.0, -0.008],
                "close_h5": [0.0, 0.0],
            }
        )

        with (
            patch("crypto.backtest.SCORE_BAND_CUTLOSS", -0.008),
            patch("crypto.backtest.SCORE_BAND_MAX_HIGH_BELOW_THRESHOLD", -1.0),
        ):
            detailed = _simulate_score_band_fixed_h5_frame(frame)

        self.assertEqual(detailed["outcome"].tolist(), ["cutloss_h1", "cutloss_h5"])
        self.assertTrue(np.allclose(detailed["realized"], [-0.008, -0.008]))
        metrics = _score_band_fixed_h5_metrics(
            detailed,
            split="val",
            band_start=0.0,
            band_end=0.05,
        )
        self.assertEqual(metrics["cutloss"], 2)
        self.assertEqual(metrics["hit_h1"], 0)
        self.assertEqual(metrics["hit_h5"], 0)
        rendered = _score_band_strategy_table(pd.DataFrame([metrics]))
        self.assertEqual(rendered.iloc[0]["cutloss"], "2 (100.0%) | -0.80%")

    def test_backtest_part1_counts_h1_low_drawdown_rates(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min")
        selected = pd.Index(idx[:3])
        base_split = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=selected,
            pred_threshold=0.5,
            top_fraction=0.25,
        )
        empty_exit = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=pd.Index([]),
            pred_threshold=0.5,
            top_fraction=0.20,
        )
        path_returns = pd.DataFrame(index=idx)
        for horizon in range(1, 6):
            path_returns[f"high_h{horizon}"] = 0.0
            path_returns[f"close_h{horizon}"] = 0.0
        path_returns["low_h1"] = [-0.0005, -0.001, -0.003] + [0.0] * 5
        path_returns["close_h2"] = [-0.001, -0.002, -0.003] + [0.0] * 5

        summary, tp_rows = _summarize_split(
            split="val",
            base_split=base_split,
            exit1_split=empty_exit,
            exit2_split=empty_exit,
            path_returns=path_returns,
            raw_index=pd.DatetimeIndex(idx),
            tp_threshold=0.004,
            tp_sweep_thresholds=[0.004],
            label_direction="long",
        )

        self.assertEqual(summary["base_low_h1_le_neg01"], 2)
        self.assertAlmostEqual(summary["base_low_h1_le_neg01_rate"], 2 / 3)
        self.assertEqual(summary["base_low_h1_le_neg005"], 3)
        self.assertAlmostEqual(summary["base_low_h1_le_neg005_rate"], 1.0)
        tp_sweep = pd.DataFrame(tp_rows)
        exit2_no_selected = tp_sweep[
            tp_sweep["group"] == "p2_exit_k2_no_selected"
        ]
        self.assertAlmostEqual(
            float(exit2_no_selected["close_h2_return_mean"].iloc[0]),
            -0.002,
        )
        rendered = _sweep_table(
            tp_sweep,
            group_name="p2_exit_k2_no_selected",
            include_close_h2=True,
        )
        self.assertIn("val mean close H2", rendered["split"].tolist())

    def test_backtest_short_uses_low_for_tp_and_high_for_adverse_h1(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min")
        selected = pd.Index(idx[:3])
        base_split = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=selected,
            pred_threshold=0.5,
            top_fraction=0.25,
        )
        empty_exit = SplitSignals(
            split="val",
            data=pd.DataFrame(index=idx),
            selected_index=pd.Index([]),
            pred_threshold=0.5,
            top_fraction=0.20,
        )
        path_returns = pd.DataFrame(index=idx)
        for horizon in range(1, 6):
            path_returns[f"low_h{horizon}"] = 0.0
            path_returns[f"high_h{horizon}"] = 0.0
            path_returns[f"close_h{horizon}"] = 0.0
        path_returns["low_h1"] = [0.005, 0.001, 0.0] + [0.0] * 5
        path_returns["high_h1"] = [-0.001, -0.0005, 0.0] + [0.0] * 5

        summary, _ = _summarize_split(
            split="val",
            base_split=base_split,
            exit1_split=empty_exit,
            exit2_split=empty_exit,
            path_returns=path_returns,
            raw_index=pd.DatetimeIndex(idx),
            tp_threshold=0.004,
            tp_sweep_thresholds=[0.004],
            label_direction="short",
        )

        self.assertEqual(summary["base_no_h1"], 2)
        self.assertEqual(summary["base_low_h1_le_neg01"], 1)
        self.assertEqual(summary["base_low_h1_le_neg005"], 2)

    def test_backtest_base_specs_accept_independent_top_fractions(self):
        mfe_spec = _parse_spec(
            "mfe.json#1#mfe#0.003#0.15",
            default_top_fraction=0.25,
        )
        close_spec = _parse_spec(
            "close.json#1#close_exit#-0.001#0.25",
            default_top_fraction=0.15,
        )

        self.assertEqual(mfe_spec.top_fraction, 0.15)
        self.assertEqual(close_spec.top_fraction, 0.25)

    def test_config_allows_finite_negative_label_threshold(self):
        old_threshold = config.LABEL_THRESHOLD
        try:
            config.LABEL_THRESHOLD = -0.001
            config.validate_config()
        finally:
            config.LABEL_THRESHOLD = old_threshold

    def test_binary_label_uses_next_open_and_future_close(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 102.5, 103.5, 104.5, 105.5],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[2],
            threshold=0.001,
            return_fn=config.close_exit_future_return,
        )

        expected = (df["close"].shift(-2) - df["open"].shift(-1)) / df["open"].shift(-1)
        pd.testing.assert_series_equal(
            labeled["future_return_h2"],
            expected.rename("future_return_h2"),
        )
        self.assertEqual(labeled["label_h2"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labeled["label_h2"].iloc[-1]))

    def test_mfe_label_uses_mfe_but_fitness_return_uses_tp_or_final_close(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 104.0, 101.0, 101.0, 101.0],
                "low": [99.6] * 5,
                "close": [100.0, 100.0, 100.0, 100.0, 98.0],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[3],
            threshold=0.03,
            label_mode="mfe",
        )

        expected = pd.Series(
            [0.03, -0.02, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue(pd.isna(labeled["label_h3"].iloc[-1]))

    def test_mfe_ahead_uses_current_h1_open_and_current_h1_high(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [104.0, 101.0, 101.0, 100.0, 100.0],
                "low": [99.0] * 5,
                "close": [100.0, 100.0, 98.0, 98.0, 100.0],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[3],
            threshold=0.03,
            label_mode="mfe_ahead",
            label_direction="Long",
        )

        expected = pd.Series(
            [0.03, -0.02, 0.0, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertEqual(
            config.canonical_label_mode("MFE_Ahead"),
            "mfe_ahead",
        )

    def test_payoff_label_requires_long_and_short_adverse_path_floor(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        common = {
            "open": [100.0] * 5,
            "close": [100.0] * 5,
            "volume": [10.0] * 5,
            "trade_count": [10] * 5,
            "taker_buy_base_volume": [5.0] * 5,
            "taker_buy_quote_volume": [500.0] * 5,
        }
        long_df = pd.DataFrame(
            {
                **common,
                "high": [100.0, 100.5, 100.0, 100.0, 100.0],
                "low": [100.0, 99.8, 99.4, 99.8, 99.8],
            },
            index=idx,
        )
        short_df = pd.DataFrame(
            {
                **common,
                "high": [100.0, 100.2, 100.6, 100.2, 100.2],
                "low": [100.0, 99.5, 100.0, 100.0, 100.0],
            },
            index=idx,
        )

        old_tp = config.PAYOFF_TP
        old_floor = config.PAYOFF_ADVERSE_FLOOR
        try:
            config.PAYOFF_TP = 0.003
            config.PAYOFF_ADVERSE_FLOOR = -0.005
            long_labeled = add_binary_labels(
                long_df,
                horizons=[3],
                label_mode="payoff",
                label_direction="Long",
                threshold=0.0005,
            )
            short_labeled = add_binary_labels(
                short_df,
                horizons=[3],
                label_mode="payoff",
                label_direction="Short",
                threshold=0.0005,
            )
        finally:
            config.PAYOFF_TP = old_tp
            config.PAYOFF_ADVERSE_FLOOR = old_floor

        self.assertAlmostEqual(long_labeled["future_return_h3"].iloc[0], 0.003)
        self.assertEqual(long_labeled["label_h3"].iloc[0], 0.0)
        self.assertAlmostEqual(short_labeled["future_return_h3"].iloc[0], 0.003)
        self.assertEqual(short_labeled["label_h3"].iloc[0], 0.0)

    def test_short_direction_uses_price_down_as_positive_return(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 120.0, 130.0, 101.0, 100.0],
                "low": [100.0, 99.5, 98.0, 99.0, 100.0],
                "close": [100.0, 100.0, 99.0, 101.0, 100.0],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        close_labeled = add_binary_labels(
            df,
            horizons=[2],
            label_mode="close_exit",
            label_direction="Short",
            threshold=0.005,
        )
        self.assertAlmostEqual(close_labeled["future_return_h2"].iloc[0], 0.01)
        self.assertEqual(close_labeled["label_h2"].iloc[0], 1.0)
        self.assertAlmostEqual(close_labeled["future_return_h2"].iloc[1], -0.01)
        self.assertEqual(close_labeled["label_h2"].iloc[1], 0.0)

        mfe_labeled = add_binary_labels(
            df,
            horizons=[3],
            label_mode="mfe",
            label_direction="Short",
            threshold=0.015,
        )
        self.assertAlmostEqual(mfe_labeled["future_return_h3"].iloc[0], 0.015)
        self.assertEqual(mfe_labeled["label_h3"].iloc[0], 1.0)

    def test_adverse_floor_uses_directional_path_and_zero_fitness_return(self):
        idx = pd.date_range("2024-01-01", periods=8, freq="15min")
        frame = pd.DataFrame(
            {
                "open": [100.0] * 8,
                "high": [100.0, 100.2, 100.2, 100.2, 100.4, 100.1, 100.1, 100.1],
                "low": [100.0, 99.8, 99.8, 99.8, 99.6, 99.9, 99.9, 99.9],
                "close": [100.0] * 8,
                "volume": [10.0] * 8,
                "trade_count": [10] * 8,
                "taker_buy_base_volume": [5.0] * 8,
                "taker_buy_quote_volume": [500.0] * 8,
            },
            index=idx,
        )

        long_labeled = add_binary_labels(
            frame,
            horizons=[3],
            label_mode="adverse_floor",
            label_direction="Long",
            threshold=0.003,
        )
        short_labeled = add_binary_labels(
            frame,
            horizons=[3],
            label_mode="adverse_floor",
            label_direction="Short",
            threshold=0.003,
        )

        self.assertEqual(long_labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(long_labeled["label_h3"].iloc[1], 0.0)
        self.assertEqual(short_labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(short_labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue((long_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue((short_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue(long_labeled["label_h3"].iloc[-3:].isna().all())
        self.assertIs(
            config.get_label_return_fn("adverse_floor"),
            config.adverse_floor_future_return,
        )

    def test_high_exit_uses_exact_h_candle_and_zero_fitness_return(self):
        idx = pd.date_range("2024-01-01", periods=7, freq="15min")
        frame = pd.DataFrame(
            {
                "open": [100.0] * 7,
                # Earlier extremes must not turn the exact-H3 labels positive.
                "high": [100.0, 110.0, 110.0, 100.1, 100.4, 100.0, 100.0],
                "low": [100.0, 90.0, 90.0, 99.9, 99.6, 100.0, 100.0],
                "close": [100.0] * 7,
                "volume": [10.0] * 7,
                "trade_count": [10] * 7,
                "taker_buy_base_volume": [5.0] * 7,
                "taker_buy_quote_volume": [500.0] * 7,
            },
            index=idx,
        )

        long_labeled = add_binary_labels(
            frame,
            horizons=[3],
            label_mode="high_exit",
            label_direction="Long",
            threshold=0.002,
        )
        short_labeled = add_binary_labels(
            frame,
            horizons=[3],
            label_mode="high_exit",
            label_direction="Short",
            threshold=0.002,
        )

        self.assertEqual(long_labeled["label_h3"].iloc[0], 0.0)
        self.assertEqual(long_labeled["label_h3"].iloc[1], 1.0)
        self.assertEqual(short_labeled["label_h3"].iloc[0], 0.0)
        self.assertEqual(short_labeled["label_h3"].iloc[1], 1.0)
        self.assertTrue((long_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue((short_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue(long_labeled["label_h3"].iloc[-3:].isna().all())
        self.assertTrue(short_labeled["label_h3"].iloc[-3:].isna().all())
        self.assertIs(
            config.get_label_return_fn("high_exit"),
            config.high_exit_future_return,
        )
        self.assertTrue(config.is_precision_only_label_mode("high_exit"))
        self.assertFalse(config.is_precision_only_label_mode("close_exit"))

    def test_slope_slowdown_uses_high_and_aligns_expanded_window_to_t(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="15min")

        def frame_with_high(high):
            return pd.DataFrame(
                {
                    "open": [90.0] * len(high),
                    "high": high,
                    "low": [80.0] * len(high),
                    # Flat close makes the test fail if close is used.
                    "close": [90.0] * len(high),
                    "volume": [10.0] * len(high),
                    "trade_count": [10] * len(high),
                    "taker_buy_base_volume": [5.0] * len(high),
                    "taker_buy_quote_volume": [500.0] * len(high),
                },
                index=idx,
            )

        long_high = [100, 101, 102, 103, 104, 103, 101, 99, 100, 100, 100, 100]
        short_high = [104, 103, 102, 101, 100, 101, 103, 105, 104, 104, 104, 104]

        long_source = config.slope_slowdown_future_return(
            frame_with_high(long_high),
            horizon=3,
            direction="Long",
        )
        short_source = config.slope_slowdown_future_return(
            frame_with_high(short_high),
            horizon=3,
            direction="Short",
        )

        x5 = np.arange(5, dtype=float)
        x8 = np.arange(8, dtype=float)
        long_initial = np.expm1(np.polyfit(x5, np.log(long_high[:5]), 1)[0])
        long_expanded = np.expm1(np.polyfit(x8, np.log(long_high[:8]), 1)[0])
        short_initial = np.expm1(np.polyfit(x5, np.log(short_high[:5]), 1)[0])
        short_expanded = np.expm1(np.polyfit(x8, np.log(short_high[:8]), 1)[0])

        self.assertAlmostEqual(
            long_source.iloc[4],
            long_initial - long_expanded,
            places=12,
        )
        self.assertAlmostEqual(
            short_source.iloc[4],
            short_expanded - short_initial,
            places=12,
        )
        self.assertGreater(long_source.iloc[4], 0.0003)
        self.assertGreater(short_source.iloc[4], 0.0003)

    def test_slope_slowdown_labels_both_directions_and_zeroes_fitness_return(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="15min")

        def labeled(high, direction):
            frame = pd.DataFrame(
                {
                    "open": [90.0] * len(high),
                    "high": high,
                    "low": [80.0] * len(high),
                    "close": [90.0] * len(high),
                    "volume": [10.0] * len(high),
                    "trade_count": [10] * len(high),
                    "taker_buy_base_volume": [5.0] * len(high),
                    "taker_buy_quote_volume": [500.0] * len(high),
                },
                index=idx,
            )
            return add_binary_labels(
                frame,
                horizons=[3],
                label_mode="slope_slowdown",
                label_direction=direction,
                threshold=0.0003,
            )

        long_labeled = labeled(
            [100, 101, 102, 103, 104, 103, 101, 99, 100, 100, 100, 100],
            "Long",
        )
        short_labeled = labeled(
            [104, 103, 102, 101, 100, 101, 103, 105, 104, 104, 104, 104],
            "Short",
        )

        self.assertEqual(long_labeled["label_h3"].iloc[4], 1.0)
        self.assertEqual(short_labeled["label_h3"].iloc[4], 1.0)
        # Complete future paths that fail the observable initial-slope gate
        # are excluded, not supplied to the model as easy label-0 samples.
        self.assertTrue(pd.isna(long_labeled["label_h3"].iloc[7]))
        self.assertTrue(pd.isna(short_labeled["label_h3"].iloc[7]))
        self.assertTrue(pd.isna(long_labeled["future_return_h3"].iloc[7]))
        self.assertTrue(pd.isna(short_labeled["future_return_h3"].iloc[7]))
        self.assertTrue((long_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue((short_labeled["future_return_h3"].dropna() == 0.0).all())
        self.assertTrue(long_labeled["label_h3"].iloc[-3:].isna().all())
        self.assertTrue(short_labeled["label_h3"].iloc[-3:].isna().all())
        self.assertIs(
            config.get_label_return_fn("slope_slowdown"),
            config.slope_slowdown_future_return,
        )
        self.assertTrue(config.is_precision_only_label_mode("slope_slowdown"))
        self.assertEqual(
            config.default_label_threshold("slope_slowdown"),
            config.SLOPE_SLOWDOWN_THRESHOLD,
        )

    def test_slope_slowdown_rejects_non_positive_threshold(self):
        frame = _synthetic_crypto_frame(20)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            add_binary_labels(
                frame,
                horizons=[3],
                label_mode="slope_slowdown",
                threshold=0.0,
            )

    def test_precision_only_metrics_do_not_charge_trade_cost_on_zero_return(self):
        y = pd.Series([1, 0, 1, 0])
        pred = pd.Series([0.9, 0.8, 0.7, 0.6])
        zero_return = pd.Series([0.0] * 4)

        metrics = _classification_trade_metrics(
            y_true=y,
            pred=pred,
            future_return=zero_return,
            charge_trade_cost=False,
        )

        self.assertEqual(metrics.trade_return_mean, 0.0)
        self.assertEqual(metrics.trade_return_score, 0.0)

    def test_adverse_floor_rejects_non_positive_threshold(self):
        frame = _synthetic_crypto_frame(20)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            add_binary_labels(
                frame,
                horizons=[3],
                label_mode="adverse_floor",
                threshold=0.0,
            )

    def test_long_safe_path_mfe_uses_stop_first_adverse_lows(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")

        def labeled_for(highs, lows, closes):
            frame = pd.DataFrame(
                {
                    "open": [100.0] * 6,
                    "high": [100.0, *highs],
                    "low": [100.0, *lows],
                    "close": [100.0, *closes],
                    "volume": [10.0] * 6,
                    "trade_count": [10] * 6,
                    "taker_buy_base_volume": [5.0] * 6,
                    "taker_buy_quote_volume": [500.0] * 6,
                },
                index=idx,
            )
            return add_binary_labels(
                frame,
                horizons=[5],
                label_mode="safe_path_mfe",
                threshold=-0.002,
            )

        old_tp = config.TP_SAFE_PATH
        try:
            config.TP_SAFE_PATH = 0.004

            clean_hit_h1 = labeled_for(
                highs=[100.4, 100.0, 100.0, 100.0, 100.0],
                lows=[99.9, 99.9, 99.9, 99.9, 99.9],
                closes=[100.1, 100.0, 100.0, 100.0, 100.0],
            )
            self.assertEqual(clean_hit_h1["label_h5"].iloc[0], 1.0)
            self.assertAlmostEqual(clean_hit_h1["future_return_h5"].iloc[0], 0.004)

            both_hit_and_stop_h1 = labeled_for(
                highs=[100.4, 100.0, 100.0, 100.0, 100.0],
                lows=[99.8, 99.9, 99.9, 99.9, 99.9],
                closes=[100.1, 100.0, 100.0, 100.0, 100.0],
            )
            self.assertEqual(both_hit_and_stop_h1["label_h5"].iloc[0], 0.0)
            self.assertAlmostEqual(
                both_hit_and_stop_h1["future_return_h5"].iloc[0],
                -0.002,
            )

            clean_h3 = labeled_for(
                highs=[100.1, 100.2, 100.4, 100.0, 100.0],
                lows=[99.9, 99.85, 99.9, 99.9, 99.9],
                closes=[100.0, 100.0, 100.1, 100.0, 100.0],
            )
            self.assertEqual(clean_h3["label_h5"].iloc[0], 1.0)
            self.assertAlmostEqual(clean_h3["future_return_h5"].iloc[0], 0.004)

            stopped_before_h3 = labeled_for(
                highs=[100.1, 100.2, 100.4, 100.0, 100.0],
                lows=[99.9, 99.7, 99.9, 99.9, 99.9],
                closes=[100.0, 99.9, 100.1, 100.0, 100.0],
            )
            self.assertEqual(stopped_before_h3["label_h5"].iloc[0], 0.0)
            self.assertAlmostEqual(
                stopped_before_h3["future_return_h5"].iloc[0],
                -0.002,
            )
            self.assertTrue(stopped_before_h3["label_h5"].iloc[1:].isna().all())
        finally:
            config.TP_SAFE_PATH = old_tp

    def test_safe_path_mfe_threshold_overrides_default_adverse_floor(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="15min")
        frame = pd.DataFrame(
            {
                "open": [100.0] * 4,
                "high": [100.0, 100.1, 100.4, 100.0],
                "low": [100.0, 99.85, 99.9, 99.9],
                "close": [100.0, 100.0, 100.1, 100.0],
                "volume": [10.0] * 4,
                "trade_count": [10] * 4,
                "taker_buy_base_volume": [5.0] * 4,
                "taker_buy_quote_volume": [500.0] * 4,
            },
            index=idx,
        )
        old_tp = config.TP_SAFE_PATH
        try:
            config.TP_SAFE_PATH = 0.004
            default_floor = add_binary_labels(
                frame,
                horizons=[2],
                label_mode="safe_path_mfe",
                threshold=-0.002,
            )
            tighter_floor = add_binary_labels(
                frame,
                horizons=[2],
                label_mode="safe_path_mfe",
                threshold=-0.001,
            )
        finally:
            config.TP_SAFE_PATH = old_tp

        self.assertEqual(default_floor["label_h2"].iloc[0], 1.0)
        self.assertEqual(tighter_floor["label_h2"].iloc[0], 0.0)

    def test_safe_path_mfe_is_registered_with_its_adverse_floor_default(self):
        self.assertEqual(
            config.default_label_threshold("safe_path_mfe"),
            config.SAFE_ADVERSE_FLOOR,
        )
        self.assertIs(
            config.get_label_return_fn("safe_path_mfe"),
            config.safe_path_mfe_future_return,
        )

    def test_short_safe_path_mfe_uses_stop_first_adverse_highs(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")

        def labeled_for(highs, lows, closes):
            frame = pd.DataFrame(
                {
                    "open": [100.0] * 6,
                    "high": [100.0, *highs],
                    "low": [100.0, *lows],
                    "close": [100.0, *closes],
                    "volume": [10.0] * 6,
                    "trade_count": [10] * 6,
                    "taker_buy_base_volume": [5.0] * 6,
                    "taker_buy_quote_volume": [500.0] * 6,
                },
                index=idx,
            )
            return add_binary_labels(
                frame,
                horizons=[5],
                label_mode="safe_path_mfe",
                label_direction="Short",
                threshold=-0.002,
            )

        old_tp = config.TP_SAFE_PATH
        try:
            config.TP_SAFE_PATH = 0.004

            clean_hit_h1 = labeled_for(
                highs=[100.1, 100.1, 100.1, 100.1, 100.1],
                lows=[99.6, 100.0, 100.0, 100.0, 100.0],
                closes=[99.9, 100.0, 100.0, 100.0, 100.0],
            )
            self.assertEqual(clean_hit_h1["label_h5"].iloc[0], 1.0)
            self.assertAlmostEqual(clean_hit_h1["future_return_h5"].iloc[0], 0.004)

            both_hit_and_stop_h1 = labeled_for(
                highs=[100.2, 100.1, 100.1, 100.1, 100.1],
                lows=[99.6, 100.0, 100.0, 100.0, 100.0],
                closes=[99.9, 100.0, 100.0, 100.0, 100.0],
            )
            self.assertEqual(both_hit_and_stop_h1["label_h5"].iloc[0], 0.0)
            self.assertAlmostEqual(
                both_hit_and_stop_h1["future_return_h5"].iloc[0],
                -0.002,
            )

            clean_h3 = labeled_for(
                highs=[100.1, 100.15, 100.1, 100.1, 100.1],
                lows=[99.9, 99.8, 99.6, 100.0, 100.0],
                closes=[100.0, 100.0, 99.9, 100.0, 100.0],
            )
            self.assertEqual(clean_h3["label_h5"].iloc[0], 1.0)
            self.assertAlmostEqual(clean_h3["future_return_h5"].iloc[0], 0.004)

            stopped_before_h3 = labeled_for(
                highs=[100.1, 100.3, 100.1, 100.1, 100.1],
                lows=[99.9, 99.8, 99.6, 100.0, 100.0],
                closes=[100.0, 100.1, 99.9, 100.0, 100.0],
            )
            self.assertEqual(stopped_before_h3["label_h5"].iloc[0], 0.0)
            self.assertAlmostEqual(
                stopped_before_h3["future_return_h5"].iloc[0],
                -0.002,
            )
        finally:
            config.TP_SAFE_PATH = old_tp

    def test_close_path_mean_label_uses_mean_future_closes_from_next_open(self):
        idx = pd.date_range("2024-01-01", periods=6, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 6,
                "high": [111.0] * 6,
                "low": [89.0] * 6,
                "close": [100.0, 99.0, 101.0, 103.0, 90.0, 110.0],
                "volume": [10.0] * 6,
                "trade_count": [10] * 6,
                "taker_buy_base_volume": [5.0] * 6,
                "taker_buy_quote_volume": [500.0] * 6,
            },
            index=idx,
        )

        labeled = add_binary_labels(
            df,
            horizons=[3],
            label_mode="close_path_mean",
            threshold=0.0,
        )

        expected = pd.Series(
            [0.01, -0.02, 0.01, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue(labeled["label_h3"].iloc[2] == 1.0)
        self.assertTrue(labeled["label_h3"].iloc[3:].isna().all())

    def test_close_path_mean_is_registered_as_default_mode(self):
        self.assertEqual(config.LABEL_MODE, "close_path_mean")
        self.assertEqual(config.default_label_threshold("close_path_mean"), 0.0)
        self.assertIs(
            config.get_label_return_fn("close_path_mean"),
            config.close_path_mean_future_return,
        )

    def test_payoff_label_uses_tp_or_future_close_and_preserves_tail_nan(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 100.5, 100.0, 100.0, 100.0],
                "low": [99.6] * 5,
                "close": [100.0, 100.0, 100.0, 100.0, 99.0],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        old_tp = config.PAYOFF_TP
        try:
            config.PAYOFF_TP = 0.003
            labeled = add_binary_labels(df, horizons=[3], label_mode="payoff")
        finally:
            config.PAYOFF_TP = old_tp

        expected = pd.Series(
            [0.003, -0.01, np.nan, np.nan, np.nan],
            index=idx,
            name="future_return_h3",
        )
        pd.testing.assert_series_equal(labeled["future_return_h3"], expected)
        self.assertEqual(labeled["label_h3"].iloc[0], 1.0)
        self.assertEqual(labeled["label_h3"].iloc[1], 0.0)
        self.assertTrue(pd.isna(labeled["label_h3"].iloc[-1]))

    def test_payoff_default_label_threshold_uses_trade_cost(self):
        self.assertEqual(
            config.default_label_threshold("payoff"),
            config.TRADE_COST,
        )
        self.assertEqual(
            config.default_label_threshold("mfe"),
            config.LABEL_THRESHOLD,
        )

    def test_two_sided_tp_payoff_covers_all_branches_and_ignores_direction(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                # Rows 1..4 are the H1 candles for signal rows 0..3.
                "high": [100.0, 101.0, 101.0, 100.2, 100.2],
                "low": [100.0, 99.0, 99.8, 99.0, 99.8],
                "close": [100.0, 100.0, 99.8, 100.2, 100.1],
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        long_labeled = add_binary_labels(
            df,
            horizons=[1],
            label_mode="two_sided_tp",
            label_direction="Long",
            threshold=0.005,
        )
        short_labeled = add_binary_labels(
            df,
            horizons=[1],
            label_mode="two_sided_tp",
            label_direction="Short",
            threshold=0.005,
        )

        expected_return = pd.Series(
            [0.01, 0.007, 0.007, 0.0, np.nan],
            index=idx,
            name="future_return_h1",
        )
        expected_label = pd.Series(
            [1.0, 0.0, 0.0, 0.0, np.nan],
            index=idx,
            name="label_h1",
        )
        pd.testing.assert_series_equal(long_labeled["future_return_h1"], expected_return)
        pd.testing.assert_series_equal(long_labeled["label_h1"], expected_label)
        pd.testing.assert_series_equal(
            short_labeled["future_return_h1"], long_labeled["future_return_h1"]
        )
        pd.testing.assert_series_equal(
            short_labeled["label_h1"], long_labeled["label_h1"]
        )

    def test_two_sided_tp_requires_positive_threshold(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="15min")
        df = pd.DataFrame(
            {"open": [100.0] * 3, "high": [101.0] * 3,
             "low": [99.0] * 3, "close": [100.0] * 3},
            index=idx,
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            add_binary_labels(
                df,
                horizons=[1],
                label_mode="two_sided_tp",
                threshold=0.0,
            )

    def test_two_sided_tp_is_registered_as_direction_neutral_payoff_mode(self):
        self.assertIs(
            config.get_label_return_fn("two_sided_tp"),
            config.two_sided_tp_future_return,
        )
        self.assertTrue(config.is_direction_neutral_label_mode("two_sided_tp"))
        self.assertFalse(config.is_precision_only_label_mode("two_sided_tp"))

    def test_crypto_archive_preserves_metadata_on_load(self):
        archive = CryptoArchive(
            metadata={"label_mode": "mfe", "label_threshold": 0.003}
        )
        individual = CryptoIndividual(features=["ret_close_3"], score=0.1)
        self.assertTrue(archive.try_add(individual))

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "archive.json"
            archive.save(path, metadata=archive.metadata)
            loaded = CryptoArchive.load(path)

        self.assertEqual(loaded.metadata["label_mode"], "mfe")
        self.assertEqual(loaded.metadata["label_threshold"], 0.003)
        self.assertEqual(len(loaded), 1)

    def test_resume_metadata_mismatch_is_rejected(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [3, 5],
                "label_mode": "mfe",
                "label_direction": "long",
                "label_threshold": 0.003,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "label_mode"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("archive.json"),
                horizons=[3, 5],
                label_mode="payoff",
                label_direction="long",
                label_threshold=config.TRADE_COST,
            )

    def test_resume_metadata_mismatch_checks_label_direction(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [5],
                "label_mode": "mfe",
                "label_direction": "short",
                "label_threshold": 0.003,
                "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "label_direction"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("archive.json"),
                horizons=[5],
                label_mode="mfe",
                label_direction="long",
                label_threshold=0.003,
            )

    def test_resume_metadata_ignores_direction_for_two_sided_tp(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [5],
                "label_mode": "two_sided_tp",
                "label_direction": "short",
                "label_threshold": 0.004,
                "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        _validate_resume_metadata(
            archive=archive,
            resume_path=Path("archive.json"),
            horizons=[5],
            label_mode="two_sided_tp",
            label_direction="long",
            label_threshold=0.004,
        )

    def test_resume_metadata_checks_slope_slowdown_definition(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [3],
                "label_mode": "slope_slowdown",
                "label_direction": "long",
                "label_threshold": config.SLOPE_SLOWDOWN_THRESHOLD,
                "slope_lookback": config.SLOPE_LOOKBACK + 1,
                "slope_min_initial": config.SLOPE_MIN_INITIAL,
                "slope_price_column": "high",
                "slope_slowdown_rule": config.SLOPE_SLOWDOWN_RULE,
                "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "slope_lookback"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("slope.json"),
                horizons=[3],
                label_mode="slope_slowdown",
                label_direction="long",
                label_threshold=config.SLOPE_SLOWDOWN_THRESHOLD,
            )

    def test_resume_rejects_legacy_slope_slowdown_rule(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [3],
                "label_mode": "slope_slowdown",
                "label_direction": "long",
                "label_threshold": config.SLOPE_SLOWDOWN_THRESHOLD,
                "slope_lookback": config.SLOPE_LOOKBACK,
                "slope_min_initial": config.SLOPE_MIN_INITIAL,
                "slope_price_column": "high",
                "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "incompatible slope_slowdown rule"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("legacy_slope.json"),
                horizons=[3],
                label_mode="slope_slowdown",
                label_direction="long",
                label_threshold=config.SLOPE_SLOWDOWN_THRESHOLD,
            )

    def test_resume_metadata_mismatch_checks_safe_path_tp(self):
        old_tp = config.TP_SAFE_PATH
        try:
            config.TP_SAFE_PATH = 0.004
            archive = CryptoArchive(
                metadata={
                    "horizons": [5],
                    "label_mode": "safe_path_mfe",
                    "label_direction": "long",
                    "label_threshold": -0.002,
                    "tp_safe_path": 0.003,
                    "safe_path_rule": config.SAFE_PATH_RULE,
                    "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                    "trade_top_fraction": config.TRADE_TOP_FRACTION,
                    "trade_cost": config.TRADE_COST,
                }
            )

            with self.assertRaisesRegex(ValueError, "tp_safe_path"):
                _validate_resume_metadata(
                    archive=archive,
                    resume_path=Path("archive.json"),
                    horizons=[5],
                    label_mode="safe_path_mfe",
                    label_direction="long",
                    label_threshold=-0.002,
                )
        finally:
            config.TP_SAFE_PATH = old_tp

    def test_resume_rejects_legacy_safe_path_rule(self):
        archive = CryptoArchive(
            metadata={
                "horizons": [5],
                "label_mode": "safe_path_mfe",
                "label_direction": "long",
                "label_threshold": -0.002,
                "tp_safe_path": config.TP_SAFE_PATH,
                "fitness_horizon_mode": config.FITNESS_HORIZON_MODE,
                "trade_top_fraction": config.TRADE_TOP_FRACTION,
                "trade_cost": config.TRADE_COST,
            }
        )

        with self.assertRaisesRegex(ValueError, "incompatible safe_path_mfe rule"):
            _validate_resume_metadata(
                archive=archive,
                resume_path=Path("legacy_safe_path.json"),
                horizons=[5],
                label_mode="safe_path_mfe",
                label_direction="long",
                label_threshold=-0.002,
            )

    def test_label_mode_mfe_is_used_by_default_labeling(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="15min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [100.0, 100.5, 101.5, 100.2, 100.1],
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [10.0] * 5,
                "trade_count": [10] * 5,
                "taker_buy_base_volume": [5.0] * 5,
                "taker_buy_quote_volume": [500.0] * 5,
            },
            index=idx,
        )

        old_mode = config.LABEL_MODE
        try:
            config.LABEL_MODE = "mfe"
            labeled = add_binary_labels(df, horizons=[2], threshold=0.01)
        finally:
            config.LABEL_MODE = old_mode

        self.assertEqual(labeled["label_h2"].iloc[0], 1.0)
        self.assertTrue(pd.isna(labeled["label_h2"].iloc[-1]))

    def test_feature_pool_excludes_raw_scale_columns(self):
        df = _synthetic_crypto_frame(700)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)

        self.assertGreater(len(pool), 20)
        self.assertFalse(set(pool) & RAW_SCALE_COLUMNS)

    def test_advanced_volatility_and_imbalance_features_are_available(self):
        df = _synthetic_crypto_frame(900)
        features = build_feature_frame(df, windows=[10, 20], min_valid_ratio=0.5)
        pool = set(selectable_features(features))

        expected = {
            "realized_vol_20",
            "downside_realized_vol_20",
            "upside_realized_vol_20",
            "parkinson_vol_20",
            "garman_klass_vol_20",
            "rogers_satchell_vol_20",
            "vol_of_vol_20",
            "taker_delta_sum_ratio_20",
            "taker_delta_z_20",
            "buy_pressure_persistence_20",
            "imbalance_x_high_volume_20",
            "imbalance_x_high_volatility_20",
            "imbalance_return_corr_20",
        }
        self.assertTrue(expected <= pool, sorted(expected - pool))

    def test_seed_individual_samples_randomly_from_feature_pool(self):
        pool = [f"feature_{idx}" for idx in range(12)]
        idx = pd.date_range("2024-01-01", periods=30, freq="15min")
        feature_df = pd.DataFrame(
            {
                name: np.linspace(float(pos), float(pos + 1), len(idx))
                for pos, name in enumerate(pool)
            },
            index=idx,
        )
        seed = 123
        mutator = CryptoMutator(pool, feature_df, idx, seed=seed)

        individual = mutator.seed_individual()

        expected_idx = np.random.default_rng(seed).choice(
            len(pool),
            size=config.FEATURE_MIN,
            replace=False,
        )
        expected = [pool[int(pos)] for pos in expected_idx]
        self.assertEqual(individual.features, expected)
        self.assertEqual(len(individual.features), config.FEATURE_MIN)
        self.assertEqual(len(set(individual.features)), config.FEATURE_MIN)
        self.assertTrue(set(individual.features) <= set(pool))

    def test_crypto_mutator_c1_adds_from_domain(self):
        idx, feature_df = _synthetic_feature_space(
            [f"feature_{idx}" for idx in range(12)]
        )
        pool = list(feature_df.columns)
        mutator = CryptoMutator(pool, feature_df, idx, seed=7)
        child = mutator.seed_individual()

        changed = mutator._c1(child)

        self.assertTrue(changed)
        self.assertEqual(len(child.features), config.FEATURE_MIN + 1)
        self.assertEqual(len(set(child.features)), len(child.features))
        self.assertTrue(set(child.features) <= set(pool))

    def test_crypto_mutator_c2_changes_window(self):
        pool = [f"alpha_{window}" for window in config.WINDOWS]
        idx, feature_df = _synthetic_feature_space(pool, rows=2000)
        mutator = CryptoMutator(pool, feature_df, idx, seed=3)
        individual = mutator.seed_individual()
        individual.features = ["alpha_3"]

        changed = mutator._c2(individual)

        self.assertTrue(changed)
        self.assertEqual(len(individual.features), 1)
        self.assertNotEqual(individual.features[0], "alpha_3")
        self.assertRegex(individual.features[0], r"^alpha_\d+$")
        self.assertIn(individual.features[0], pool)

    def test_crypto_mutator_c1_c2_fallback_to_c3(self):
        pool = [f"feature_{idx}" for idx in range(8)]
        idx, feature_df = _synthetic_feature_space(pool)
        original_probs = dict(config.MUTATOR_PROBS)
        cases = [
            ("c1", {"c1": 1.0, "c2": 0.0, "c3": 0.0}),
            ("c2", {"c1": 0.0, "c2": 1.0, "c3": 0.0}),
        ]

        for strategy, probs in cases:
            mutator = CryptoMutator(pool, feature_df, idx, seed=5)
            calls: list[str] = []

            def fake_c3(individual):
                calls.append("c3")
                individual.features[0] = "feature_7"
                return True

            try:
                config.MUTATOR_PROBS.update(probs)
                if strategy == "c1":
                    mutator._c1 = lambda individual: False  # type: ignore[method-assign]
                else:
                    mutator._c2 = lambda individual: False  # type: ignore[method-assign]
                mutator._c3 = fake_c3  # type: ignore[method-assign]
                parent = CryptoIndividual(features=["feature_0", "feature_1"])
                child = mutator.mutate(parent)
            finally:
                config.MUTATOR_PROBS.clear()
                config.MUTATOR_PROBS.update(original_probs)

            self.assertEqual(calls, ["c3"])
            self.assertIn("feature_7", child.features)

    def test_feature_quality_filter_can_use_train_only_index(self):
        df = _synthetic_crypto_frame(120)
        train_only = df.index[:50]

        full_quality = build_feature_frame(
            df,
            windows=[80],
            min_valid_ratio=0.10,
        )
        train_quality = build_feature_frame(
            df,
            windows=[80],
            min_valid_ratio=0.10,
            quality_index=train_only,
        )

        self.assertIn("ret_close_80", full_quality.columns)
        self.assertNotIn("ret_close_80", train_quality.columns)

    def test_feature_values_are_causal_under_future_truncation(self):
        df = _synthetic_crypto_frame(300)
        quality_index = df.index[:180]
        full = build_feature_frame(
            df,
            windows=[3, 10, 20],
            min_valid_ratio=0.10,
            quality_index=quality_index,
        )
        truncated = build_feature_frame(
            df.iloc[:220],
            windows=[3, 10, 20],
            min_valid_ratio=0.10,
            quality_index=quality_index,
        )
        common = sorted(set(full.columns) & set(truncated.columns))

        pd.testing.assert_frame_equal(
            full.loc[truncated.index, common],
            truncated[common],
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_test_end_split_purges_tail_labels(self):
        df = _synthetic_crypto_frame(12)
        labeled = add_binary_labels(df, horizons=[2], threshold=0.0)
        _, _, test_df = split_labeled_by_dates(
            labeled,
            val_start=str(df.index[2]),
            test_start=str(df.index[4]),
            test_end=str(df.index[9]),
            purge_bars=3,
        )

        self.assertEqual(test_df.index[0], df.index[4])
        self.assertEqual(test_df.index[-1], df.index[6])
        self.assertNotIn(df.index[7], test_df.index)

    def test_exit_after_k_supports_k1_and_k2(self):
        df = _synthetic_crypto_frame(40)
        for decision_k in (1, 2):
            labeled = add_binary_labels(
                df,
                horizons=[5],
                threshold=0.001,
                label_mode="exit_after_k",
                label_direction="Long",
                exit_after_k=decision_k,
            )
            self.assertIn("future_return_h5", labeled)
            self.assertIn("label_h5", labeled)
            self.assertGreater(labeled["label_h5"].notna().sum(), 0)

    def test_exit_after_k3_uses_open_h1_and_only_h4_h5_as_future(self):
        index = pd.date_range("2025-01-01", periods=8, freq="5min")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 1.0,
                "trade_count": 1.0,
                "taker_buy_base_volume": 0.5,
                "taker_buy_quote_volume": 50.0,
            },
            index=index,
        )
        # Decision row is index[3] (H3), entry is open index[1] (H1).
        df.loc[index[4], "high"] = 100.5  # H4 reaches +0.5%.
        labeled = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Long",
            exit_after_k=3,
        )

        # MFE reaches +0.5%, but executable payoff is capped at the +0.4% TP.
        self.assertAlmostEqual(labeled.loc[index[3], "future_return_h5"], 0.004)
        self.assertEqual(labeled.loc[index[3], "label_h5"], 1.0)

        # A prior H2 hit makes this decision row ineligible rather than label 0.
        df.loc[index[2], "high"] = 100.6
        filtered = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Long",
            exit_after_k=3,
        )
        self.assertTrue(pd.isna(filtered.loc[index[3], "label_h5"]))

    def test_exit_after_k3_short_uses_lows_and_excludes_prior_hit(self):
        index = pd.date_range("2025-01-01", periods=8, freq="5min")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 1.0,
                "trade_count": 1.0,
                "taker_buy_base_volume": 0.5,
                "taker_buy_quote_volume": 50.0,
            },
            index=index,
        )
        df.loc[index[5], "low"] = 99.4  # H5 reaches +0.6% for Short.
        labeled = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Short",
            exit_after_k=3,
        )
        # Short MFE reaches +0.6%, but executable payoff is capped at TP.
        self.assertAlmostEqual(labeled.loc[index[3], "future_return_h5"], 0.004)
        self.assertEqual(labeled.loc[index[3], "label_h5"], 1.0)

        df.loc[index[1], "low"] = 99.5  # H1 already hit Short TP.
        filtered = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Short",
            exit_after_k=3,
        )
        self.assertTrue(pd.isna(filtered.loc[index[3], "label_h5"]))

    def test_exit_after_k_miss_uses_final_close_return_for_long_and_short(self):
        index = pd.date_range("2025-01-01", periods=8, freq="5min")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 1.0,
                "trade_count": 1.0,
                "taker_buy_base_volume": 0.5,
                "taker_buy_quote_volume": 50.0,
            },
            index=index,
        )
        # Decision row index[3] is H3; final exit index[5] is H5.
        df.loc[index[5], "close"] = 99.0
        long_labeled = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Long",
            exit_after_k=3,
        )
        self.assertEqual(long_labeled.loc[index[3], "label_h5"], 0.0)
        self.assertAlmostEqual(
            long_labeled.loc[index[3], "future_return_h5"],
            -0.01,
        )

        df.loc[index[5], "close"] = 101.0
        short_labeled = add_binary_labels(
            df,
            horizons=[5],
            threshold=0.004,
            label_mode="exit_after_k",
            label_direction="Short",
            exit_after_k=3,
        )
        self.assertEqual(short_labeled.loc[index[3], "label_h5"], 0.0)
        self.assertAlmostEqual(
            short_labeled.loc[index[3], "future_return_h5"],
            -0.01,
        )

    def test_backtest_spec_accepts_explicit_exit_after_k(self):
        spec = _parse_spec(
            "missing.json#1#exit_after_k#0.004#0.20#Short#3",
            default_top_fraction=0.25,
        )
        self.assertEqual(spec.label_mode, "exit_after_k")
        self.assertEqual(spec.label_direction, "short")
        self.assertEqual(spec.exit_after_k, 3)

    def test_exit_after_k_must_precede_every_horizon(self):
        df = _synthetic_crypto_frame(20)
        with self.assertRaisesRegex(ValueError, "smaller than every holding horizon"):
            add_binary_labels(
                df,
                horizons=[3],
                threshold=0.001,
                label_mode="exit_after_k",
                exit_after_k=3,
            )

    def test_crypto_fitness_runs_on_synthetic_binary_fold(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(df, horizons=[3], threshold=0.001)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        fold = CryptoFold(
            name="wf_01",
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:850],
            train_start=labeled.index[0],
            train_end=labeled.index[599],
            val_start=labeled.index[620],
            val_end=labeled.index[849],
        )
        mutator = CryptoMutator(pool, feature_space, fold.train_df.index, seed=11)
        individual = mutator.seed_individual()
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 11,
        }

        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
        )
        score = evaluator.evaluate_walk_forward(individual, [fold], feature_space)

        self.assertTrue(np.isfinite(score))
        self.assertIn("mean_auc", individual.metrics)
        self.assertIn("precision_excess", individual.metrics)
        self.assertIn("trade_return_score", individual.metrics)

    def test_adverse_floor_fitness_is_return_neutral_in_walk_forward_and_final(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(
            df,
            horizons=[3],
            threshold=0.005,
            label_mode="adverse_floor",
            label_direction="Long",
        )
        feature_df = pd.DataFrame(
            {"ret": df["close"].pct_change().fillna(0.0)},
            index=df.index,
        )
        feature_space = CryptoFeatureSpace(feature_df, ["ret"])
        individual = CryptoIndividual(features=["ret"])
        fold = CryptoFold(
            name="wf_01",
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:750],
            train_start=labeled.index[0],
            train_end=labeled.index[599],
            val_start=labeled.index[620],
            val_end=labeled.index[749],
        )
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 19,
        }
        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
            precision_only=True,
        )

        score = evaluator.evaluate_walk_forward(individual, [fold], feature_space)
        final = evaluator.evaluate_final(
            individual=individual,
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:750],
            test_df=labeled.iloc[760:890],
            feature_data=feature_space,
        )

        self.assertTrue(np.isfinite(score))
        self.assertEqual(individual.metrics["trade_return_mean"], 0.0)
        self.assertEqual(individual.metrics["trade_return_score"], 0.0)
        self.assertEqual(final["final_val_trade_return_mean"], 0.0)
        self.assertEqual(final["final_val_trade_return_score"], 0.0)
        self.assertEqual(final["final_test_trade_return_mean"], 0.0)
        self.assertEqual(final["final_test_trade_return_score"], 0.0)

    def test_high_exit_fitness_is_return_neutral_in_walk_forward_and_final(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(
            df,
            horizons=[3],
            threshold=0.001,
            label_mode="high_exit",
            label_direction="Long",
        )
        feature_df = pd.DataFrame(
            {"ret": df["close"].pct_change().fillna(0.0)},
            index=df.index,
        )
        feature_space = CryptoFeatureSpace(feature_df, ["ret"])
        individual = CryptoIndividual(features=["ret"])
        fold = CryptoFold(
            name="wf_01",
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:750],
            train_start=labeled.index[0],
            train_end=labeled.index[599],
            val_start=labeled.index[620],
            val_end=labeled.index[749],
        )
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 23,
        }
        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
            precision_only=config.is_precision_only_label_mode("high_exit"),
        )

        score = evaluator.evaluate_walk_forward(individual, [fold], feature_space)
        final = evaluator.evaluate_final(
            individual=individual,
            train_df=labeled.iloc[:600],
            val_df=labeled.iloc[620:750],
            test_df=labeled.iloc[760:890],
            feature_data=feature_space,
        )

        self.assertTrue(np.isfinite(score))
        self.assertEqual(individual.metrics["trade_return_mean"], 0.0)
        self.assertEqual(individual.metrics["trade_return_score"], 0.0)
        self.assertEqual(final["final_val_trade_return_mean"], 0.0)
        self.assertEqual(final["final_val_trade_return_score"], 0.0)
        self.assertEqual(final["final_test_trade_return_mean"], 0.0)
        self.assertEqual(final["final_test_trade_return_score"], 0.0)

    def test_crypto_final_evaluation_appends_final_metrics(self):
        df = _synthetic_crypto_frame(900)
        labeled = add_binary_labels(df, horizons=[3], threshold=0.001)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        mutator = CryptoMutator(pool, feature_space, labeled.index[:500], seed=13)
        individual = mutator.seed_individual()
        individual.score = 0.01
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "force_col_wise": True,
            "verbose": -1,
            "seed": 13,
        }
        evaluator = CryptoFitnessEvaluator(
            horizons=[3],
            lgbm_params=params,
            num_boost_round=5,
            early_stopping_rounds=2,
        )

        metrics = evaluator.evaluate_final(
            individual=individual,
            train_df=labeled.iloc[:500],
            val_df=labeled.iloc[520:700],
            test_df=labeled.iloc[720:880],
            feature_data=feature_space,
        )

        self.assertIn("final_val_mean_auc", metrics)
        self.assertIn("final_test_mean_auc", metrics)
        self.assertIn("final_test_precision_excess", individual.metrics)
        self.assertTrue(np.isfinite(individual.metrics["final_test_mean_auc"]))

        archive = CryptoArchive(max_size=5)
        archive.try_add(individual)
        row = archive.summary()[0]
        self.assertIn("final_val_metrics", row)
        self.assertIn("test_metrics", row)
        self.assertIn("mean_auc", row["final_val_metrics"])
        self.assertIn("mean_auc", row["test_metrics"])
        self.assertIn("h3", row["test_metrics"]["horizons"])
        self.assertNotIn("final_test_mean_auc", row["metrics"])

    def test_internal_early_stopping_split_uses_train_tail_only(self):
        idx = pd.date_range("2024-01-01", periods=200, freq="15min")
        X = pd.DataFrame({"x": np.arange(200)}, index=idx)
        y = pd.Series(([0, 1] * 100), index=idx)

        split = _internal_early_stop_split(X, y)

        self.assertIsNotNone(split)
        X_fit, y_fit, X_stop, y_stop = split
        self.assertLess(X_fit.index.max(), X_stop.index.min())
        self.assertEqual(len(X_fit) + len(X_stop), len(X))
        self.assertEqual(y_fit.nunique(), 2)
        self.assertEqual(y_stop.nunique(), 2)

    def test_expression_space_and_mutator_generate_safe_formulas(self):
        df = _synthetic_crypto_frame(900)
        features = build_feature_frame(df, windows=[3, 5, 10, 20], min_valid_ratio=0.5)
        pool = selectable_features(features)
        feature_space = CryptoFeatureSpace(features, pool)
        train_index = features.index[:650]
        mutator = CryptoMutator(pool, feature_space, train_index, seed=21)
        individual = mutator.seed_individual()

        generated = None
        for _ in range(80):
            child = mutator.mutate(individual)
            new_features = [
                feature for feature in child.features if feature not in pool
            ]
            if new_features:
                generated = new_features[0]
                break

        self.assertIsNotNone(generated)
        quality = feature_space.quality(generated, train_index)
        self.assertTrue(quality.ok, quality.reason)
        matrix = feature_space.matrix([generated], train_index)
        self.assertEqual(len(matrix), len(train_index))
        self.assertFalse(set(child.features) & RAW_SCALE_COLUMNS)

    def test_expression_cache_is_bounded(self):
        index = pd.date_range("2024-01-01", periods=20, freq="15min")
        base = pd.DataFrame({"feature": np.arange(20, dtype=float)}, index=index)
        space = CryptoFeatureSpace(base, ["feature"])

        for value in range(config.EXPR_CACHE_MAX_ITEMS + 25):
            space.evaluate(f"const({value})")

        self.assertEqual(space.cache_size, config.EXPR_CACHE_MAX_ITEMS)
        space.clear_expression_cache()
        self.assertEqual(space.cache_size, 0)

    def test_rolling_rank_matches_legacy_average_rank(self):
        index = pd.date_range("2024-01-01", periods=12, freq="15min")
        series = pd.Series(
            [1.0, 2.0, 2.0, 4.0, np.nan, 3.0, 3.0, 1.0, 5.0, 5.0, 2.0, 4.0],
            index=index,
        )

        def legacy_rank_last(values: np.ndarray) -> float:
            last = values[-1]
            if np.isnan(last):
                return np.nan
            valid = values[~np.isnan(values)]
            less = np.sum(valid < last)
            equal = np.sum(valid == last)
            return float((less + ((equal + 1.0) / 2.0)) / len(valid))

        expected = series.rolling(4, min_periods=4).apply(legacy_rank_last, raw=True)
        actual = _rolling_rank_pct(series, 4)
        pd.testing.assert_series_equal(actual, expected, check_exact=True)

        space = CryptoFeatureSpace(pd.DataFrame({"feature": series}), ["feature"])
        expression_rank = space.evaluate("ts_rank(feature, 4)")
        pd.testing.assert_series_equal(
            expression_rank,
            expected.rename("feature"),
            check_exact=True,
        )

    def test_expression_matrix_fold_slice_matches_full_matrix_then_slice(self):
        index = pd.date_range("2024-01-01", periods=30, freq="15min")
        base = pd.DataFrame(
            {
                "left": np.linspace(-1.0, 1.0, len(index)),
                "right": np.arange(len(index), dtype=float),
            },
            index=index,
        )
        formulas = ["left", "ts_rank(right, 5)", "(left + right)"]
        fold_index = index[10:23]
        space = CryptoFeatureSpace(base, ["left", "right"])

        full_matrix = space.matrix(formulas)
        fold_matrix = space.matrix(formulas, fold_index)

        pd.testing.assert_frame_equal(
            fold_matrix,
            full_matrix.loc[fold_index],
            check_exact=True,
        )


def _synthetic_crypto_frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=n, freq="15min")
    returns = rng.normal(0.0001, 0.004, size=n)
    close = 10000.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.0005, size=n))
    spread = np.abs(rng.normal(0.002, 0.001, size=n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(mean=4.5, sigma=0.3, size=n)
    trade_count = rng.integers(100, 800, size=n)
    buy_ratio = np.clip(0.5 + rng.normal(0.0, 0.08, size=n), 0.05, 0.95)
    taker_base = volume * buy_ratio
    taker_quote = close * taker_base
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": trade_count,
            "taker_buy_base_volume": taker_base,
            "taker_buy_quote_volume": taker_quote,
        },
        index=idx,
    )


def _synthetic_feature_space(
    names: list[str],
    rows: int = 700,
) -> tuple[pd.Index, pd.DataFrame]:
    rng = np.random.default_rng(321)
    idx = pd.date_range("2024-01-01", periods=rows, freq="15min")
    data = {}
    for name in names:
        data[name] = rng.normal(0.0, 1.0, size=rows)
    return idx, pd.DataFrame(data, index=idx)


if __name__ == "__main__":
    unittest.main()
