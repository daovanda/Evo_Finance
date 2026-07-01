import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mutator.gene import Individual
from prod import index as prod_index
from prod import predict as prod_predict
from prod.common import (
    SelectedIndividual,
    build_backtest_trades,
    build_signal_table,
    load_selected_individuals,
    prediction_frame_for_model,
)


def _selected(model_id: str, rank: int = 1) -> SelectedIndividual:
    return SelectedIndividual(
        model_id=model_id,
        archive_path=Path(f"{model_id}.json"),
        archive_name=f"{model_id}.json",
        rank=rank,
        individual=Individual.seed(),
    )


class ProdCommonTests(unittest.TestCase):
    def test_load_selected_individuals_uses_archive_file_and_rank(self):
        rows = [
            {"rank": 1, "genes": ["ret(close_1, 5)"]},
            {"rank": 3, "genes": [" volume_ratio(20) ", "volume_ratio(20)"]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp)
            archive_path = results_dir / "archive.json"
            archive_path.write_text(json.dumps(rows), encoding="utf-8")

            selected = load_selected_individuals(
                [{"archive": "archive.json", "rank": 3}],
                results_dir=results_dir,
            )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].rank, 3)
        self.assertEqual(selected[0].individual.formulas, ["volume_ratio(20)"])

    def test_signal_table_uses_single_model_top_k_directly(self):
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-01")], ["AAA", "BBB", "CCC"]],
            names=["date", "ticker"],
        )
        pred = pd.Series([0.3, 0.2, 0.1], index=idx)
        frame = prediction_frame_for_model(_selected("m1"), pred, split="val")

        signals = build_signal_table([frame], top_k=2)
        selected = signals[signals["selected"]].drop_duplicates(
            ["signal_date", "ticker"]
        )

        self.assertEqual(selected["ticker"].tolist(), ["AAA", "BBB"])
        self.assertTrue((selected["vote_count"] == 1).all())

    def test_signal_table_votes_overlapping_top_k_across_models(self):
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-01")], ["AAA", "BBB", "CCC"]],
            names=["date", "ticker"],
        )
        pred_1 = pd.Series([0.3, 0.2, 0.1], index=idx)
        pred_2 = pd.Series([0.1, 0.3, 0.2], index=idx)
        frame_1 = prediction_frame_for_model(_selected("m1"), pred_1, split="test")
        frame_2 = prediction_frame_for_model(_selected("m2"), pred_2, split="test")

        signals = build_signal_table([frame_1, frame_2], top_k=2, min_votes=2)
        selected = signals[signals["selected"]].drop_duplicates(
            ["signal_date", "ticker"]
        )

        self.assertEqual(selected["ticker"].tolist(), ["BBB"])
        self.assertEqual(selected["vote_count"].tolist(), [2])

    def test_signal_table_requires_all_models_by_default(self):
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-01")], ["AAA", "BBB", "CCC", "DDD"]],
            names=["date", "ticker"],
        )
        pred_1 = pd.Series([0.4, 0.3, 0.2, 0.1], index=idx)  # AAA, BBB
        pred_2 = pd.Series([0.4, 0.1, 0.3, 0.2], index=idx)  # AAA, CCC
        pred_3 = pd.Series([0.4, 0.1, 0.2, 0.3], index=idx)  # AAA, DDD
        frames = [
            prediction_frame_for_model(_selected("m1"), pred_1, split="test"),
            prediction_frame_for_model(_selected("m2"), pred_2, split="test"),
            prediction_frame_for_model(_selected("m3"), pred_3, split="test"),
        ]

        signals = build_signal_table(frames, top_k=2, min_votes=2)
        selected = signals[signals["selected"]].drop_duplicates(
            ["signal_date", "ticker"]
        )

        self.assertEqual(selected["ticker"].tolist(), ["AAA"])
        self.assertEqual(selected["vote_count"].tolist(), [3])
        self.assertEqual(selected["vote_threshold"].tolist(), [3])
        self.assertLessEqual(len(selected), 2)

    def test_signal_table_can_use_min_votes_when_all_models_disabled(self):
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-01")], ["AAA", "BBB", "CCC", "DDD"]],
            names=["date", "ticker"],
        )
        pred_1 = pd.Series([0.4, 0.3, 0.2, 0.1], index=idx)
        pred_2 = pd.Series([0.4, 0.3, 0.1, 0.2], index=idx)
        pred_3 = pd.Series([0.1, 0.2, 0.4, 0.3], index=idx)
        frames = [
            prediction_frame_for_model(_selected("m1"), pred_1, split="test"),
            prediction_frame_for_model(_selected("m2"), pred_2, split="test"),
            prediction_frame_for_model(_selected("m3"), pred_3, split="test"),
        ]

        signals = build_signal_table(
            frames,
            top_k=2,
            min_votes=2,
            require_all_models=False,
        )
        selected = signals[signals["selected"]].drop_duplicates(
            ["signal_date", "ticker"]
        )

        self.assertEqual(selected["ticker"].tolist(), ["AAA", "BBB"])
        self.assertEqual(selected["vote_threshold"].tolist(), [2, 2])

    def test_backtest_uses_next_open_and_horizon_close_equal_weighted(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        idx = pd.MultiIndex.from_product(
            [dates, ["AAA", "BBB"]],
            names=["date", "ticker"],
        )
        price = pd.DataFrame(index=idx)
        price["open"] = 10.0
        price["close"] = 10.0
        price.loc[(dates[1], "AAA"), "open"] = 10.0
        price.loc[(dates[1], "BBB"), "open"] = 20.0
        price.loc[(dates[2], "AAA"), "close"] = 11.0
        price.loc[(dates[2], "BBB"), "close"] = 18.0

        signals = pd.DataFrame(
            {
                "signal_date": [dates[0], dates[0]],
                "ticker": ["AAA", "BBB"],
                "selected": [True, True],
            }
        )

        trades = build_backtest_trades(signals, price, holding_horizon=2)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.loc[0, "entry_date"], dates[1].date().isoformat())
        self.assertEqual(trades.loc[0, "exit_date"], dates[2].date().isoformat())
        self.assertEqual(trades.loc[0, "tickers"], "AAA;BBB")
        self.assertEqual(trades.loc[0, "n_tickers"], 2)
        self.assertAlmostEqual(trades.loc[0, "ret"], 0.0)


class ProdIndexTests(unittest.TestCase):
    def test_index_runs_predict_then_backtest_with_same_config(self):
        calls = []

        def fake_predict(config_module):
            calls.append(("predict", config_module))
            return pd.DataFrame({"x": [1, 2]})

        def fake_backtest(config_module):
            calls.append(("backtest", config_module))
            return pd.DataFrame({"ret": [0.1]})

        with (
            patch.object(prod_index, "run_predict", side_effect=fake_predict),
            patch.object(prod_index, "run_backtest", side_effect=fake_backtest),
        ):
            predictions, trades = prod_index.run("prod.config")

        self.assertEqual(calls, [("predict", "prod.config"), ("backtest", "prod.config")])
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(trades), 1)


class ProdPredictMetricTests(unittest.TestCase):
    def test_mean_ic_uses_daily_spearman_ic(self):
        idx = pd.MultiIndex.from_product(
            [[pd.Timestamp("2024-01-01")], ["AAA", "BBB", "CCC", "DDD", "EEE"]],
            names=["date", "ticker"],
        )
        pred = pd.Series([5, 4, 3, 2, 1], index=idx)
        label = pd.Series([50, 40, 30, 20, 10], index=idx)

        mean_ic, n_dates = prod_predict._mean_ic(pred, label, idx)

        self.assertEqual(n_dates, 1)
        self.assertAlmostEqual(mean_ic, 1.0)


if __name__ == "__main__":
    unittest.main()
