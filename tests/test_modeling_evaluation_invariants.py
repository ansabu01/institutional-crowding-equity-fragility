"""Check the central modeling and economic-evaluation invariants."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.evaluation.fragility import (
    _build_stock_quarters,
    _portfolio_metrics,
)
from src.evaluation.robustness import (
    _build_portfolios,
    _load_inputs,
)
from src.modeling.feature_sets import MARKET_FEATURES
from src.modeling.robustness import _target_samples
from src.modeling.tabular_selection import prediction_table


class ModelingEvaluationInvariantTests(unittest.TestCase):
    def test_fragility_scores_and_deciles_follow_prediction_order(self):
        period = pd.Timestamp("2024-03-31")
        information_date = pd.Timestamp("2024-05-15")
        sample = pd.DataFrame(
            {
                "report_period": period,
                "information_date": information_date,
                "permno": np.arange(10),
                "future_max_drawdown_63d": np.linspace(0.01, 0.10, 10),
            }
        )
        predictions = prediction_table(
            sample,
            np.linspace(0.01, 0.10, 10),
        )
        singleton = prediction_table(
            sample.iloc[:1],
            np.array([0.05]),
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "predictions.parquet"
            sample_path = root / "sample.parquet"
            predictions.to_parquet(predictions_path, index=False)
            sample.assign(
                target_window_start=pd.Timestamp("2024-05-16"),
                target_window_end=pd.Timestamp("2024-08-14"),
                sample_split="test",
                future_cumulative_return_63d=0.0,
                future_downside_volatility_63d=0.1,
                future_worst_five_day_return_63d=-0.05,
            ).to_parquet(sample_path, index=False)
            panel = _build_stock_quarters(predictions_path, sample_path)

        self.assertEqual(singleton.loc[0, "fragility_score"], 50)
        self.assertEqual(panel.loc[panel["permno"] == 0, "fragility_decile"].item(), 1)
        self.assertEqual(panel.loc[panel["permno"] == 9, "fragility_decile"].item(), 10)

    def test_buy_and_hold_keeps_delisting_proceeds_in_cash(self):
        returns = pd.DataFrame(
            {
                1: [0.10, np.nan, np.nan, np.nan, np.nan],
                2: [0.00, 0.00, 0.00, 0.00, 0.00],
            }
        )
        metrics = _portfolio_metrics(returns, pd.Index([1, 2]))

        self.assertAlmostEqual(metrics["cumulative_return"], 0.05)
        self.assertAlmostEqual(metrics["maximum_drawdown"], 0.0)

    def test_defensive_factor_directions_and_breadth(self):
        period = pd.Timestamp("2024-03-31")
        dates = pd.date_range("2024-05-16", periods=5, freq="B")
        signals = pd.DataFrame(
            {
                "report_period": [period] * 10,
                "permno": range(1, 11),
                "fragility_decile": range(1, 11),
                "market_cap_usd": [900, 100, 200, 300, 400, 500, 600, 700, 800, 1000],
            }
        )
        primary = pd.DataFrame({"date": dates, "report_period": period})
        returns_by_permno = {1: 0.01, 2: 0.03, 9: -0.03, 10: -0.01}
        daily_stock = pd.DataFrame(
            {
                "permno": np.repeat(range(1, 11), len(dates)),
                "dlycaldt": list(dates) * 10,
                "dlyret": [
                    returns_by_permno.get(permno, 0.0)
                    for permno in range(1, 11)
                    for _ in dates
                ],
            }
        )

        daily, _ = _build_portfolios(signals, primary, daily_stock)

        self.assertTrue(
            np.allclose(daily["least_fragile_return"], 0.01)
        )
        self.assertTrue(
            np.allclose(daily["most_fragile_return"], -0.01)
        )
        self.assertTrue(
            np.allclose(daily["defensive_fragility_factor_return"], 0.02)
        )
        self.assertAlmostEqual(daily.loc[0, "least_fragile_20_return"], 0.02)
        self.assertAlmostEqual(daily.loc[0, "most_fragile_20_return"], -0.02)
        self.assertAlmostEqual(
            daily.loc[0, "defensive_fragility_factor_20_return"],
            0.04,
        )
        self.assertTrue(
            np.allclose(
                daily["defensive_fragility_factor_20_return"],
                daily["least_fragile_20_return"]
                - daily["most_fragile_20_return"],
            )
        )

    def test_robustness_targets_use_horizon_specific_purge_boundaries(self):
        periods = pd.to_datetime(["2021-09-30", "2023-09-30", "2024-03-31"])
        information_dates = pd.to_datetime(
            ["2021-11-15", "2023-11-14", "2024-05-15"]
        )
        sample = pd.DataFrame(
            {
                "report_period": periods,
                "information_date": information_dates,
                "target_window_end": pd.to_datetime(
                    ["2023-11-14", "2024-05-15", "2024-08-14"]
                ),
                "permno": [1, 2, 3],
                "sample_split": ["train", "validation", "test"],
                "future_downside_volatility_63d": [0.1, 0.1, 0.1],
                "future_worst_five_day_return_63d": [-0.1, -0.1, -0.1],
                **{feature: [1.0, 1.0, 1.0] for feature in MARKET_FEATURES},
            }
        )
        alternative_rows = []

        for horizon, end_dates in {
            21: ["2023-11-13", "2024-05-14", "2024-06-13"],
            126: ["2023-11-14", "2024-05-15", "2024-11-13"],
        }.items():
            for index in range(3):
                alternative_rows.append(
                    {
                        "report_period": periods[index],
                        "information_date": information_dates[index],
                        "permno": index + 1,
                        "horizon_market_days": horizon,
                        "target_window_end": pd.Timestamp(end_dates[index]),
                        "future_max_drawdown": 0.1,
                        "target_usable": True,
                    }
                )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.parquet"
            targets_path = root / "targets.parquet"
            sample.to_parquet(sample_path, index=False)
            pd.DataFrame(alternative_rows).to_parquet(targets_path, index=False)
            samples = _target_samples(
                sample_path,
                targets_path,
                MARKET_FEATURES,
            )

        self.assertEqual(len(samples["future_downside_volatility_63d"]), 1)
        self.assertEqual(len(samples["future_worst_five_day_loss_63d"]), 1)
        self.assertEqual(len(samples["future_max_drawdown_21d"]), 3)
        self.assertEqual(len(samples["future_max_drawdown_126d"]), 1)

    def test_robustness_loader_resets_sorted_market_index(self):
        dates = pd.date_range("2024-05-16", periods=10, freq="B")
        period = pd.Timestamp("2024-03-31")
        primary = pd.DataFrame(
            {
                "date": dates[::-1],
                "report_period": period,
                "all_stocks_return": 0.0,
                "screened_return": 0.0,
                "active_return": 0.0,
                "market_excess_return": 0.0,
                "smb": 0.0,
                "hml": 0.0,
                "rmw": 0.0,
                "cma": 0.0,
                "momentum": 0.0,
                "risk_free_rate": 0.0,
            }
        )
        signals = pd.DataFrame(
            {
                "report_period": [period],
                "information_date": [pd.Timestamp("2024-05-15")],
                "permno": [1],
                "fragility_decile": [1],
            }
        )
        sample = pd.DataFrame(
            {
                "report_period": [period],
                "permno": [1],
                "sample_split": ["test"],
                "market_cap_usd": [100.0],
            }
        )
        daily_stock = pd.DataFrame(
            {"permno": 1, "dlycaldt": dates, "dlyret": 0.0}
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"input_{index}.parquet" for index in range(4)]
            for frame, path in zip(
                (signals, sample, daily_stock, primary),
                paths,
                strict=True,
            ):
                frame.to_parquet(path, index=False)
            _, loaded_primary, _ = _load_inputs(*paths)

        self.assertTrue(loaded_primary.index.equals(pd.RangeIndex(10)))
        self.assertTrue(loaded_primary["date"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
