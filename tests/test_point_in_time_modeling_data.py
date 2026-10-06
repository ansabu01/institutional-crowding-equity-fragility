"""Check point-in-time feature, target, panel, and split construction."""

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.data.panel.modeling import build_stock_quarter_modeling_panel
from src.features.market_features import build_crsp_market_features
from src.features.targets import build_downside_targets
from src.modeling.build_splits import build_modeling_splits


class PointInTimeModelingDataTests(unittest.TestCase):
    def test_feature_target_and_panel_windows_do_not_leak(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.bdate_range("2019-01-02", periods=380)
            information_date = dates[270]
            report_period = pd.Timestamp("2019-12-31")
            market_returns = 0.001 * np.sin(np.arange(len(dates)) / 7)
            stock_returns = 0.0002 + 0.8 * market_returns
            daily_path = root / "daily_stock.parquet"
            market_path = root / "daily_market_index.parquet"
            security_path = root / "security_nodes.parquet"
            network_path = root / "network_summary.parquet"

            pd.DataFrame(
                {
                    "permno": 10001,
                    "dlycaldt": dates,
                    "conditionaltype": "RW",
                    "tradingstatusflg": "A",
                    "dlyprc": 10.0,
                    "dlycap": 1_000.0,
                    "dlyret": stock_returns,
                    "dlyvol": 1_000.0,
                    "dlydelflg": "N",
                }
            ).to_parquet(daily_path, index=False)
            pd.DataFrame(
                {"dlycaldt": dates, "dlytotret": market_returns}
            ).to_parquet(market_path, index=False)
            pd.DataFrame(
                {
                    "PERIODOFREPORT": ["2019Q4"],
                    "report_period": [report_period],
                    "information_date": [information_date],
                    "permno": [10001],
                    "permco": [1000],
                    "CUSIP": ["12345678"],
                    "ticker": ["TEST"],
                    "securitynm": ["Test Security"],
                    "primaryexch": ["N"],
                    "issuertype": ["CORP"],
                    "institutional_holder_count": [2],
                    "total_reported_value_usd": [1_000_000],
                    "reported_ownership_hhi": [0.5],
                    "top_five_reported_ownership_share": [1.0],
                    "mean_owner_portfolio_weight": [0.1],
                    "maximum_owner_portfolio_weight": [0.15],
                }
            ).to_parquet(security_path, index=False)
            pd.DataFrame(
                {
                    "report_period": [report_period],
                    "managers": [10],
                    "securities": [1],
                    "ownership_edges": [2],
                    "bipartite_density": [0.2],
                }
            ).to_parquet(network_path, index=False)

            crsp_directory = root / "crsp"
            feature_result = build_crsp_market_features(
                daily_path,
                market_path,
                security_path,
                crsp_directory,
            )
            target_result = build_downside_targets(
                daily_path,
                security_path,
                crsp_directory,
            )
            panel_result = build_stock_quarter_modeling_panel(
                security_path,
                network_path,
                crsp_directory / "market_features.parquet",
                crsp_directory / "downside_targets.parquet",
                root / "modeling",
            )

            features = pd.read_parquet(
                crsp_directory / "market_features.parquet"
            ).iloc[0]
            targets = pd.read_parquet(
                crsp_directory / "downside_targets.parquet"
            ).iloc[0]
            panel = pd.read_parquet(
                root / "modeling" / "stock_quarter_panel.parquet"
            ).iloc[0]

        self.assertTrue(feature_result.created)
        self.assertTrue(target_result.created)
        self.assertTrue(panel_result.created)
        self.assertEqual(features.feature_window_start, dates[18])
        self.assertEqual(features.feature_window_end, dates[269])
        self.assertEqual(features.return_observations_252d, 252)
        self.assertTrue(features.market_feature_usable)
        self.assertEqual(targets.target_window_start, dates[271])
        self.assertEqual(targets.target_window_end, dates[333])
        self.assertEqual(targets.return_observations_63d, 63)
        self.assertTrue(targets.target_usable)
        self.assertLess(panel.feature_window_end, panel.information_date)
        self.assertGreater(panel.target_window_start, panel.information_date)
        self.assertTrue(panel.model_eligible)

    def test_splits_enforce_the_training_start_and_purge_gaps(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            panel_path = root / "panel.parquet"
            output_directory = root / "output"
            pd.DataFrame(
                {
                    "report_period": pd.to_datetime(
                        [
                            "2013-03-31",
                            "2021-09-30",
                            "2021-12-31",
                            "2022-03-31",
                            "2023-09-30",
                            "2023-12-31",
                            "2024-03-31",
                        ]
                    ),
                    "information_date": pd.to_datetime(
                        [
                            "2013-05-15",
                            "2021-11-15",
                            "2022-02-14",
                            "2022-05-16",
                            "2023-11-14",
                            "2024-02-14",
                            "2024-05-15",
                        ]
                    ),
                    "target_window_end": pd.to_datetime(
                        [
                            "2013-08-14",
                            "2022-02-15",
                            "2022-05-13",
                            "2022-08-15",
                            "2024-02-15",
                            "2024-05-14",
                            "2024-08-14",
                        ]
                    ),
                    "permno": np.arange(10001, 10008),
                    "model_eligible": True,
                }
            ).to_parquet(panel_path, index=False)

            result = build_modeling_splits(
                panel_path,
                output_directory,
                train_start_date=date(2013, 6, 30),
                train_end_date=date(2021, 9, 30),
                validation_start_date=date(2022, 3, 31),
                validation_end_date=date(2023, 9, 30),
                test_start_date=date(2024, 3, 31),
                test_end_date=date(2025, 12, 31),
            )
            sample = pd.read_parquet(result.sample_path)

        self.assertEqual(result.sample_rows, 4)
        self.assertEqual(result.purged_quarters, 2)
        self.assertEqual(result.purged_security_periods, 2)
        self.assertEqual(
            sample.sample_split.tolist(),
            ["train", "validation", "validation", "test"],
        )
        self.assertEqual(sample.report_period.min(), pd.Timestamp("2021-09-30"))


if __name__ == "__main__":
    unittest.main()
