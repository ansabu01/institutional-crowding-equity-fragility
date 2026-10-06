"""Check construction of quarterly bipartite ownership networks."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

import pandas as pd

from src.network.build_ownership_network import build_ownership_networks


class OwnershipNetworkTests(unittest.TestCase):
    def test_nodes_and_network_summary(self):
        period = pd.Timestamp("2020-03-31")
        information_date = pd.Timestamp("2020-05-15")
        holdings = pd.DataFrame(
            [
                self._holding("1", 10001, 60, 100, 0.6),
                self._holding("1", 10002, 40, 100, 0.4),
                self._holding("2", 10001, 50, 50, 1.0),
            ]
        )
        totals = pd.DataFrame(
            {
                "CIK": ["1", "2"],
                "PERIODOFREPORT": ["2020-03-31"] * 2,
                "report_period": [period] * 2,
                "information_date": [information_date] * 2,
                "FILINGMANAGER_NAME": ["Manager 1", "Manager 2"],
                "portfolio_value_usd": [100, 50],
                "security_count": [2, 1],
                "underlying_position_rows": [2, 1],
            }
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            holdings_path = root / "holdings.parquet"
            totals_path = root / "totals.parquet"
            output = root / "output"
            holdings.to_parquet(holdings_path, index=False)
            totals.to_parquet(totals_path, index=False)

            created = build_ownership_networks(
                holdings_path,
                totals_path,
                output,
            )
            existing = build_ownership_networks(
                holdings_path,
                totals_path,
                output,
            )
            output_mtime = min(
                path.stat().st_mtime_ns for path in output.glob("*.parquet")
            )
            os.utime(
                totals_path,
                ns=(output_mtime + 1_000_000, output_mtime + 1_000_000),
            )

            with self.assertRaisesRegex(RuntimeError, "newer"):
                build_ownership_networks(
                    holdings_path,
                    totals_path,
                    output,
                )
            managers = pd.read_parquet(output / "manager_nodes.parquet")
            securities = pd.read_parquet(output / "security_nodes.parquet")
            summary = pd.read_parquet(output / "network_summary.parquet")

        self.assertTrue(created.created)
        self.assertFalse(existing.created)
        self.assertEqual([table.name for table in created.tables], [
            "manager_nodes",
            "security_nodes",
            "network_summary",
        ])
        self.assertEqual(created.report_periods, 1)
        manager_hhi = managers.set_index("CIK")["portfolio_hhi"]
        self.assertAlmostEqual(manager_hhi["1"], 0.52)
        self.assertAlmostEqual(manager_hhi["2"], 1.0)
        holder_counts = securities.set_index("permno")[
            "institutional_holder_count"
        ]
        self.assertEqual(holder_counts.to_dict(), {10001: 2, 10002: 1})
        self.assertEqual(summary.loc[0, "ownership_edges"], 3)
        self.assertAlmostEqual(summary.loc[0, "bipartite_density"], 0.75)
        self.assertEqual(summary.loc[0, "connected_components"], 1)
        self.assertAlmostEqual(summary.loc[0, "largest_component_share"], 1.0)

    @staticmethod
    def _holding(
        cik: str,
        permno: int,
        value: int,
        portfolio_value: int,
        weight: float,
    ) -> dict[str, object]:
        return {
            "CIK": cik,
            "PERIODOFREPORT": "2020-03-31",
            "report_period": pd.Timestamp("2020-03-31"),
            "information_date": pd.Timestamp("2020-05-15"),
            "permno": permno,
            "permco": permno,
            "CUSIP": str(permno),
            "ticker": f"T{permno}",
            "securitynm": f"Security {permno}",
            "primaryexch": "N",
            "issuertype": "CORP",
            "position_value_usd": value,
            "portfolio_value_usd": portfolio_value,
            "portfolio_weight": weight,
        }


if __name__ == "__main__":
    unittest.main()
