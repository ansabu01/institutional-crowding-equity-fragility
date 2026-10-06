"""Check construction of the eligible manager holdings panel."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

import pandas as pd

from src.data.panel.holdings import build_sec_13f_crsp_holdings


class HoldingsPanelTests(unittest.TestCase):
    def test_filtering_aggregation_weights_and_audit(self):
        period = pd.Timestamp("2020-03-31")
        positions = pd.DataFrame(
            {
                "CIK": ["1", "1", "1", "1", "2"],
                "PERIODOFREPORT": ["2020-03-31"] * 5,
                "report_period": [period] * 5,
                "CUSIP": [
                    "111111111",
                    "222222222",
                    "333333333",
                    "444444444",
                    "111111111",
                ],
                "NAMEOFISSUER": ["A", "B", "C", "D", "A"],
                "TITLEOFCLASS": ["COM"] * 5,
                "position_value_usd": [100, 50, 25, 30, 200],
                "raw_position_rows": [1] * 5,
            }
        )
        mapping = pd.DataFrame(
            [
                self._mapping_row("111111111", "MATCHED", 1, 10001),
                self._mapping_row("222222222", "MATCHED", 1, 10001),
                self._mapping_row("333333333", "UNMATCHED", 0, None),
                self._mapping_row(
                    "444444444",
                    "MATCHED",
                    1,
                    40004,
                    usincflg="N",
                ),
            ]
        )
        selected = pd.DataFrame(
            {
                "CIK": ["1", "2"],
                "PERIODOFREPORT": ["2020-03-31"] * 2,
                "report_period": [period] * 2,
                "FILINGMANAGER_NAME": ["Manager 1", "Manager 2"],
                "filing_deadline": [pd.Timestamp("2020-05-15")] * 2,
                "selection_decision": ["SELECTED_BASE"] * 2,
                "is_selected": [True] * 2,
            }
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            positions_path = root / "positions.parquet"
            mapping_path = root / "mapping.parquet"
            selected_path = root / "selected.parquet"
            output = root / "output"
            positions.to_parquet(positions_path, index=False)
            mapping.to_parquet(mapping_path, index=False)
            selected.to_parquet(selected_path, index=False)

            result = build_sec_13f_crsp_holdings(
                positions_path,
                mapping_path,
                selected_path,
                output,
            )
            existing = build_sec_13f_crsp_holdings(
                positions_path,
                mapping_path,
                selected_path,
                output,
            )
            holdings = pd.read_parquet(
                output / "manager_security_holdings.parquet"
            )
            totals = pd.read_parquet(
                output / "manager_portfolio_totals.parquet"
            )
            audit = pd.read_parquet(
                output / "sample_construction_audit.parquet"
            )
            output_mtime = min(
                path.stat().st_mtime_ns for path in output.glob("*.parquet")
            )
            os.utime(
                mapping_path,
                ns=(output_mtime + 1_000_000, output_mtime + 1_000_000),
            )

            with self.assertRaisesRegex(RuntimeError, "newer"):
                build_sec_13f_crsp_holdings(
                    positions_path,
                    mapping_path,
                    selected_path,
                    output,
                )

        self.assertTrue(result.created)
        self.assertFalse(existing.created)
        self.assertEqual(len(holdings), 2)
        self.assertEqual(
            holdings.set_index("CIK")["position_value_usd"].to_dict(),
            {"1": 150, "2": 200},
        )
        self.assertTrue(holdings["portfolio_weight"].eq(1.0).all())
        self.assertEqual(len(totals), 2)
        self.assertEqual(
            audit.set_index("stage")["manager_positions"].to_dict(),
            {
                "FORM_13F_POSITIONS": 5,
                "EXACT_CRSP_MATCHES": 4,
                "ELIGIBLE_US_COMMON_EQUITIES": 3,
            },
        )

    @staticmethod
    def _mapping_row(
        cusip: str,
        status: str,
        candidate_count: int,
        permno: int | None,
        *,
        usincflg: str = "Y",
    ) -> dict[str, object]:
        return {
            "PERIODOFREPORT": "2020-03-31",
            "report_period": pd.Timestamp("2020-03-31"),
            "CUSIP": cusip,
            "match_status": status,
            "candidate_permno_count": candidate_count,
            "permno": permno,
            "permco": permno,
            "crsp_cusip9": cusip if permno is not None else None,
            "ticker": "TEST",
            "securitynm": "Test security",
            "primaryexch": "N",
            "sharetype": "NS",
            "securitytype": "EQTY",
            "securitysubtype": "COM",
            "usincflg": usincflg,
            "issuertype": "CORP",
        }


if __name__ == "__main__":
    unittest.main()
