"""Check point-in-time SEC Form 13F-to-CRSP security mapping."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest

import pandas as pd

from src.data.crsp.map_securities import build_sec_13f_crsp_mapping


class SecurityMappingTests(unittest.TestCase):
    def test_match_states_and_reuse(self):
        period = pd.Timestamp("2020-03-31")
        positions = pd.DataFrame(
            {
                "CIK": ["1", "2", "1", "1"],
                "PERIODOFREPORT": ["2020-03-31"] * 4,
                "report_period": [period] * 4,
                "CUSIP": [
                    "111111111",
                    "111111111",
                    "222222222",
                    "333333333",
                ],
                "position_value_usd": [100, 200, 50, 75],
            }
        )
        history = pd.DataFrame(
            [
                self._history_row(10001, "111111111"),
                self._history_row(30001, "333333333"),
                self._history_row(30002, "333333333"),
            ]
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            positions_path = root / "positions.parquet"
            history_path = root / "history.parquet"
            output_directory = root / "output"
            positions.to_parquet(positions_path, index=False)
            history.to_parquet(history_path, index=False)

            created = build_sec_13f_crsp_mapping(
                positions_path,
                history_path,
                output_directory,
            )
            existing = build_sec_13f_crsp_mapping(
                positions_path,
                history_path,
                output_directory,
            )
            output_mtime = min(
                (output_directory / "security_mapping.parquet").stat().st_mtime_ns,
                (output_directory / "security_mapping_audit.parquet").stat().st_mtime_ns,
            )
            os.utime(
                positions_path,
                ns=(output_mtime + 1_000_000, output_mtime + 1_000_000),
            )

            with self.assertRaisesRegex(RuntimeError, "newer"):
                build_sec_13f_crsp_mapping(
                    positions_path,
                    history_path,
                    output_directory,
                )

        self.assertTrue(created.created)
        self.assertEqual(
            (
                created.mapping_rows,
                created.matched_rows,
                created.unmatched_rows,
                created.ambiguous_rows,
                created.audit_rows,
            ),
            (3, 1, 1, 1, 3),
        )
        self.assertFalse(existing.created)
        self.assertEqual(existing.mapping_rows, 3)

    @staticmethod
    def _history_row(permno: int, cusip: str) -> dict[str, object]:
        return {
            "permno": permno,
            "permco": permno,
            "secinfostartdt": pd.Timestamp("2019-01-01"),
            "secinfoenddt": pd.Timestamp("2021-12-31"),
            "cusip9": cusip,
            "ticker": f"T{permno}",
            "securitynm": f"Security {permno}",
            "primaryexch": "N",
            "sharetype": "NS",
            "securitytype": "EQTY",
            "securitysubtype": "COM",
            "usincflg": "Y",
            "issuertype": "CORP",
        }


if __name__ == "__main__":
    unittest.main()
