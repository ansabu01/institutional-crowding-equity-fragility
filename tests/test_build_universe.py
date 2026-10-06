"""Check SEC universe construction using small synthetic filings."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data.sec_13f.build_universe import build_13f_position_universe


class UniverseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archives = {name: self.root / name for name in ("x", "y")}
        for directory in self.archives.values():
            directory.mkdir()
        self.selected_path = self.root / "selected.parquet"
        self.output = self.root / "universe"
        filings = []
        for accession in "abcdefghi":
            filings.append(
                {
                    "ACCESSION_NUMBER": accession,
                    "source_archive_id": "y" if accession == "b" else "x",
                    "CIK": "a" if accession == "b" else accession,
                    "PERIODOFREPORT": "31-MAR-2023"
                    if accession == "e"
                    else "31-MAR-2022",
                    "FILINGMANAGER_NAME": accession,
                    "filing_date": pd.Timestamp(
                        "2023-05-01" if accession == "e" else "2022-05-01"
                    ),
                    "REPORTTYPE": "13F COMBINATION REPORT"
                    if accession == "h"
                    else "13F HOLDINGS REPORT",
                    "selection_decision": "SELECTED_ADDITION"
                    if accession == "b"
                    else "SELECTED_BASE",
                    "is_selected": True,
                }
            )
        self.filings = pd.DataFrame(filings)
        self.filings.to_parquet(self.selected_path, index=False)
        self.rows = []

        def position(accession, value, cusip="111111111", shares="SH", option=None):
            self.rows.append(
                {
                    "ACCESSION_NUMBER": accession,
                    "NAMEOFISSUER": "Issuer",
                    "TITLEOFCLASS": "COM",
                    "CUSIP": cusip,
                    "VALUE": value,
                    "SSHPRNAMTTYPE": shares,
                    "PUTCALL": option,
                }
            )

        position("a", "10")
        position("a", "20")
        position("a", "5", option="CALL")
        position("a", "7", shares="PRN")
        position("a", "8", cusip="short")
        position("a", "8", cusip="12345678 ")
        position("a", "6", cusip=" abcdefghi ")
        position("a", "0")
        position("a", "-1")
        position("a", "bad")
        for accession, value in (
            ("b", "4"),
            ("c", "2000"),
            ("d", "7"),
            ("e", "9"),
            ("f", "3"),
            ("h", "100"),
            ("i", "10"),
        ):
            position(accession, value)
        position("i", "bad")
        positions = pd.DataFrame(self.rows, dtype="string")
        # The wrong archive must not contribute positions to the selected addition.
        wrong_archive = positions.loc[positions.ACCESSION_NUMBER.eq("b")].copy()
        wrong_archive["VALUE"] = "999"
        for archive_id, directory in self.archives.items():
            part = positions.loc[
                positions.ACCESSION_NUMBER.eq("b")
                if archive_id == "y"
                else ~positions.ACCESSION_NUMBER.eq("b")
            ]
            if archive_id == "x":
                part = pd.concat([part, wrong_archive], ignore_index=True)
            part.to_parquet(directory / "infotable.parquet", index=False)
        summaries = pd.DataFrame(
            [
                {
                    "ACCESSION_NUMBER": accession,
                    "TABLEENTRYTOTAL": count,
                    "TABLEVALUETOTAL": value,
                }
                for accession, count, value in (
                    ("a", "8", "49"),
                    ("b", "1", "4"),
                    ("c", "1", "2"),
                    ("d", "2", "8"),
                    ("e", "1", "9"),
                    ("g", "0", "0"),
                    ("h", "1", "100"),
                    ("i", "2", "10"),
                )
            ],
            dtype="string",
        )
        for archive_id, directory in self.archives.items():
            summaries.loc[
                summaries.ACCESSION_NUMBER.eq("b")
                if archive_id == "y"
                else ~summaries.ACCESSION_NUMBER.eq("b")
            ].to_parquet(directory / "summarypage.parquet", index=False)

    def build(self, overwrite=False):
        return build_13f_position_universe(
            self.selected_path,
            self.archives,
            self.output,
            overwrite=overwrite,
        )

    def test_positions_units_and_audits(self):
        result = self.build()
        self.assertTrue(result.created)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {
                "manager_security_positions.parquet",
                "position_filter_audit.parquet",
                "filing_audit.parquet",
            },
        )
        positions = pd.read_parquet(self.output / "manager_security_positions.parquet")
        self.assertEqual(
            positions.groupby("CIK").position_value_usd.sum().to_dict(),
            {"a": 40000, "c": 2000, "d": 7000, "e": 9, "f": 3000, "i": 10000},
        )
        a_positions = positions.loc[positions["CIK"] == "a"]
        self.assertEqual(a_positions["raw_position_rows"].sum(), 4)
        self.assertIn("ABCDEFGHI", set(a_positions["CUSIP"]))
        self.assertNotIn("12345678", set(a_positions["CUSIP"]))
        audit = pd.read_parquet(self.output / "filing_audit.parquet").set_index(
            "ACCESSION_NUMBER"
        )
        self.assertEqual(len(audit), 9)
        self.assertFalse(audit.loc["f", "has_summary_row"])
        self.assertTrue(pd.isna(audit.loc["f", "position_count_matches_summary"]))
        self.assertEqual(audit.loc["g", "observed_position_rows"], 0)
        self.assertTrue(audit.loc["g", "position_count_matches_summary"])
        self.assertFalse(audit.loc["d", "position_count_matches_summary"])
        self.assertEqual(audit.loc["c", "position_value_multiplier_to_usd"], 1)
        self.assertEqual(
            audit.loc["d", "value_unit_classification"], "PRE_2023_UNRESOLVED"
        )
        self.assertTrue(pd.isna(audit.loc["i", "value_matches_summary"]))
        filters = pd.read_parquet(self.output / "position_filter_audit.parquet")
        last = filters.loc[filters.stage.eq("Positive-value baseline rows")]
        self.assertEqual(last.rows.sum(), positions.raw_position_rows.sum())
        self.assertEqual(last.value_usd.sum(), positions.position_value_usd.sum())

    def test_reuse_and_explicit_rebuild(self):
        self.build()
        self.assertFalse(self.build().created)
        output = self.output / "filing_audit.parquet"
        newer = output.stat().st_mtime_ns + 1_000_000_000
        os.utime(self.selected_path, ns=(newer, newer))
        with self.assertRaisesRegex(RuntimeError, "newer"):
            self.build()
        self.assertTrue(self.build(overwrite=True).created)

    def test_partial_outputs_require_overwrite(self):
        self.output.mkdir()
        pd.DataFrame({"old": [1]}).to_parquet(
            self.output / "manager_security_positions.parquet",
            index=False,
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            self.build()
        self.assertTrue(self.build(overwrite=True).created)

    def test_legacy_summary_columns_are_not_used(self):
        self.filings["TABLEVALUETOTAL"] = "999999"
        self.filings["TABLEENTRYTOTAL"] = "999999"
        self.filings.to_parquet(self.selected_path, index=False)
        self.build()
        audit = pd.read_parquet(self.output / "filing_audit.parquet").set_index(
            "ACCESSION_NUMBER"
        )
        self.assertEqual(audit.loc["c", "declared_value_usd"], 2000)

    def test_addition_requires_base(self):
        self.filings.loc[self.filings.ACCESSION_NUMBER.eq("b"), "CIK"] = "orphan"
        self.filings.to_parquet(self.selected_path, index=False)
        with self.assertRaisesRegex(RuntimeError, "one selected base"):
            self.build()

    def test_missing_summary_file_and_duplicate_summary_stop(self):
        summary_path = self.archives["x"] / "summarypage.parquet"
        summaries = pd.read_parquet(summary_path)
        pd.concat([summaries, summaries.iloc[:1]], ignore_index=True).to_parquet(
            summary_path,
            index=False,
        )
        with self.assertRaisesRegex(RuntimeError, "unique"):
            self.build()
        summary_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()


if __name__ == "__main__":
    unittest.main()
