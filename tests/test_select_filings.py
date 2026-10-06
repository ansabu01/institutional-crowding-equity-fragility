"""Check point-in-time Form 13F filing selection."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data.sec_13f.select_filings import select_13f_filings


class FilingSelectionTests(unittest.TestCase):
    def test_selection_rules_and_saved_output_freshness(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_directory = root / "archive"
            archive_directory.mkdir()
            decisions_path = root / "output" / "decisions.parquet"
            selected_path = root / "output" / "selected.parquet"

            rows = [
                ("a", "1", "31-MAR-2024", "01-MAY-2024", "13F-HR", "N", None, None, "13F HOLDINGS REPORT"),
                ("b", "1", "31-MAR-2024", "10-MAY-2024", "13F-HR/A", "Y", "1", "RESTATEMENT", "13F HOLDINGS REPORT"),
                ("c", "1", "31-MAR-2024", "12-MAY-2024", "13F-HR/A", "Y", "2", "NEW HOLDINGS", "13F HOLDINGS REPORT"),
                ("d", "2", "31-MAR-2024", "01-MAY-2024", "13F-NT", "N", None, None, "13F NOTICE"),
                ("e", "3", "31-MAR-2024", "16-MAY-2024", "13F-HR", "N", None, None, "13F HOLDINGS REPORT"),
                ("f", "4", "31-MAR-2024", "01-MAY-2024", "13F-HR", "N", None, None, "13F NOTICE"),
                ("g", "5", "31-MAR-2024", "01-MAY-2024", "13F-HR/A", "Y", "1", "NEW HOLDINGS", "13F HOLDINGS REPORT"),
                ("h", "6", "31-DEC-2023", "01-FEB-2024", "13F-HR", "N", None, None, "13F HOLDINGS REPORT"),
                ("i", "7", "31-DEC-2025", "17-FEB-2026", "13F-HR", "N", None, None, "13F HOLDINGS REPORT"),
            ]
            submissions = pd.DataFrame(
                [
                    {
                        "ACCESSION_NUMBER": accession,
                        "FILING_DATE": filing_date,
                        "SUBMISSIONTYPE": submission_type,
                        "CIK": cik,
                        "PERIODOFREPORT": report_period,
                    }
                    for accession, cik, report_period, filing_date,
                    submission_type, *_ in rows
                ],
                dtype="string",
            )
            coverpages = pd.DataFrame(
                [
                    {
                        "ACCESSION_NUMBER": accession,
                        "ISAMENDMENT": is_amendment,
                        "AMENDMENTNO": amendment_number,
                        "AMENDMENTTYPE": amendment_type,
                        "FILINGMANAGER_NAME": f"Manager {cik}",
                        "REPORTTYPE": report_type,
                    }
                    for accession, cik, _, _, _, is_amendment,
                    amendment_number, amendment_type, report_type in rows
                ],
                dtype="string",
            )
            submission_path = archive_directory / "submission.parquet"
            submissions.to_parquet(submission_path, index=False)
            coverpages.to_parquet(
                archive_directory / "coverpage.parquet",
                index=False,
            )

            result = select_13f_filings(
                {"archive": archive_directory},
                decisions_path,
                selected_path,
                report_period_start_date=pd.Timestamp("2024-03-31").date(),
                report_period_end_date=pd.Timestamp("2025-12-31").date(),
            )
            decisions = result.filing_decisions.set_index("ACCESSION_NUMBER")

            self.assertEqual(
                decisions["selection_decision"].to_dict(),
                {
                    "h": "OUTSIDE_SAMPLE_PERIOD",
                    "a": "SUPERSEDED",
                    "b": "SELECTED_BASE",
                    "c": "SELECTED_ADDITION",
                    "d": "NOTICE_EXCLUDED",
                    "e": "FILED_AFTER_DEADLINE",
                    "f": "CLASSIFICATION_DISAGREEMENT",
                    "g": "UNRESOLVED",
                    "i": "SELECTED_BASE",
                },
            )
            self.assertEqual(set(result.selected_filings["ACCESSION_NUMBER"]), {"b", "c", "i"})
            self.assertEqual(
                decisions.loc["i", "filing_deadline"],
                pd.Timestamp("2026-02-17"),
            )

            with self.assertRaisesRegex(RuntimeError, "report-period range"):
                select_13f_filings(
                    {"archive": archive_directory},
                    decisions_path,
                    selected_path,
                    report_period_start_date=pd.Timestamp("2023-12-31").date(),
                    report_period_end_date=pd.Timestamp("2025-12-31").date(),
                )

            newer = min(
                decisions_path.stat().st_mtime_ns,
                selected_path.stat().st_mtime_ns,
            ) + 1_000_000_000
            os.utime(submission_path, ns=(newer, newer))

            with self.assertRaisesRegex(RuntimeError, "newer"):
                select_13f_filings(
                    {"archive": archive_directory},
                    decisions_path,
                    selected_path,
                    report_period_start_date=pd.Timestamp("2024-03-31").date(),
                    report_period_end_date=pd.Timestamp("2025-12-31").date(),
                )


if __name__ == "__main__":
    unittest.main()
