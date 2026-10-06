"""Check SEC archive extraction and reuse validation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

import pandas as pd

from src.data.sec_13f.extract import extract_sec_13f_tables


_TABLE_NAMES = (
    "COVERPAGE.tsv",
    "INFOTABLE.tsv",
    "OTHERMANAGER.tsv",
    "OTHERMANAGER2.tsv",
    "SIGNATURE.tsv",
    "SUBMISSION.tsv",
    "SUMMARYPAGE.tsv",
)


class Sec13FExtractionTests(unittest.TestCase):
    def test_extract_and_validate_existing_outputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "archive.zip"
            output_directory = root / "tables"

            with ZipFile(archive_path, "w") as archive:
                for table_name in _TABLE_NAMES:
                    archive.writestr(table_name, "ID\tVALUE\n001\t10\n")
                archive.writestr(
                    "FORM13F_metadata.json",
                    json.dumps({"version": 1}),
                )

            created = extract_sec_13f_tables(
                archive_path,
                output_directory,
                chunk_size=1,
            )
            reused = extract_sec_13f_tables(archive_path, output_directory)

            self.assertTrue(all(table.created for table in created.tables))
            self.assertTrue(all(not table.created for table in reused.tables))
            self.assertEqual(
                pd.read_parquet(output_directory / "coverpage.parquet").to_dict(
                    orient="records"
                ),
                [{"ID": "001", "VALUE": "10"}],
            )

            pd.DataFrame({"WRONG": ["value"]}).to_parquet(
                output_directory / "coverpage.parquet",
                index=False,
            )

            with self.assertRaisesRegex(RuntimeError, "schema does not match"):
                extract_sec_13f_tables(archive_path, output_directory)


if __name__ == "__main__":
    unittest.main()
