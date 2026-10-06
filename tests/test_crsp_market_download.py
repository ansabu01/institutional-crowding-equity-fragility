"""Check fixed-manifest reuse of CRSP daily market data."""

import hashlib
import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data.crsp.download_daily_stock import download_crsp_daily_stock
from src.data.crsp.download_market_index import download_crsp_market_index
from src.data.manifests import calculate_sha256
from src.utils.configuration import CrspConfig, CrspDatasetConfig


class CrspMarketDownloadTests(unittest.TestCase):
    def test_daily_stock_reuse_checks_the_exact_schema(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            universe_path = root / "universe.parquet"
            output_path = root / "daily_stock.parquet"
            manifest_path = root / "daily_stock.json"
            pd.DataFrame({"permno": [10001]}).to_parquet(
                universe_path,
                index=False,
            )
            pd.DataFrame({"permno": [10001]}).to_parquet(
                output_path,
                index=False,
            )
            self._write_manifest(
                manifest_path,
                {
                    "source_library": "crsp_m_stock",
                    "source_table": "dsf_v2",
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                    "universe_sha256": hashlib.sha256(b"10001").hexdigest(),
                    "file_name": output_path.name,
                    "sha256": calculate_sha256(output_path),
                },
            )

            with self.assertRaisesRegex(RuntimeError, "Unexpected.*columns"):
                download_crsp_daily_stock(
                    self._config(),
                    universe_path,
                    manifest_path,
                    output_path,
                )

    def test_market_index_reuses_a_valid_fixed_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "daily_market_index.parquet"
            manifest_path = root / "daily_market_index.json"
            pd.DataFrame(
                {
                    "indno": [1000080],
                    "dlycaldt": [pd.Timestamp("2020-01-02")],
                    "dlytotret": [0.01],
                }
            ).to_parquet(output_path, index=False)
            self._write_manifest(
                manifest_path,
                {
                    "source_library": "crsp_m_indexes",
                    "source_table": "inddlyseriesdata_ind",
                    "index_id": 1000080,
                    "start_date": "2020-01-01",
                    "end_date": "2020-12-31",
                    "file_name": output_path.name,
                    "sha256": calculate_sha256(output_path),
                },
            )

            result = download_crsp_market_index(
                self._config(),
                manifest_path,
                output_path,
            )

        self.assertFalse(result.downloaded)
        self.assertEqual(result.row_count, 1)

    @staticmethod
    def _config() -> CrspConfig:
        period = CrspDatasetConfig(
            table="unused",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
        )

        return CrspConfig(
            stock_library="crsp_m_stock",
            index_library="crsp_m_indexes",
            security_history=period,
            daily_stock=CrspDatasetConfig(
                table="dsf_v2",
                start_date=period.start_date,
                end_date=period.end_date,
            ),
            daily_market_index=CrspDatasetConfig(
                table="inddlyseriesdata_ind",
                start_date=period.start_date,
                end_date=period.end_date,
            ),
            market_index_id=1000080,
        )

    @staticmethod
    def _write_manifest(path: Path, data: dict[str, object]) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
