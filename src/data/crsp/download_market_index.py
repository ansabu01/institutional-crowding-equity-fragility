"""Download and verify the fixed CRSP daily market-index snapshot."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import wrds

from ...utils.configuration import CrspConfig
from ..manifests import calculate_sha256, load_manifest
from .client import open_wrds_connection
from .download import CrspDownloadResult


_COLUMNS = ("indno", "dlycaldt", "dlytotret")
_LOGGER = logging.getLogger(__name__)


def _validate_manifest(
    data: dict[str, Any],
    path: Path,
    source: CrspConfig,
    output_path: Path,
) -> str:
    """Check that the manifest describes the configured market index."""
    dataset = source.daily_market_index
    expected_values = {
        "source_library": source.index_library,
        "source_table": dataset.table,
        "index_id": source.market_index_id,
        "start_date": dataset.start_date.isoformat(),
        "end_date": dataset.end_date.isoformat(),
        "file_name": output_path.name,
    }

    for name, expected_value in expected_values.items():
        if data.get(name) != expected_value:
            raise RuntimeError(
                f"CRSP market-index manifest field {name!r} does not match "
                f"the configured extract: {path}"
            )

    sha256 = data.get("sha256")

    if not isinstance(sha256, str) or len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"CRSP market-index manifest SHA-256 is invalid: {path}")

    return sha256


def _query_market_index(
    connection: wrds.Connection,
    source: CrspConfig,
) -> pd.DataFrame:
    """Retrieve the configured daily market-index returns."""
    dataset = source.daily_market_index
    source_table = f"{source.index_library}.{dataset.table}"
    _LOGGER.info("Downloading CRSP daily market index from %s.", source_table)
    table = connection.raw_sql(
        f"""
        SELECT
            CAST(indno AS BIGINT) AS indno,
            dlycaldt,
            CAST(dlytotret AS DOUBLE PRECISION) AS dlytotret
        FROM {source_table}
        WHERE
            indno = %(index_id)s
            AND dlycaldt BETWEEN %(start_date)s AND %(end_date)s
        ORDER BY dlycaldt
        """,
        date_cols=["dlycaldt"],
        params={
            "index_id": source.market_index_id,
            "start_date": dataset.start_date,
            "end_date": dataset.end_date,
        },
    )

    if table.empty:
        raise RuntimeError("WRDS returned an empty CRSP market-index table.")

    if table["dlycaldt"].duplicated().any():
        raise RuntimeError("CRSP market-index data contains duplicate dates.")

    if table["dlytotret"].isna().any():
        raise RuntimeError("CRSP market-index data contains missing returns.")

    return table[list(_COLUMNS)]


def _verify_market_index(path: Path, expected_sha256: str) -> int:
    """Check that the market-index file matches its fixed manifest."""
    _LOGGER.info("Verifying CRSP market index against the fixed manifest.")

    if not path.is_file():
        raise FileNotFoundError(f"CRSP market-index file not found: {path}")

    if calculate_sha256(path) != expected_sha256:
        raise RuntimeError(f"CRSP market-index SHA-256 mismatch: {path}")

    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid CRSP market-index Parquet: {path}") from error

    if tuple(parquet_file.schema_arrow.names) != _COLUMNS:
        raise RuntimeError(f"Unexpected CRSP market-index columns: {path}")

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"CRSP market-index file is empty: {path}")

    return row_count


def download_crsp_market_index(
    source: CrspConfig,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> CrspDownloadResult:
    """Download or verify the configured CRSP daily market index."""
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    dataset = source.daily_market_index
    source_table = f"{source.index_library}.{dataset.table}"
    expected_sha256 = _validate_manifest(
        load_manifest(manifest_path),
        manifest_path,
        source,
        output_path,
    )

    if output_path.exists() and not output_path.is_file():
        raise RuntimeError(f"CRSP market-index output path is not a file: {output_path}")

    if output_path.is_file() and not overwrite:
        return CrspDownloadResult(
            downloaded=False,
            dataset="daily_market_index",
            source_table=source_table,
            path=output_path,
            row_count=_verify_market_index(output_path, expected_sha256),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part")
    connection = open_wrds_connection()

    try:
        table = _query_market_index(connection, source)
        table.to_parquet(temporary_path, index=False, compression="zstd")
        row_count = _verify_market_index(temporary_path, expected_sha256)
        temporary_path.replace(output_path)
    finally:
        connection.close()
        temporary_path.unlink(missing_ok=True)

    return CrspDownloadResult(
        downloaded=True,
        dataset="daily_market_index",
        source_table=source_table,
        path=output_path,
        row_count=row_count,
    )


__all__ = ["download_crsp_market_index"]
