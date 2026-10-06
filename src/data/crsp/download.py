"""Download and verify the fixed CRSP security-history extract."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import wrds

from ...utils.configuration import CrspConfig
from ..manifests import calculate_sha256, load_manifest
from .client import open_wrds_connection


_SECURITY_HISTORY_COLUMNS = (
    "permno",
    "permco",
    "secinfostartdt",
    "secinfoenddt",
    "securitybegdt",
    "securityenddt",
    "cusip",
    "cusip9",
    "hdrcusip",
    "hdrcusip9",
    "ticker",
    "tradingsymbol",
    "securitynm",
    "issuernm",
    "primaryexch",
    "shareclass",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "conditionaltype",
    "tradingstatusflg",
    "securityactiveflg",
    "siccd",
    "naics",
)
_DATE_COLUMNS = (
    "secinfostartdt",
    "secinfoenddt",
    "securitybegdt",
    "securityenddt",
)
_ORDER_BY_COLUMNS = (
    "permno",
    "secinfostartdt",
    "secinfoenddt",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrspDownloadResult:
    """Result of downloading or finding the CRSP security history."""

    downloaded: bool
    dataset: str
    source_table: str
    path: Path
    row_count: int


def _validate_manifest(
    data: dict[str, Any],
    path: Path,
    source: CrspConfig,
    output_path: Path,
) -> str:
    """Check that the manifest describes the configured extract."""
    dataset = source.security_history
    expected_values = {
        "source_library": source.stock_library,
        "source_table": dataset.table,
        "start_date": dataset.start_date.isoformat(),
        "end_date": dataset.end_date.isoformat(),
        "file_name": output_path.name,
    }

    for name, expected_value in expected_values.items():
        if data.get(name) != expected_value:
            raise RuntimeError(
                f"CRSP manifest field {name!r} does not match "
                f"the configured extract: {path}"
            )

    sha256 = data.get("sha256")

    if not isinstance(sha256, str) or len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"CRSP manifest SHA-256 is invalid: {path}")

    return sha256


def _verify_security_history(path: Path, expected_sha256: str) -> int:
    """Check that a local CRSP extract matches its fixed manifest."""
    _LOGGER.info("Verifying CRSP security history against the fixed manifest.")

    if not path.is_file():
        raise FileNotFoundError(f"CRSP security-history file not found: {path}")

    if calculate_sha256(path) != expected_sha256:
        raise RuntimeError(f"CRSP security-history SHA-256 mismatch: {path}")

    try:
        row_count = pq.ParquetFile(path).metadata.num_rows
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid CRSP Parquet file: {path}") from error

    if row_count == 0:
        raise RuntimeError(f"CRSP security-history file is empty: {path}")

    return row_count


def _query_security_history(
    connection: wrds.Connection,
    source: CrspConfig,
) -> pd.DataFrame:
    """Retrieve the configured CRSP security-history rows."""
    dataset = source.security_history
    source_table = f"{source.stock_library}.{dataset.table}"
    columns = ",\n            ".join(_SECURITY_HISTORY_COLUMNS)
    _LOGGER.info("Downloading CRSP security history from %s.", source_table)

    table = connection.raw_sql(
        f"""
        SELECT
            {columns}
        FROM {source_table}
        WHERE
            secinfoenddt >= %(start_date)s
            AND secinfostartdt <= %(end_date)s
        ORDER BY
            {", ".join(_ORDER_BY_COLUMNS)}
        """,
        date_cols=list(_DATE_COLUMNS),
        params={
            "start_date": dataset.start_date,
            "end_date": dataset.end_date,
        },
    )

    if table.empty:
        raise RuntimeError("WRDS returned an empty CRSP security-history table.")

    if table["permno"].isna().any():
        raise RuntimeError("CRSP security history contains a missing PERMNO.")

    return table[list(_SECURITY_HISTORY_COLUMNS)]


def download_crsp_security_history(
    source: CrspConfig,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> CrspDownloadResult:
    """Download or verify the configured CRSP security history."""
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    dataset = source.security_history
    source_table = f"{source.stock_library}.{dataset.table}"
    expected_sha256 = _validate_manifest(
        load_manifest(manifest_path),
        manifest_path,
        source,
        output_path,
    )

    if output_path.exists() and not output_path.is_file():
        raise RuntimeError(f"CRSP output path is not a file: {output_path}")

    if output_path.is_file() and not overwrite:
        return CrspDownloadResult(
            downloaded=False,
            dataset="security_history",
            source_table=source_table,
            path=output_path,
            row_count=_verify_security_history(output_path, expected_sha256),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part")
    connection = open_wrds_connection()

    try:
        table = _query_security_history(connection, source)
        table.to_parquet(
            temporary_path,
            index=False,
            compression="zstd",
        )
        row_count = _verify_security_history(temporary_path, expected_sha256)
        temporary_path.replace(output_path)
    finally:
        connection.close()
        temporary_path.unlink(missing_ok=True)

    return CrspDownloadResult(
        downloaded=True,
        dataset="security_history",
        source_table=source_table,
        path=output_path,
        row_count=row_count,
    )


def verify_crsp_security_history(
    source: CrspConfig,
    manifest_path: Path,
    data_path: Path,
) -> int:
    """Verify the local CRSP security history against its fixed manifest."""
    manifest_path = Path(manifest_path)
    data_path = Path(data_path)
    expected_sha256 = _validate_manifest(
        load_manifest(manifest_path),
        manifest_path,
        source,
        data_path,
    )

    return _verify_security_history(data_path, expected_sha256)


__all__ = [
    "CrspDownloadResult",
    "download_crsp_security_history",
    "verify_crsp_security_history",
]
