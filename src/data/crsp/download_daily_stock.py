"""Download and verify the fixed CRSP daily-stock snapshot."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import wrds

from ...utils.configuration import CrspConfig
from ..manifests import calculate_sha256, load_manifest
from .client import open_wrds_connection
from .download import CrspDownloadResult


_DAILY_COLUMNS = (
    "permno",
    "permco",
    "dlycaldt",
    "conditionaltype",
    "tradingstatusflg",
    "dlydelflg",
    "dlyprc",
    "dlyprcflg",
    "dlycap",
    "dlycapflg",
    "dlyret",
    "dlyretx",
    "dlyretmissflg",
    "dlyretdurflg",
    "dlydistretflg",
    "dlyvol",
    "shrout",
)
_PERMNO_BATCH_SIZE = 250
_LOGGER = logging.getLogger(__name__)


def _load_permnos(path: Path) -> tuple[int, ...]:
    """Load the distinct PERMNOs defining the daily-data universe."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"CRSP PERMNO-universe file not found: {path}")

    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Invalid CRSP PERMNO-universe Parquet: {path}"
        ) from error

    if "permno" not in columns:
        raise RuntimeError("CRSP PERMNO-universe file does not contain permno.")

    connection = duckdb.connect()

    try:
        rows = connection.execute(
            """
            SELECT DISTINCT CAST(permno AS BIGINT) AS permno
            FROM read_parquet(?)
            WHERE permno IS NOT NULL
            ORDER BY permno
            """,
            [str(path)],
        ).fetchall()
    finally:
        connection.close()

    permnos = tuple(int(row[0]) for row in rows)

    if not permnos:
        raise RuntimeError("The CRSP PERMNO universe is empty.")

    return permnos


def _calculate_permno_sha256(permnos: tuple[int, ...]) -> str:
    """Calculate a stable fingerprint of the ordered PERMNO universe."""
    content = "\n".join(str(permno) for permno in permnos).encode("ascii")

    return hashlib.sha256(content).hexdigest()


def _validate_manifest(
    data: dict[str, Any],
    path: Path,
    source: CrspConfig,
    universe_sha256: str,
    output_path: Path,
) -> str:
    """Check that the manifest describes the configured daily extract."""
    dataset = source.daily_stock
    expected_values = {
        "source_library": source.stock_library,
        "source_table": dataset.table,
        "start_date": dataset.start_date.isoformat(),
        "end_date": dataset.end_date.isoformat(),
        "universe_sha256": universe_sha256,
        "file_name": output_path.name,
    }

    for name, expected_value in expected_values.items():
        if data.get(name) != expected_value:
            raise RuntimeError(
                f"CRSP daily-stock manifest field {name!r} does not match "
                f"the configured extract: {path}"
            )

    sha256 = data.get("sha256")

    if not isinstance(sha256, str) or len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"CRSP daily-stock manifest SHA-256 is invalid: {path}")

    return sha256


def _copy_daily_csv(
    connection: wrds.Connection,
    source: CrspConfig,
    permnos: tuple[int, ...],
    output_path: Path,
) -> None:
    """Download the CRSP daily rows in manageable PERMNO batches."""
    dataset = source.daily_stock
    source_table = f"{source.stock_library}.{dataset.table}"
    columns = ",\n            ".join(_DAILY_COLUMNS)
    batch_count = (
        len(permnos) + _PERMNO_BATCH_SIZE - 1
    ) // _PERMNO_BATCH_SIZE
    raw_connection = connection.engine.raw_connection()
    cursor = raw_connection.cursor()

    try:
        with output_path.open("wb") as file:
            for batch_number, start in enumerate(
                range(0, len(permnos), _PERMNO_BATCH_SIZE),
                start=1,
            ):
                batch = permnos[start : start + _PERMNO_BATCH_SIZE]
                _LOGGER.info(
                    "[%02d/%02d] Downloading CRSP daily PERMNO group.",
                    batch_number,
                    batch_count,
                )
                query = cursor.mogrify(
                    f"""
                    SELECT
                        {columns}
                    FROM {source_table}
                    WHERE
                        dlycaldt BETWEEN %s AND %s
                        AND permno = ANY(%s)
                    """,
                    (
                        dataset.start_date,
                        dataset.end_date,
                        list(batch),
                    ),
                ).decode("utf-8")
                header = "TRUE" if batch_number == 1 else "FALSE"
                cursor.copy_expert(
                    f"COPY ({query}) TO STDOUT WITH "
                    f"(FORMAT CSV, HEADER {header})",
                    file,
                )
    finally:
        cursor.close()
        raw_connection.close()


def _convert_csv_to_parquet(csv_path: Path, output_path: Path) -> None:
    """Convert the downloaded CSV to an ordered typed Parquet file."""
    connection = duckdb.connect()
    csv_literal = str(csv_path).replace("'", "''")
    output_literal = str(output_path).replace("'", "''")

    try:
        connection.execute("SET preserve_insertion_order = true")
        connection.execute(
            f"""
            COPY (
                SELECT
                    CAST(permno AS BIGINT) AS permno,
                    CAST(permco AS BIGINT) AS permco,
                    CAST(dlycaldt AS TIMESTAMP_NS) AS dlycaldt,
                    CAST(conditionaltype AS VARCHAR) AS conditionaltype,
                    CAST(tradingstatusflg AS VARCHAR) AS tradingstatusflg,
                    CAST(dlydelflg AS VARCHAR) AS dlydelflg,
                    CAST(dlyprc AS DOUBLE) AS dlyprc,
                    CAST(dlyprcflg AS VARCHAR) AS dlyprcflg,
                    CAST(dlycap AS DOUBLE) AS dlycap,
                    CAST(dlycapflg AS VARCHAR) AS dlycapflg,
                    CAST(dlyret AS DOUBLE) AS dlyret,
                    CAST(dlyretx AS DOUBLE) AS dlyretx,
                    CAST(dlyretmissflg AS VARCHAR) AS dlyretmissflg,
                    CAST(dlyretdurflg AS VARCHAR) AS dlyretdurflg,
                    CAST(dlydistretflg AS VARCHAR) AS dlydistretflg,
                    CAST(dlyvol AS DOUBLE) AS dlyvol,
                    CAST(shrout AS BIGINT) AS shrout
                FROM read_csv('{csv_literal}', header = true, auto_detect = true)
                ORDER BY dlycaldt, permno
            ) TO '{output_literal}' (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000,
                PRESERVE_ORDER true
            )
            """
        )
    finally:
        connection.close()


def _verify_daily_file(path: Path, expected_sha256: str) -> int:
    """Check that the daily-stock file matches its fixed manifest."""
    _LOGGER.info("Verifying CRSP daily stock data against the fixed manifest.")

    if not path.is_file():
        raise FileNotFoundError(f"CRSP daily-stock file not found: {path}")

    if calculate_sha256(path) != expected_sha256:
        raise RuntimeError(f"CRSP daily-stock SHA-256 mismatch: {path}")

    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid CRSP daily-stock Parquet: {path}") from error

    if tuple(parquet_file.schema_arrow.names) != _DAILY_COLUMNS:
        raise RuntimeError(f"Unexpected CRSP daily-stock columns: {path}")

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"CRSP daily-stock file is empty: {path}")

    return row_count


def download_crsp_daily_stock(
    source: CrspConfig,
    universe_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> CrspDownloadResult:
    """Download or verify the configured CRSP daily-stock snapshot."""
    universe_path = Path(universe_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    dataset = source.daily_stock
    source_table = f"{source.stock_library}.{dataset.table}"
    permnos = _load_permnos(universe_path)
    expected_sha256 = _validate_manifest(
        load_manifest(manifest_path),
        manifest_path,
        source,
        _calculate_permno_sha256(permnos),
        output_path,
    )

    if output_path.exists() and not output_path.is_file():
        raise RuntimeError(f"CRSP daily output path is not a file: {output_path}")

    if output_path.is_file() and not overwrite:
        return CrspDownloadResult(
            downloaded=False,
            dataset="daily_stock",
            source_table=source_table,
            path=output_path,
            row_count=_verify_daily_file(output_path, expected_sha256),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_part = output_path.with_name(f".{output_path.name}.part")
    csv_part = output_path.with_name(f".{output_path.stem}.csv.part")
    parquet_part.unlink(missing_ok=True)
    csv_part.unlink(missing_ok=True)
    connection = open_wrds_connection()

    try:
        _LOGGER.info("Downloading CRSP daily stock data from %s.", source_table)
        _copy_daily_csv(connection, source, permnos, csv_part)
        _LOGGER.info("Converting the CRSP daily extract to Parquet.")
        _convert_csv_to_parquet(csv_part, parquet_part)
        row_count = _verify_daily_file(parquet_part, expected_sha256)
        parquet_part.replace(output_path)
    finally:
        connection.close()
        csv_part.unlink(missing_ok=True)
        parquet_part.unlink(missing_ok=True)

    return CrspDownloadResult(
        downloaded=True,
        dataset="daily_stock",
        source_table=source_table,
        path=output_path,
        row_count=row_count,
    )


__all__ = ["download_crsp_daily_stock"]
