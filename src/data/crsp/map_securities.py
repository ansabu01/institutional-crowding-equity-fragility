"""Map SEC Form 13F securities to historical CRSP identifiers."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_POSITION_COLUMNS = {
    "CIK",
    "PERIODOFREPORT",
    "report_period",
    "CUSIP",
    "position_value_usd",
}
_SECURITY_HISTORY_COLUMNS = {
    "permno",
    "permco",
    "secinfostartdt",
    "secinfoenddt",
    "cusip9",
    "ticker",
    "securitynm",
    "primaryexch",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
}
_MAPPING_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "CUSIP",
    "match_status",
    "candidate_permno_count",
    "permno",
    "permco",
    "crsp_cusip9",
    "ticker",
    "securitynm",
    "primaryexch",
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "secinfostartdt",
    "secinfoenddt",
}
_AUDIT_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "match_status",
    "securities",
    "manager_positions",
    "position_value_usd",
    "security_share_percent",
    "position_share_percent",
    "value_share_percent",
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecurityMappingResult:
    """Result of creating or loading the security mapping."""

    created: bool
    mapping_path: Path
    audit_path: Path
    mapping_rows: int
    audit_rows: int
    matched_rows: int
    unmatched_rows: int
    ambiguous_rows: int


def _require_parquet(
    path: Path,
    required_columns: set[str],
    description: str,
) -> int:
    """Check a Parquet file and return its row count."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing_columns = required_columns - set(parquet_file.schema_arrow.names)

    if missing_columns:
        raise RuntimeError(
            f"Missing columns in {description}: {sorted(missing_columns)}"
        )

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"{description} is empty: {path}")

    return row_count


def _summarize_mapping(path: Path) -> tuple[int, int, int, int]:
    """Validate the mapping keys and return counts by match status."""
    row_count = _require_parquet(
        path,
        _MAPPING_COLUMNS,
        "Security mapping",
    )
    connection = duckdb.connect()

    try:
        summary = connection.execute(
            """
            SELECT
                COUNT(DISTINCT (report_period, CUSIP)) AS unique_keys,
                COUNT(*) FILTER (WHERE match_status = 'MATCHED') AS matched,
                COUNT(*) FILTER (WHERE match_status = 'UNMATCHED') AS unmatched,
                COUNT(*) FILTER (WHERE match_status = 'AMBIGUOUS') AS ambiguous
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    finally:
        connection.close()

    if summary is None:
        raise RuntimeError("Could not summarize the security mapping.")

    unique_keys, matched, unmatched, ambiguous = summary

    if unique_keys != row_count:
        raise RuntimeError("Security-mapping keys are not unique.")

    if matched + unmatched + ambiguous != row_count:
        raise RuntimeError("Security mapping contains an invalid match status.")

    return row_count, int(matched), int(unmatched), int(ambiguous)


def _existing_result(
    mapping_path: Path,
    audit_path: Path,
    input_paths: tuple[Path, ...],
) -> SecurityMappingResult | None:
    """Return existing outputs unless an input or this code is newer."""
    existing_files = (mapping_path.is_file(), audit_path.is_file())

    if not any(existing_files):
        return None

    if not all(existing_files):
        raise RuntimeError(
            "Security-mapping outputs are incomplete: both files must exist."
        )

    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    oldest_output = min(
        mapping_path.stat().st_mtime_ns,
        audit_path.stat().st_mtime_ns,
    )

    if newest_input > oldest_output:
        raise RuntimeError(
            "Security-mapping inputs or code are newer than the existing "
            "outputs. Run script 6 with --overwrite."
        )

    mapping_rows, matched, unmatched, ambiguous = _summarize_mapping(
        mapping_path
    )
    audit_rows = _require_parquet(
        audit_path,
        _AUDIT_COLUMNS,
        "Security-mapping audit",
    )

    return SecurityMappingResult(
        created=False,
        mapping_path=mapping_path,
        audit_path=audit_path,
        mapping_rows=mapping_rows,
        audit_rows=audit_rows,
        matched_rows=matched,
        unmatched_rows=unmatched,
        ambiguous_rows=ambiguous,
    )


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    positions_path: Path,
    security_history_path: Path,
) -> None:
    """Register the inputs and their normalized mapping fields."""
    connection.read_parquet(str(positions_path)).create_view("_raw_positions")
    connection.execute(
        """
        CREATE TEMP VIEW _positions AS
        SELECT
            CIK,
            PERIODOFREPORT,
            report_period,
            upper(trim(CUSIP)) AS CUSIP,
            position_value_usd
        FROM _raw_positions
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE _security_periods AS
        SELECT DISTINCT
            PERIODOFREPORT,
            report_period,
            CUSIP
        FROM _positions
        """
    )

    connection.read_parquet(str(security_history_path)).create_view(
        "_raw_security_history"
    )
    connection.execute(
        """
        CREATE TEMP VIEW _security_history AS
        SELECT
            permno,
            permco,
            secinfostartdt,
            secinfoenddt,
            upper(trim(cusip9)) AS cusip9,
            ticker,
            securitynm,
            primaryexch,
            sharetype,
            securitytype,
            securitysubtype,
            usincflg,
            issuertype
        FROM _raw_security_history
        WHERE permno IS NOT NULL
            AND cusip9 IS NOT NULL
        """
    )


def _write_mapping(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Write one mapping row per report period and 13F CUSIP."""
    _LOGGER.info("Matching historical 13F CUSIPs to CRSP PERMNOs.")
    connection.execute(
        """
        COPY (
            WITH candidate_rows AS (
                SELECT
                    securities.PERIODOFREPORT,
                    securities.report_period,
                    securities.CUSIP,
                    history.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY
                            securities.report_period,
                            securities.CUSIP,
                            history.permno
                        ORDER BY
                            history.secinfostartdt DESC,
                            history.secinfoenddt DESC
                    ) AS permno_row_rank
                FROM _security_periods AS securities
                INNER JOIN _security_history AS history
                    ON securities.CUSIP = history.cusip9
                    AND securities.report_period BETWEEN
                        history.secinfostartdt AND history.secinfoenddt
            ),
            candidates AS (
                SELECT * EXCLUDE (permno_row_rank)
                FROM candidate_rows
                WHERE permno_row_rank = 1
            ),
            candidate_counts AS (
                SELECT
                    PERIODOFREPORT,
                    report_period,
                    CUSIP,
                    COUNT(*) AS candidate_permno_count
                FROM candidates
                GROUP BY
                    PERIODOFREPORT,
                    report_period,
                    CUSIP
            )
            SELECT
                securities.PERIODOFREPORT,
                securities.report_period,
                securities.CUSIP,
                CASE
                    WHEN COALESCE(counts.candidate_permno_count, 0) = 0
                        THEN 'UNMATCHED'
                    WHEN counts.candidate_permno_count = 1
                        THEN 'MATCHED'
                    ELSE 'AMBIGUOUS'
                END AS match_status,
                COALESCE(counts.candidate_permno_count, 0)::INTEGER
                    AS candidate_permno_count,
                candidates.permno,
                candidates.permco,
                candidates.cusip9 AS crsp_cusip9,
                candidates.ticker,
                candidates.securitynm,
                candidates.primaryexch,
                candidates.sharetype,
                candidates.securitytype,
                candidates.securitysubtype,
                candidates.usincflg,
                candidates.issuertype,
                candidates.secinfostartdt,
                candidates.secinfoenddt
            FROM _security_periods AS securities
            LEFT JOIN candidate_counts AS counts
                USING (PERIODOFREPORT, report_period, CUSIP)
            LEFT JOIN candidates
                ON counts.candidate_permno_count = 1
                AND securities.PERIODOFREPORT = candidates.PERIODOFREPORT
                AND securities.report_period = candidates.report_period
                AND securities.CUSIP = candidates.CUSIP
            ORDER BY
                securities.report_period,
                securities.CUSIP
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """,
        [str(output_path)],
    )


def _write_audit(
    connection: duckdb.DuckDBPyConnection,
    mapping_path: Path,
    output_path: Path,
) -> None:
    """Write quarterly mapping coverage by match status."""
    _LOGGER.info("Building the quarterly CRSP mapping audit.")
    connection.execute(
        """
        COPY (
            WITH status_totals AS (
                SELECT
                    positions.PERIODOFREPORT,
                    positions.report_period,
                    mapping.match_status,
                    COUNT(DISTINCT positions.CUSIP) AS securities,
                    COUNT(*) AS manager_positions,
                    CAST(SUM(positions.position_value_usd) AS BIGINT)
                        AS position_value_usd
                FROM _positions AS positions
                INNER JOIN read_parquet($2) AS mapping
                    USING (PERIODOFREPORT, report_period, CUSIP)
                GROUP BY
                    positions.PERIODOFREPORT,
                    positions.report_period,
                    mapping.match_status
            )
            SELECT
                PERIODOFREPORT,
                report_period,
                match_status,
                securities,
                manager_positions,
                position_value_usd,
                100.0 * securities
                    / SUM(securities) OVER (PARTITION BY report_period)
                    AS security_share_percent,
                100.0 * manager_positions
                    / SUM(manager_positions) OVER (PARTITION BY report_period)
                    AS position_share_percent,
                100.0 * position_value_usd
                    / SUM(position_value_usd) OVER (PARTITION BY report_period)
                    AS value_share_percent
            FROM status_totals
            ORDER BY
                report_period,
                CASE match_status
                    WHEN 'MATCHED' THEN 1
                    WHEN 'UNMATCHED' THEN 2
                    ELSE 3
                END
        ) TO $1 (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """,
        [str(output_path), str(mapping_path)],
    )


def build_sec_13f_crsp_mapping(
    positions_path: Path,
    security_history_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> SecurityMappingResult:
    """Create or load the point-in-time SEC 13F-to-CRSP mapping."""
    positions_path = Path(positions_path)
    security_history_path = Path(security_history_path)
    output_directory = Path(output_directory)
    mapping_path = output_directory / "security_mapping.parquet"
    audit_path = output_directory / "security_mapping_audit.parquet"
    input_paths = (positions_path, security_history_path, Path(__file__))

    _require_parquet(
        positions_path,
        _POSITION_COLUMNS,
        "Form 13F manager-security positions",
    )
    _require_parquet(
        security_history_path,
        _SECURITY_HISTORY_COLUMNS,
        "CRSP security history",
    )

    if not overwrite:
        result = _existing_result(mapping_path, audit_path, input_paths)

        if result is not None:
            return result

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_mapping = mapping_path.with_name(f".{mapping_path.name}.part")
    temporary_audit = audit_path.with_name(f".{audit_path.name}.part")
    connection = duckdb.connect()

    try:
        _register_inputs(
            connection,
            positions_path,
            security_history_path,
        )
        _write_mapping(connection, temporary_mapping)
        mapping_rows, matched, unmatched, ambiguous = _summarize_mapping(
            temporary_mapping
        )
        _write_audit(
            connection,
            temporary_mapping,
            temporary_audit,
        )
        audit_rows = _require_parquet(
            temporary_audit,
            _AUDIT_COLUMNS,
            "Security-mapping audit",
        )
        temporary_mapping.replace(mapping_path)
        temporary_audit.replace(audit_path)
    finally:
        connection.close()
        temporary_mapping.unlink(missing_ok=True)
        temporary_audit.unlink(missing_ok=True)

    return SecurityMappingResult(
        created=True,
        mapping_path=mapping_path,
        audit_path=audit_path,
        mapping_rows=mapping_rows,
        audit_rows=audit_rows,
        matched_rows=matched,
        unmatched_rows=unmatched,
        ambiguous_rows=ambiguous,
    )


__all__ = [
    "SecurityMappingResult",
    "build_sec_13f_crsp_mapping",
]
