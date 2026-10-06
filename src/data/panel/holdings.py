"""Build the analysis-ready SEC Form 13F-CRSP holdings panel."""

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
    "NAMEOFISSUER",
    "TITLEOFCLASS",
    "position_value_usd",
    "raw_position_rows",
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
}
_SELECTED_FILING_COLUMNS = {
    "CIK",
    "PERIODOFREPORT",
    "report_period",
    "FILINGMANAGER_NAME",
    "filing_deadline",
    "selection_decision",
    "is_selected",
}
_OUTPUT_NAMES = (
    "manager_security_holdings",
    "manager_portfolio_totals",
    "sample_construction_audit",
)
_ELIGIBLE_EXCHANGES = "('N', 'A', 'Q')"
_DUCKDB_MEMORY_LIMIT = "8GB"
_DUCKDB_THREADS = 4
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoldingsTableResult:
    """Metadata for one holdings-panel output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class HoldingsPanelResult:
    """Result of creating or loading the holdings-panel outputs."""

    created: bool
    tables: tuple[HoldingsTableResult, ...]


def _open_connection() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection suitable for the large 13F table."""
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = '{_DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET threads = {_DUCKDB_THREADS}")
    connection.execute("SET preserve_insertion_order = false")

    return connection


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


def _output_paths(output_directory: Path) -> dict[str, Path]:
    """Return the three holdings-panel output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_NAMES
    }


def _table_results(
    paths: dict[str, Path],
) -> tuple[HoldingsTableResult, ...]:
    """Return basic metadata for the output tables."""
    return tuple(
        HoldingsTableResult(
            name=name,
            path=paths[name],
            row_count=_require_parquet(
                paths[name],
                set(),
                f"{name} output",
            ),
        )
        for name in _OUTPUT_NAMES
    )


def _existing_result(
    output_paths: dict[str, Path],
    input_paths: tuple[Path, ...],
) -> HoldingsPanelResult | None:
    """Return existing outputs unless an input or this code is newer."""
    existing_names = {
        name for name, path in output_paths.items() if path.is_file()
    }

    if not existing_names:
        return None

    if len(existing_names) != len(output_paths):
        missing = [
            str(path)
            for name, path in output_paths.items()
            if name not in existing_names
        ]
        raise RuntimeError(
            "Holdings-panel outputs are incomplete. Missing files:\n  - "
            + "\n  - ".join(missing)
        )

    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    oldest_output = min(
        path.stat().st_mtime_ns for path in output_paths.values()
    )

    if newest_input > oldest_output:
        raise RuntimeError(
            "Holdings-panel inputs or code are newer than the existing outputs. "
            "Run script 7 with --overwrite."
        )

    return HoldingsPanelResult(
        created=False,
        tables=_table_results(output_paths),
    )


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    positions_path: Path,
    mapping_path: Path,
    selected_filings_path: Path,
) -> None:
    """Register inputs and create the two position views used below."""
    connection.read_parquet(str(positions_path)).create_view("_raw_positions")
    connection.read_parquet(str(mapping_path)).create_view("_security_mapping")
    connection.read_parquet(str(selected_filings_path)).create_view(
        "_selected_filings"
    )
    connection.execute(
        """
        CREATE TEMP VIEW _positions AS
        SELECT
            * EXCLUDE (CUSIP),
            upper(trim(CUSIP)) AS CUSIP
        FROM _raw_positions
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE _manager_metadata AS
        SELECT
            CIK,
            PERIODOFREPORT,
            min(report_period) AS report_period,
            min(filing_deadline) AS information_date,
            max(FILINGMANAGER_NAME) FILTER (
                WHERE selection_decision = 'SELECTED_BASE'
            ) AS FILINGMANAGER_NAME
        FROM _selected_filings
        WHERE is_selected
        GROUP BY CIK, PERIODOFREPORT
        """
    )
    connection.execute(
        """
        CREATE TEMP VIEW _mapped_positions AS
        SELECT
            positions.CIK,
            positions.PERIODOFREPORT,
            positions.report_period,
            metadata.information_date,
            positions.CUSIP,
            positions.NAMEOFISSUER,
            positions.TITLEOFCLASS,
            positions.position_value_usd,
            positions.raw_position_rows,
            mapping.match_status,
            mapping.candidate_permno_count,
            mapping.permno,
            mapping.permco,
            mapping.crsp_cusip9,
            mapping.ticker,
            mapping.securitynm,
            mapping.primaryexch,
            mapping.sharetype,
            mapping.securitytype,
            mapping.securitysubtype,
            mapping.usincflg,
            mapping.issuertype
        FROM _positions AS positions
        INNER JOIN _manager_metadata AS metadata
            USING (CIK, PERIODOFREPORT, report_period)
        INNER JOIN _security_mapping AS mapping
            USING (PERIODOFREPORT, report_period, CUSIP)
        """
    )
    connection.execute(
        f"""
        CREATE TEMP VIEW _eligible_positions AS
        SELECT *
        FROM _mapped_positions
        WHERE match_status = 'MATCHED'
            AND candidate_permno_count = 1
            AND permno IS NOT NULL
            AND securitytype = 'EQTY'
            AND securitysubtype = 'COM'
            AND sharetype = 'NS'
            AND usincflg = 'Y'
            AND primaryexch IN {_ELIGIBLE_EXCHANGES}
        """
    )


def _write_holdings(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Aggregate eligible positions to manager-quarter-PERMNO holdings."""
    report_periods = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT report_period "
            "FROM _manager_metadata ORDER BY report_period"
        ).fetchall()
    ]
    part_paths: list[Path] = []

    try:
        for index, report_period in enumerate(report_periods, start=1):
            part_path = output_path.with_name(
                f"{output_path.name}.{index:03d}.quarter.parquet"
            )
            part_paths.append(part_path)
            _LOGGER.info(
                "[%02d/%02d] Building holdings for %s.",
                index,
                len(report_periods),
                report_period.date(),
            )
            connection.execute(
                """
                COPY (
                    WITH aggregated AS (
                        SELECT
                            CIK,
                            PERIODOFREPORT,
                            report_period,
                            information_date,
                            permno,
                            arg_max(permco, position_value_usd) AS permco,
                            arg_max(CUSIP, position_value_usd) AS CUSIP,
                            arg_max(crsp_cusip9, position_value_usd)
                                AS crsp_cusip9,
                            COUNT(DISTINCT CUSIP)::INTEGER
                                AS mapped_cusip_count,
                            arg_max(NAMEOFISSUER, position_value_usd)
                                AS NAMEOFISSUER,
                            arg_max(TITLEOFCLASS, position_value_usd)
                                AS TITLEOFCLASS,
                            arg_max(ticker, position_value_usd) AS ticker,
                            arg_max(securitynm, position_value_usd) AS securitynm,
                            arg_max(primaryexch, position_value_usd)
                                AS primaryexch,
                            arg_max(sharetype, position_value_usd) AS sharetype,
                            arg_max(securitytype, position_value_usd)
                                AS securitytype,
                            arg_max(securitysubtype, position_value_usd)
                                AS securitysubtype,
                            arg_max(usincflg, position_value_usd) AS usincflg,
                            arg_max(issuertype, position_value_usd) AS issuertype,
                            CAST(SUM(position_value_usd) AS BIGINT)
                                AS position_value_usd,
                            CAST(SUM(raw_position_rows) AS BIGINT)
                                AS raw_position_rows
                        FROM _eligible_positions
                        WHERE report_period = $2
                        GROUP BY
                            CIK,
                            PERIODOFREPORT,
                            report_period,
                            information_date,
                            permno
                    ),
                    with_totals AS (
                        SELECT
                            *,
                            CAST(
                                SUM(position_value_usd) OVER (
                                    PARTITION BY CIK, PERIODOFREPORT
                                ) AS BIGINT
                            ) AS portfolio_value_usd
                        FROM aggregated
                    )
                    SELECT
                        *,
                        CAST(position_value_usd AS DOUBLE)
                            / portfolio_value_usd AS portfolio_weight
                    FROM with_totals
                ) TO $1 (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE 100000
                )
                """,
                [str(part_path), report_period],
            )

        connection.read_parquet(
            [str(path) for path in part_paths]
        ).create_view("_quarterly_holdings")
        connection.execute(
            """
            COPY (
                SELECT * FROM _quarterly_holdings
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(output_path)],
        )
        connection.execute("DROP VIEW _quarterly_holdings")
    finally:
        for path in part_paths:
            path.unlink(missing_ok=True)


def _write_portfolio_totals(
    connection: duckdb.DuckDBPyConnection,
    holdings_path: Path,
    output_path: Path,
) -> None:
    """Write one summary row per eligible manager-quarter portfolio."""
    _LOGGER.info("Building eligible manager portfolio totals.")
    connection.execute(
        """
        COPY (
            SELECT
                holdings.CIK,
                holdings.PERIODOFREPORT,
                min(holdings.report_period) AS report_period,
                min(holdings.information_date) AS information_date,
                max(metadata.FILINGMANAGER_NAME) AS FILINGMANAGER_NAME,
                CAST(SUM(holdings.position_value_usd) AS BIGINT)
                    AS portfolio_value_usd,
                COUNT(*) AS security_count,
                CAST(SUM(holdings.raw_position_rows) AS BIGINT)
                    AS underlying_position_rows
            FROM read_parquet($2) AS holdings
            INNER JOIN _manager_metadata AS metadata
                USING (CIK, PERIODOFREPORT, report_period)
            GROUP BY holdings.CIK, holdings.PERIODOFREPORT
            ORDER BY report_period, holdings.CIK
        ) TO $1 (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """,
        [str(output_path), str(holdings_path)],
    )


def _write_audit(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Write comparable CUSIP-level counts for each sample filter."""
    _LOGGER.info("Building the holdings sample-construction audit.")
    connection.execute(
        """
        COPY (
            WITH stage_totals AS (
                SELECT
                    PERIODOFREPORT,
                    report_period,
                    min(information_date) AS information_date,
                    1 AS stage_order,
                    'FORM_13F_POSITIONS' AS stage,
                    COUNT(*) AS manager_positions,
                    COUNT(DISTINCT (CIK, PERIODOFREPORT))
                        AS manager_periods,
                    COUNT(DISTINCT CUSIP) AS securities,
                    CAST(SUM(position_value_usd) AS BIGINT)
                        AS position_value_usd
                FROM _mapped_positions
                GROUP BY PERIODOFREPORT, report_period

                UNION ALL

                SELECT
                    PERIODOFREPORT,
                    report_period,
                    min(information_date) AS information_date,
                    2 AS stage_order,
                    'EXACT_CRSP_MATCHES' AS stage,
                    COUNT(*) AS manager_positions,
                    COUNT(DISTINCT (CIK, PERIODOFREPORT))
                        AS manager_periods,
                    COUNT(DISTINCT CUSIP) AS securities,
                    CAST(SUM(position_value_usd) AS BIGINT)
                        AS position_value_usd
                FROM _mapped_positions
                WHERE match_status = 'MATCHED'
                GROUP BY PERIODOFREPORT, report_period

                UNION ALL

                SELECT
                    PERIODOFREPORT,
                    report_period,
                    min(information_date) AS information_date,
                    3 AS stage_order,
                    'ELIGIBLE_US_COMMON_EQUITIES' AS stage,
                    COUNT(*) AS manager_positions,
                    COUNT(DISTINCT (CIK, PERIODOFREPORT))
                        AS manager_periods,
                    COUNT(DISTINCT CUSIP) AS securities,
                    CAST(SUM(position_value_usd) AS BIGINT)
                        AS position_value_usd
                FROM _eligible_positions
                GROUP BY PERIODOFREPORT, report_period
            ),
            with_baselines AS (
                SELECT
                    *,
                    max(manager_positions) FILTER (WHERE stage_order = 1)
                        OVER (PARTITION BY report_period)
                        AS baseline_positions,
                    max(manager_periods) FILTER (WHERE stage_order = 1)
                        OVER (PARTITION BY report_period)
                        AS baseline_managers,
                    max(position_value_usd) FILTER (WHERE stage_order = 1)
                        OVER (PARTITION BY report_period)
                        AS baseline_value
                FROM stage_totals
            )
            SELECT
                PERIODOFREPORT,
                report_period,
                information_date,
                stage,
                manager_positions,
                manager_periods,
                securities,
                position_value_usd,
                100.0 * manager_positions / baseline_positions
                    AS position_retention_percent,
                100.0 * manager_periods / baseline_managers
                    AS manager_retention_percent,
                100.0 * position_value_usd / baseline_value
                    AS value_retention_percent
            FROM with_baselines
            ORDER BY report_period, stage_order
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """,
        [str(output_path)],
    )


def build_sec_13f_crsp_holdings(
    positions_path: Path,
    mapping_path: Path,
    selected_filings_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> HoldingsPanelResult:
    """Create or load the eligible manager-quarter holdings panel."""
    positions_path = Path(positions_path)
    mapping_path = Path(mapping_path)
    selected_filings_path = Path(selected_filings_path)
    output_directory = Path(output_directory)
    input_paths = (
        positions_path,
        mapping_path,
        selected_filings_path,
        Path(__file__),
    )
    output_paths = _output_paths(output_directory)

    _require_parquet(
        positions_path,
        _POSITION_COLUMNS,
        "Form 13F manager-security positions",
    )
    _require_parquet(
        mapping_path,
        _MAPPING_COLUMNS,
        "SEC Form 13F-CRSP security mapping",
    )
    _require_parquet(
        selected_filings_path,
        _SELECTED_FILING_COLUMNS,
        "Selected Form 13F filings",
    )

    if not overwrite:
        result = _existing_result(output_paths, input_paths)

        if result is not None:
            return result

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_name(f".{path.name}.part")
        for name, path in output_paths.items()
    }

    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    connection = _open_connection()

    try:
        _register_inputs(
            connection,
            positions_path,
            mapping_path,
            selected_filings_path,
        )
        _write_holdings(
            connection,
            temporary_paths["manager_security_holdings"],
        )
        _write_portfolio_totals(
            connection,
            temporary_paths["manager_security_holdings"],
            temporary_paths["manager_portfolio_totals"],
        )
        _write_audit(
            connection,
            temporary_paths["sample_construction_audit"],
        )

        tables = _table_results(temporary_paths)

        for name in _OUTPUT_NAMES:
            temporary_paths[name].replace(output_paths[name])
    finally:
        connection.close()

        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    return HoldingsPanelResult(
        created=True,
        tables=tables,
    )


__all__ = [
    "HoldingsPanelResult",
    "HoldingsTableResult",
    "build_sec_13f_crsp_holdings",
]
