"""Build future downside outcomes for stock-quarter observations."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_DAILY_COLUMNS = {"permno", "dlycaldt", "dlydelflg", "dlyret"}
_SECURITY_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "permno",
}
_OUTPUT_COLUMNS = {
    "downside_targets": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "permno",
        "target_horizon_market_days",
        "target_window_start",
        "target_window_end",
        "return_observations_63d",
        "delisted_in_window",
        "delisting_return_observed",
        "future_max_drawdown_63d",
        "future_cumulative_return_63d",
        "future_downside_volatility_63d",
        "future_worst_five_day_return_63d",
        "future_worst_daily_return_63d",
        "target_window_complete",
        "target_usable",
    ),
    "target_construction_audit": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "security_periods",
        "complete_horizon_security_periods",
        "usable_security_periods",
        "usable_share",
        "median_return_observations_63d",
        "delisting_events",
        "missing_delisting_returns",
        "median_future_max_drawdown_63d",
    ),
}
_TARGET_HORIZON_MARKET_DAYS = 63
_MINIMUM_RETURN_OBSERVATIONS = 50
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownsideTargetTableResult:
    """Metadata for one downside-target output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class DownsideTargetBuildResult:
    """Result of creating or loading the downside-target outputs."""

    created: bool
    tables: tuple[DownsideTargetTableResult, ...]
    security_periods: int
    usable_security_periods: int


def _require_parquet_columns(
    path: Path,
    required_columns: set[str],
    description: str,
) -> None:
    """Validate an input Parquet file and its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing_columns = required_columns - columns

    if missing_columns:
        raise RuntimeError(
            f"Missing columns in {description}: {sorted(missing_columns)}"
        )


def _get_output_paths(output_directory: Path) -> dict[str, Path]:
    """Return the downside-target output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _validate_output_schema(path: Path, expected_columns: tuple[str, ...]) -> int:
    """Validate one output schema and return its row count."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid downside-target output: {path}") from error

    actual_columns = tuple(parquet_file.schema_arrow.names)

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: "
            f"expected {list(expected_columns)}, found {list(actual_columns)}."
        )

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"Downside-target output is empty: {path}")

    return row_count


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    daily_path: Path,
    security_nodes_path: Path,
) -> None:
    """Register source data and construct common future market windows."""
    connection.read_parquet(str(daily_path)).create_view("_daily_market")
    connection.read_parquet(str(security_nodes_path)).create_view(
        "_security_nodes"
    )
    connection.execute(
        """
        CREATE TEMP TABLE _market_calendar AS
        SELECT
            dlycaldt,
            ROW_NUMBER() OVER (ORDER BY dlycaldt) AS market_day
        FROM (
            SELECT DISTINCT dlycaldt
            FROM _daily_market
        )
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE _predictions AS
        SELECT DISTINCT
            PERIODOFREPORT,
            report_period,
            information_date,
            permno
        FROM _security_nodes
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE _target_windows AS
        WITH prediction_dates AS (
            SELECT DISTINCT information_date
            FROM _predictions
        ),
        first_days AS (
            SELECT
                p.information_date,
                c.market_day,
                c.dlycaldt AS target_window_start
            FROM prediction_dates p
            ASOF LEFT JOIN _market_calendar c
                ON p.information_date < c.dlycaldt
        )
        SELECT
            p.*,
            f.target_window_start,
            e.dlycaldt AS target_window_end
        FROM _predictions p
        LEFT JOIN first_days f USING (information_date)
        LEFT JOIN _market_calendar e
            ON e.market_day = f.market_day + {_TARGET_HORIZON_MARKET_DAYS - 1}
        """
    )


def _create_downside_targets(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Create primary and robustness outcomes over the common future window."""
    _LOGGER.info("Building future downside targets.")
    connection.execute(
        f"""
        COPY (
            WITH future_observations AS (
                SELECT
                    w.*,
                    d.dlycaldt,
                    d.dlyret,
                    d.dlydelflg
                FROM _target_windows w
                JOIN _daily_market d
                    ON d.permno = w.permno
                    AND d.dlycaldt BETWEEN
                        w.target_window_start AND w.target_window_end
            ),
            delisting_status AS (
                SELECT
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno,
                    BOOL_OR(dlydelflg = 'Y') AS delisted_in_window,
                    BOOL_OR(
                        dlydelflg = 'Y' AND dlyret IS NOT NULL
                    ) AS delisting_return_observed
                FROM future_observations
                GROUP BY
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno
            ),
            future_returns AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY report_period, permno
                        ORDER BY dlycaldt
                    ) AS security_day,
                    PRODUCT(1.0 + dlyret) OVER (
                        PARTITION BY report_period, permno
                        ORDER BY dlycaldt
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS wealth,
                    PRODUCT(1.0 + dlyret) OVER (
                        PARTITION BY report_period, permno
                        ORDER BY dlycaldt
                        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) - 1.0 AS five_day_return
                FROM future_observations
                WHERE dlyret IS NOT NULL
            ),
            drawdown_paths AS (
                SELECT
                    *,
                    wealth / GREATEST(
                        1.0,
                        MAX(wealth) OVER (
                            PARTITION BY report_period, permno
                            ORDER BY dlycaldt
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        )
                    ) - 1.0 AS drawdown
                FROM future_returns
            ),
            metrics AS (
                SELECT
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno,
                    COUNT(dlyret) AS return_observations_63d,
                    -MIN(drawdown) AS future_max_drawdown_63d,
                    PRODUCT(1.0 + dlyret) - 1.0
                        AS future_cumulative_return_63d,
                    SQRT(252.0 * AVG(POW(LEAST(dlyret, 0.0), 2)))
                        AS future_downside_volatility_63d,
                    MIN(CASE WHEN security_day >= 5 THEN five_day_return END)
                        AS future_worst_five_day_return_63d,
                    MIN(dlyret) AS future_worst_daily_return_63d
                FROM drawdown_paths
                GROUP BY
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno
            ),
            combined AS (
                SELECT
                    w.*,
                    COALESCE(m.return_observations_63d, 0)
                        AS return_observations_63d,
                    COALESCE(d.delisted_in_window, false)
                        AS delisted_in_window,
                    COALESCE(d.delisting_return_observed, false)
                        AS delisting_return_observed,
                    m.future_max_drawdown_63d,
                    m.future_cumulative_return_63d,
                    m.future_downside_volatility_63d,
                    m.future_worst_five_day_return_63d,
                    m.future_worst_daily_return_63d,
                    w.target_window_end IS NOT NULL AS target_window_complete
                FROM _target_windows w
                LEFT JOIN metrics m USING (
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno
                )
                LEFT JOIN delisting_status d USING (
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno
                )
            )
            SELECT
                PERIODOFREPORT,
                report_period,
                information_date,
                permno,
                {_TARGET_HORIZON_MARKET_DAYS} AS target_horizon_market_days,
                target_window_start,
                target_window_end,
                return_observations_63d,
                delisted_in_window,
                delisting_return_observed,
                future_max_drawdown_63d,
                future_cumulative_return_63d,
                future_downside_volatility_63d,
                future_worst_five_day_return_63d,
                future_worst_daily_return_63d,
                target_window_complete,
                (
                    target_window_complete
                    AND (
                        NOT delisted_in_window
                        OR delisting_return_observed
                    )
                    AND (
                        return_observations_63d >= {_MINIMUM_RETURN_OBSERVATIONS}
                        OR delisting_return_observed
                    )
                    AND future_max_drawdown_63d IS NOT NULL
                ) AS target_usable
            FROM combined
            ORDER BY report_period, permno
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """,
        [str(output_path)],
    )


def _create_target_audit(
    connection: duckdb.DuckDBPyConnection,
    targets_path: Path,
    output_path: Path,
) -> None:
    """Create quarterly target coverage and distribution diagnostics."""
    connection.read_parquet(str(targets_path)).create_view("_downside_targets")
    connection.execute(
        """
        COPY (
            SELECT
                PERIODOFREPORT,
                report_period,
                information_date,
                COUNT(*) AS security_periods,
                COUNT(*) FILTER (WHERE target_window_complete)
                    AS complete_horizon_security_periods,
                COUNT(*) FILTER (WHERE target_usable)
                    AS usable_security_periods,
                AVG(CAST(target_usable AS DOUBLE)) AS usable_share,
                MEDIAN(return_observations_63d)
                    AS median_return_observations_63d,
                COUNT(*) FILTER (WHERE delisted_in_window) AS delisting_events,
                COUNT(*) FILTER (
                    WHERE delisted_in_window
                        AND NOT delisting_return_observed
                ) AS missing_delisting_returns,
                MEDIAN(future_max_drawdown_63d) FILTER (WHERE target_usable)
                    AS median_future_max_drawdown_63d
            FROM _downside_targets
            GROUP BY PERIODOFREPORT, report_period, information_date
            ORDER BY report_period
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """,
        [str(output_path)],
    )


def _validate_outputs(
    output_paths: dict[str, Path],
    security_nodes_path: Path,
) -> tuple[tuple[DownsideTargetTableResult, ...], int, int]:
    """Validate target schemas, keys, chronology, and primary-target bounds."""
    row_counts = {
        name: _validate_output_schema(path, _OUTPUT_COLUMNS[name])
        for name, path in output_paths.items()
    }
    connection = duckdb.connect()

    try:
        input_rows = connection.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(security_nodes_path)],
        ).fetchone()[0]
        checks = connection.execute(
            f"""
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, permno)),
                COUNT(*) FILTER (
                    WHERE target_horizon_market_days
                            != {_TARGET_HORIZON_MARKET_DAYS}
                        OR target_window_start <= information_date
                        OR target_window_end < target_window_start
                        OR return_observations_63d
                            > {_TARGET_HORIZON_MARKET_DAYS}
                        OR future_max_drawdown_63d < 0
                        OR future_max_drawdown_63d > 1
                        OR delisting_return_observed AND NOT delisted_in_window
                        OR target_usable AND (
                            NOT target_window_complete
                            OR future_max_drawdown_63d IS NULL
                            OR (
                                delisted_in_window
                                AND NOT delisting_return_observed
                            )
                            OR (
                                return_observations_63d
                                    < {_MINIMUM_RETURN_OBSERVATIONS}
                                AND NOT delisted_in_window
                            )
                        )
                ),
                COUNT(*) FILTER (WHERE target_usable)
            FROM read_parquet(?)
            """,
            [str(output_paths["downside_targets"])],
        ).fetchone()
    finally:
        connection.close()

    if checks[0] != input_rows or checks[0] != checks[1] or checks[2]:
        raise RuntimeError("Downside-target invariants are not satisfied.")

    tables = tuple(
        DownsideTargetTableResult(
            name=name,
            path=path,
            row_count=row_counts[name],
        )
        for name, path in output_paths.items()
    )

    return tables, checks[0], checks[3]


def build_downside_targets(
    daily_path: Path,
    security_nodes_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> DownsideTargetBuildResult:
    """Build or validate future downside-target outputs."""
    daily_path = Path(daily_path)
    security_nodes_path = Path(security_nodes_path)
    output_directory = Path(output_directory)
    _require_parquet_columns(daily_path, _DAILY_COLUMNS, "CRSP daily data")
    _require_parquet_columns(
        security_nodes_path,
        _SECURITY_COLUMNS,
        "ownership-network security nodes",
    )
    output_paths = _get_output_paths(output_directory)
    existing_paths = [path for path in output_paths.values() if path.exists()]

    if existing_paths and not overwrite:
        if len(existing_paths) != len(output_paths):
            raise RuntimeError(
                "Downside-target outputs are incomplete. Rebuild with overwrite=True."
            )

        newest_input = max(
            daily_path.stat().st_mtime,
            security_nodes_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
        )
        oldest_output = min(path.stat().st_mtime for path in output_paths.values())

        if newest_input > oldest_output:
            raise RuntimeError(
                "Downside-target inputs or code are newer than the outputs. "
                "Rebuild with overwrite=True."
            )

        tables, security_periods, usable_security_periods = _validate_outputs(
            output_paths,
            security_nodes_path,
        )

        return DownsideTargetBuildResult(
            created=False,
            tables=tables,
            security_periods=security_periods,
            usable_security_periods=usable_security_periods,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_name(f".{path.name}.part")
        for name, path in output_paths.items()
    }

    for temporary_path in temporary_paths.values():
        temporary_path.unlink(missing_ok=True)

    try:
        connection = duckdb.connect()
        connection.execute("SET memory_limit = '8GB'")
        connection.execute("SET threads = 4")
        connection.execute("SET preserve_insertion_order = false")

        try:
            _register_inputs(connection, daily_path, security_nodes_path)
            _create_downside_targets(
                connection,
                temporary_paths["downside_targets"],
            )
            _create_target_audit(
                connection,
                temporary_paths["downside_targets"],
                temporary_paths["target_construction_audit"],
            )
        finally:
            connection.close()

        tables, security_periods, usable_security_periods = _validate_outputs(
            temporary_paths,
            security_nodes_path,
        )

        for name, temporary_path in temporary_paths.items():
            temporary_path.replace(output_paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    final_tables = tuple(
        DownsideTargetTableResult(
            name=table.name,
            path=output_paths[table.name],
            row_count=table.row_count,
        )
        for table in tables
    )

    return DownsideTargetBuildResult(
        created=True,
        tables=final_tables,
        security_periods=security_periods,
        usable_security_periods=usable_security_periods,
    )


__all__ = [
    "DownsideTargetBuildResult",
    "DownsideTargetTableResult",
    "build_downside_targets",
]
