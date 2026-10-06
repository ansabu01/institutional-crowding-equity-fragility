"""Build point-in-time CRSP market features for stock-quarter observations."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_DAILY_COLUMNS = {
    "permno",
    "dlycaldt",
    "conditionaltype",
    "tradingstatusflg",
    "dlyprc",
    "dlycap",
    "dlyret",
    "dlyvol",
}
_MARKET_INDEX_COLUMNS = {
    "dlycaldt",
    "dlytotret",
}
_SECURITY_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "permno",
}
_OUTPUT_COLUMNS = {
    "market_features": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "permno",
        "feature_window_start",
        "feature_window_end",
        "return_observations_21d",
        "return_observations_63d",
        "return_observations_126d",
        "return_observations_252d",
        "beta_return_observations_252d",
        "market_price_usd",
        "market_cap_usd",
        "average_daily_dollar_volume_63d",
        "return_21d",
        "return_63d",
        "return_126d",
        "return_252d",
        "momentum_252_21d",
        "volatility_63d",
        "volatility_252d",
        "downside_volatility_63d",
        "downside_volatility_252d",
        "market_beta_252d",
        "negative_return_share_63d",
        "maximum_drawdown_252d",
        "market_feature_usable",
    ),
    "market_feature_audit": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "security_periods",
        "usable_security_periods",
        "usable_share",
        "median_return_observations_252d",
        "missing_market_cap",
        "missing_market_beta",
        "missing_primary_features",
    ),
}
_MINIMUM_252D_RETURN_OBSERVATIONS = 202
_MINIMUM_63D_RETURN_OBSERVATIONS = 50
_DUCKDB_MEMORY_LIMIT = "8GB"
_DUCKDB_THREADS = 4
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketFeatureTableResult:
    """Metadata for one market-feature output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class MarketFeatureBuildResult:
    """Result of creating or loading the market-feature outputs."""

    created: bool
    tables: tuple[MarketFeatureTableResult, ...]
    security_periods: int
    usable_security_periods: int


def _open_connection() -> duckdb.DuckDBPyConnection:
    """Open a bounded-memory DuckDB connection."""
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = '{_DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET threads = {_DUCKDB_THREADS}")
    connection.execute("SET preserve_insertion_order = false")

    return connection


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
    """Return the market-feature output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _validate_output_schema(path: Path, expected_columns: tuple[str, ...]) -> int:
    """Validate one output schema and return its row count."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid market-feature output: {path}") from error

    actual_columns = tuple(parquet_file.schema_arrow.names)

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: "
            f"expected {list(expected_columns)}, found {list(actual_columns)}."
        )

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"Market-feature output is empty: {path}")

    return row_count


def _register_inputs(
    connection: duckdb.DuckDBPyConnection,
    daily_stock_path: Path,
    market_index_path: Path,
    security_nodes_path: Path,
) -> None:
    """Register the raw market inputs and prediction observations."""
    connection.read_parquet(str(daily_stock_path)).create_view("_daily_stock")
    connection.read_parquet(str(market_index_path)).create_view(
        "_daily_market_index"
    )
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
            FROM _daily_market_index
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
        """
        CREATE TEMP TABLE _feature_windows AS
        WITH prediction_dates AS (
            SELECT DISTINCT information_date
            FROM _predictions
        ),
        last_days AS (
            SELECT
                p.information_date,
                c.market_day,
                c.dlycaldt AS feature_window_end
            FROM prediction_dates p
            ASOF LEFT JOIN _market_calendar c
                ON p.information_date > c.dlycaldt
        )
        SELECT
            p.*,
            s.dlycaldt AS feature_window_start,
            e.feature_window_end,
            e.market_day AS feature_window_end_index
        FROM _predictions p
        LEFT JOIN last_days e USING (information_date)
        LEFT JOIN _market_calendar s
            ON s.market_day = e.market_day - 251
        """
    )


def _create_market_features(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Create the point-in-time stock-quarter market-feature table."""
    _LOGGER.info("Building point-in-time CRSP market features.")
    connection.execute(
        f"""
        COPY (
            WITH feature_rows AS (
                SELECT
                    w.*,
                    c.market_day,
                    d.dlycaldt,
                    d.dlyprc,
                    d.dlycap,
                    d.dlyret,
                    i.dlytotret,
                    d.dlyvol,
                    PRODUCT(1.0 + d.dlyret) OVER (
                        PARTITION BY w.report_period, w.permno
                        ORDER BY d.dlycaldt
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS wealth
                FROM _feature_windows w
                JOIN _daily_stock d
                    ON d.permno = w.permno
                    AND d.dlycaldt BETWEEN
                        w.feature_window_start AND w.feature_window_end
                JOIN _market_calendar c USING (dlycaldt)
                JOIN _daily_market_index i USING (dlycaldt)
                WHERE
                    d.conditionaltype = 'RW'
                    AND d.tradingstatusflg = 'A'
            ),
            drawdown_rows AS (
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
                FROM feature_rows
            ),
            features AS (
                SELECT
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno,
                    MIN(feature_window_start) AS feature_window_start,
                    MAX(feature_window_end) AS feature_window_end,
                    COUNT(dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 21
                    ) AS return_observations_21d,
                    COUNT(dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    ) AS return_observations_63d,
                    COUNT(dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 126
                    ) AS return_observations_126d,
                    COUNT(dlyret) AS return_observations_252d,
                    COUNT(dlyret) FILTER (WHERE dlytotret IS NOT NULL)
                        AS beta_return_observations_252d,
                    ARG_MAX(dlyprc, dlycaldt) AS market_price_usd,
                    ARG_MAX(dlycap, dlycaldt) * 1000.0 AS market_cap_usd,
                    AVG(ABS(dlyprc) * dlyvol) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    ) AS average_daily_dollar_volume_63d,
                    PRODUCT(1.0 + dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 21
                    ) - 1.0 AS return_21d,
                    PRODUCT(1.0 + dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    ) - 1.0 AS return_63d,
                    PRODUCT(1.0 + dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 126
                    ) - 1.0 AS return_126d,
                    PRODUCT(1.0 + dlyret) - 1.0 AS return_252d,
                    PRODUCT(1.0 + dlyret) FILTER (
                        WHERE market_day <= feature_window_end_index - 21
                    ) - 1.0 AS momentum_252_21d,
                    STDDEV_SAMP(dlyret) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    ) * SQRT(252.0) AS volatility_63d,
                    STDDEV_SAMP(dlyret) * SQRT(252.0) AS volatility_252d,
                    SQRT(252.0 * AVG(POW(LEAST(dlyret, 0.0), 2)) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    )) AS downside_volatility_63d,
                    SQRT(252.0 * AVG(POW(LEAST(dlyret, 0.0), 2)))
                        AS downside_volatility_252d,
                    REGR_SLOPE(dlyret, dlytotret) AS market_beta_252d,
                    AVG(CAST(dlyret < 0 AS DOUBLE)) FILTER (
                        WHERE market_day > feature_window_end_index - 63
                    ) AS negative_return_share_63d,
                    -MIN(drawdown) AS maximum_drawdown_252d
                FROM drawdown_rows
                GROUP BY
                    PERIODOFREPORT,
                    report_period,
                    information_date,
                    permno
            )
            SELECT
                w.PERIODOFREPORT,
                w.report_period,
                w.information_date,
                w.permno,
                w.feature_window_start,
                w.feature_window_end,
                COALESCE(f.return_observations_21d, 0)
                    AS return_observations_21d,
                COALESCE(f.return_observations_63d, 0)
                    AS return_observations_63d,
                COALESCE(f.return_observations_126d, 0)
                    AS return_observations_126d,
                COALESCE(f.return_observations_252d, 0)
                    AS return_observations_252d,
                COALESCE(f.beta_return_observations_252d, 0)
                    AS beta_return_observations_252d,
                f.market_price_usd,
                f.market_cap_usd,
                f.average_daily_dollar_volume_63d,
                f.return_21d,
                f.return_63d,
                f.return_126d,
                f.return_252d,
                f.momentum_252_21d,
                f.volatility_63d,
                f.volatility_252d,
                f.downside_volatility_63d,
                f.downside_volatility_252d,
                f.market_beta_252d,
                f.negative_return_share_63d,
                f.maximum_drawdown_252d,
                COALESCE((
                    w.feature_window_start IS NOT NULL
                    AND f.return_observations_63d
                        >= {_MINIMUM_63D_RETURN_OBSERVATIONS}
                    AND f.return_observations_252d
                        >= {_MINIMUM_252D_RETURN_OBSERVATIONS}
                    AND f.beta_return_observations_252d
                        >= {_MINIMUM_252D_RETURN_OBSERVATIONS}
                    AND f.market_beta_252d IS NOT NULL
                    AND f.market_price_usd > 0
                    AND f.market_cap_usd > 0
                ), false) AS market_feature_usable
            FROM _feature_windows w
            LEFT JOIN features f USING (
                PERIODOFREPORT,
                report_period,
                information_date,
                permno
            )
            ORDER BY w.report_period, w.permno
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """,
        [str(output_path)],
    )


def _create_market_feature_audit(
    connection: duckdb.DuckDBPyConnection,
    features_path: Path,
    output_path: Path,
) -> None:
    """Create quarterly feature coverage and missingness diagnostics."""
    connection.read_parquet(str(features_path)).create_view("_market_features")
    connection.execute(
        """
        COPY (
            SELECT
                PERIODOFREPORT,
                report_period,
                information_date,
                COUNT(*) AS security_periods,
                COUNT(*) FILTER (WHERE market_feature_usable)
                    AS usable_security_periods,
                AVG(CAST(market_feature_usable AS DOUBLE)) AS usable_share,
                MEDIAN(return_observations_252d)
                    AS median_return_observations_252d,
                COUNT(*) FILTER (WHERE market_cap_usd IS NULL)
                    AS missing_market_cap,
                COUNT(*) FILTER (WHERE market_beta_252d IS NULL)
                    AS missing_market_beta,
                COUNT(*) FILTER (
                    WHERE return_252d IS NULL
                        OR volatility_252d IS NULL
                        OR market_beta_252d IS NULL
                        OR maximum_drawdown_252d IS NULL
                ) AS missing_primary_features
            FROM _market_features
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
) -> tuple[tuple[MarketFeatureTableResult, ...], int, int]:
    """Validate market-feature schemas, keys, timing, and coverage."""
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
                    WHERE feature_window_end >= information_date
                        OR feature_window_start > feature_window_end
                        OR return_observations_21d > 21
                        OR return_observations_63d > 63
                        OR return_observations_126d > 126
                        OR return_observations_252d > 252
                        OR beta_return_observations_252d > 252
                        OR maximum_drawdown_252d < 0
                        OR maximum_drawdown_252d > 1
                        OR market_feature_usable AND (
                            beta_return_observations_252d
                                < {_MINIMUM_252D_RETURN_OBSERVATIONS}
                            OR market_beta_252d IS NULL
                        )
                ),
                COUNT(*) FILTER (WHERE market_feature_usable)
            FROM read_parquet(?)
            """,
            [str(output_paths["market_features"])],
        ).fetchone()
    finally:
        connection.close()

    if checks[0] != input_rows or checks[0] != checks[1] or checks[2]:
        raise RuntimeError("Market-feature output invariants are not satisfied.")

    tables = tuple(
        MarketFeatureTableResult(
            name=name,
            path=path,
            row_count=row_counts[name],
        )
        for name, path in output_paths.items()
    )

    return tables, checks[0], checks[3]


def build_crsp_market_features(
    daily_stock_path: Path,
    market_index_path: Path,
    security_nodes_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> MarketFeatureBuildResult:
    """Build or validate point-in-time CRSP market-feature outputs."""
    daily_stock_path = Path(daily_stock_path)
    market_index_path = Path(market_index_path)
    security_nodes_path = Path(security_nodes_path)
    output_directory = Path(output_directory)
    _require_parquet_columns(
        daily_stock_path,
        _DAILY_COLUMNS,
        "CRSP daily stock data",
    )
    _require_parquet_columns(
        market_index_path,
        _MARKET_INDEX_COLUMNS,
        "CRSP daily market-index data",
    )
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
                "Market-feature outputs are incomplete. Rebuild with overwrite=True."
            )

        newest_input = max(
            daily_stock_path.stat().st_mtime,
            market_index_path.stat().st_mtime,
            security_nodes_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
        )
        oldest_output = min(path.stat().st_mtime for path in output_paths.values())

        if newest_input > oldest_output:
            raise RuntimeError(
                "Market-feature inputs or code are newer than the outputs. "
                "Rebuild with overwrite=True."
            )

        tables, security_periods, usable_security_periods = _validate_outputs(
            output_paths,
            security_nodes_path,
        )

        return MarketFeatureBuildResult(
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
        connection = _open_connection()

        try:
            _register_inputs(
                connection,
                daily_stock_path,
                market_index_path,
                security_nodes_path,
            )
            _create_market_features(
                connection,
                temporary_paths["market_features"],
            )
            _create_market_feature_audit(
                connection,
                temporary_paths["market_features"],
                temporary_paths["market_feature_audit"],
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
        MarketFeatureTableResult(
            name=table.name,
            path=output_paths[table.name],
            row_count=table.row_count,
        )
        for table in tables
    )

    return MarketFeatureBuildResult(
        created=True,
        tables=final_tables,
        security_periods=security_periods,
        usable_security_periods=usable_security_periods,
    )


__all__ = [
    "MarketFeatureBuildResult",
    "MarketFeatureTableResult",
    "build_crsp_market_features",
]
