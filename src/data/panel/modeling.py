"""Assemble the point-in-time stock-quarter modeling panel."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_SECURITY_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "permno",
    "permco",
    "CUSIP",
    "ticker",
    "securitynm",
    "primaryexch",
    "issuertype",
    "institutional_holder_count",
    "total_reported_value_usd",
    "reported_ownership_hhi",
    "top_five_reported_ownership_share",
    "mean_owner_portfolio_weight",
    "maximum_owner_portfolio_weight",
}
_NETWORK_SUMMARY_COLUMNS = {
    "report_period",
    "managers",
    "securities",
    "ownership_edges",
    "bipartite_density",
}
_MARKET_COLUMNS = {
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
}
_TARGET_COLUMNS = {
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
}
_OUTPUT_COLUMNS = {
    "stock_quarter_panel": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "permno",
        "permco",
        "CUSIP",
        "ticker",
        "securitynm",
        "primaryexch",
        "issuertype",
        "institutional_holder_count",
        "total_reported_value_usd",
        "reported_ownership_hhi",
        "top_five_reported_ownership_share",
        "mean_owner_portfolio_weight",
        "maximum_owner_portfolio_weight",
        "manager_count",
        "network_security_count",
        "ownership_edges",
        "bipartite_density",
        "manager_holder_share",
        "previous_report_period",
        "quarters_since_previous_observation",
        "institutional_holder_count_change",
        "institutional_holder_count_change_percent",
        "total_reported_value_change_percent",
        "reported_ownership_hhi_change",
        "top_five_reported_ownership_share_change",
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
        "target_horizon_market_days",
        "target_window_start",
        "target_window_end",
        "target_return_observations_63d",
        "delisted_in_target_window",
        "delisting_return_observed_in_target_window",
        "future_max_drawdown_63d",
        "future_cumulative_return_63d",
        "future_downside_volatility_63d",
        "future_worst_five_day_return_63d",
        "future_worst_daily_return_63d",
        "target_window_complete",
        "target_usable",
        "model_eligible",
    ),
    "stock_quarter_panel_audit": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "security_periods",
        "market_feature_usable_periods",
        "target_usable_periods",
        "model_eligible_periods",
        "model_eligible_share",
        "median_future_max_drawdown_63d",
    ),
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelingPanelTableResult:
    """Metadata for one modeling-panel output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class ModelingPanelBuildResult:
    """Result of creating or loading the modeling-panel outputs."""

    created: bool
    tables: tuple[ModelingPanelTableResult, ...]
    security_periods: int
    model_eligible_periods: int


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
    """Return the modeling-panel output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _validate_output_schema(path: Path, expected_columns: tuple[str, ...]) -> int:
    """Validate one output schema and return its row count."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid modeling-panel output: {path}") from error

    actual_columns = tuple(parquet_file.schema_arrow.names)

    if actual_columns != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: "
            f"expected {list(expected_columns)}, found {list(actual_columns)}."
        )

    row_count = parquet_file.metadata.num_rows

    if row_count == 0:
        raise RuntimeError(f"Modeling-panel output is empty: {path}")

    return row_count


def _create_panel(
    connection: duckdb.DuckDBPyConnection,
    output_path: Path,
) -> None:
    """Join ownership, network, market, and target information."""
    _LOGGER.info("Assembling the point-in-time stock-quarter modeling panel.")
    connection.execute(
        """
        COPY (
            WITH ownership AS (
                SELECT
                    s.PERIODOFREPORT,
                    s.report_period,
                    s.information_date,
                    s.permno,
                    s.permco,
                    s.CUSIP,
                    s.ticker,
                    s.securitynm,
                    s.primaryexch,
                    s.issuertype,
                    s.institutional_holder_count,
                    s.total_reported_value_usd,
                    s.reported_ownership_hhi,
                    s.top_five_reported_ownership_share,
                    s.mean_owner_portfolio_weight,
                    s.maximum_owner_portfolio_weight,
                    n.managers AS manager_count,
                    n.securities AS network_security_count,
                    n.ownership_edges,
                    n.bipartite_density,
                    s.institutional_holder_count / n.managers
                        AS manager_holder_share
                FROM _security_nodes s
                JOIN _network_summary n USING (report_period)
            ),
            ownership_lags AS (
                SELECT
                    *,
                    LAG(report_period) OVER security_history
                        AS previous_report_period,
                    LAG(institutional_holder_count) OVER security_history
                        AS previous_holder_count,
                    LAG(total_reported_value_usd) OVER security_history
                        AS previous_reported_value,
                    LAG(reported_ownership_hhi) OVER security_history
                        AS previous_ownership_hhi,
                    LAG(top_five_reported_ownership_share) OVER security_history
                        AS previous_top_five_share
                FROM ownership
                WINDOW security_history AS (
                    PARTITION BY permno ORDER BY report_period
                )
            ),
            ownership_changes AS (
                SELECT
                    *,
                    DATE_DIFF('quarter', previous_report_period, report_period)
                        AS quarters_since_previous_observation
                FROM ownership_lags
            )
            SELECT
                o.PERIODOFREPORT,
                o.report_period,
                o.information_date,
                o.permno,
                o.permco,
                o.CUSIP,
                o.ticker,
                o.securitynm,
                o.primaryexch,
                o.issuertype,
                o.institutional_holder_count,
                o.total_reported_value_usd,
                o.reported_ownership_hhi,
                o.top_five_reported_ownership_share,
                o.mean_owner_portfolio_weight,
                o.maximum_owner_portfolio_weight,
                o.manager_count,
                o.network_security_count,
                o.ownership_edges,
                o.bipartite_density,
                o.manager_holder_share,
                o.previous_report_period,
                o.quarters_since_previous_observation,
                CASE WHEN o.quarters_since_previous_observation = 1
                    THEN o.institutional_holder_count - o.previous_holder_count
                END AS institutional_holder_count_change,
                CASE WHEN o.quarters_since_previous_observation = 1
                    THEN 100.0 * (
                        o.institutional_holder_count - o.previous_holder_count
                    ) / NULLIF(o.previous_holder_count, 0)
                END AS institutional_holder_count_change_percent,
                CASE WHEN o.quarters_since_previous_observation = 1
                    THEN 100.0 * (
                        o.total_reported_value_usd - o.previous_reported_value
                    ) / NULLIF(o.previous_reported_value, 0)
                END AS total_reported_value_change_percent,
                CASE WHEN o.quarters_since_previous_observation = 1
                    THEN o.reported_ownership_hhi - o.previous_ownership_hhi
                END AS reported_ownership_hhi_change,
                CASE WHEN o.quarters_since_previous_observation = 1
                    THEN o.top_five_reported_ownership_share
                        - o.previous_top_five_share
                END AS top_five_reported_ownership_share_change,
                m.feature_window_start,
                m.feature_window_end,
                m.return_observations_21d,
                m.return_observations_63d,
                m.return_observations_126d,
                m.return_observations_252d,
                m.beta_return_observations_252d,
                m.market_price_usd,
                m.market_cap_usd,
                m.average_daily_dollar_volume_63d,
                m.return_21d,
                m.return_63d,
                m.return_126d,
                m.return_252d,
                m.momentum_252_21d,
                m.volatility_63d,
                m.volatility_252d,
                m.downside_volatility_63d,
                m.downside_volatility_252d,
                m.market_beta_252d,
                m.negative_return_share_63d,
                m.maximum_drawdown_252d,
                m.market_feature_usable,
                t.target_horizon_market_days,
                t.target_window_start,
                t.target_window_end,
                t.return_observations_63d AS target_return_observations_63d,
                t.delisted_in_window AS delisted_in_target_window,
                t.delisting_return_observed
                    AS delisting_return_observed_in_target_window,
                t.future_max_drawdown_63d,
                t.future_cumulative_return_63d,
                t.future_downside_volatility_63d,
                t.future_worst_five_day_return_63d,
                t.future_worst_daily_return_63d,
                t.target_window_complete,
                t.target_usable,
                m.market_feature_usable AND t.target_usable AS model_eligible
            FROM ownership_changes o
            JOIN _market_features m USING (
                report_period,
                information_date,
                permno
            )
            JOIN _downside_targets t USING (
                report_period,
                information_date,
                permno
            )
            ORDER BY o.report_period, o.permno
        ) TO ? (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 100000
        )
        """,
        [str(output_path)],
    )


def _create_audit(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    output_path: Path,
) -> None:
    """Create quarterly final-sample coverage diagnostics."""
    connection.read_parquet(str(panel_path)).create_view("_stock_quarter_panel")
    connection.execute(
        """
        COPY (
            SELECT
                PERIODOFREPORT,
                report_period,
                information_date,
                COUNT(*) AS security_periods,
                COUNT(*) FILTER (WHERE market_feature_usable)
                    AS market_feature_usable_periods,
                COUNT(*) FILTER (WHERE target_usable) AS target_usable_periods,
                COUNT(*) FILTER (WHERE model_eligible) AS model_eligible_periods,
                AVG(CAST(model_eligible AS DOUBLE)) AS model_eligible_share,
                MEDIAN(future_max_drawdown_63d) FILTER (WHERE model_eligible)
                    AS median_future_max_drawdown_63d
            FROM _stock_quarter_panel
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
) -> tuple[tuple[ModelingPanelTableResult, ...], int, int]:
    """Validate final-panel schemas, keys, chronology, and eligibility."""
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
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, permno)),
                COUNT(*) FILTER (
                    WHERE information_date <= report_period
                        OR feature_window_end >= information_date
                        OR target_window_start <= information_date
                        OR feature_window_end >= target_window_start
                        OR model_eligible IS DISTINCT FROM (
                            market_feature_usable AND target_usable
                        )
                        OR model_eligible AND future_max_drawdown_63d IS NULL
                ),
                COUNT(*) FILTER (WHERE model_eligible)
            FROM read_parquet(?)
            """,
            [str(output_paths["stock_quarter_panel"])],
        ).fetchone()
    finally:
        connection.close()

    if checks[0] != input_rows or checks[0] != checks[1] or checks[2]:
        raise RuntimeError("Modeling-panel invariants are not satisfied.")

    tables = tuple(
        ModelingPanelTableResult(
            name=name,
            path=path,
            row_count=row_counts[name],
        )
        for name, path in output_paths.items()
    )

    return tables, checks[0], checks[3]


def build_stock_quarter_modeling_panel(
    security_nodes_path: Path,
    network_summary_path: Path,
    market_features_path: Path,
    downside_targets_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> ModelingPanelBuildResult:
    """Build or validate the point-in-time stock-quarter modeling panel."""
    security_nodes_path = Path(security_nodes_path)
    network_summary_path = Path(network_summary_path)
    market_features_path = Path(market_features_path)
    downside_targets_path = Path(downside_targets_path)
    output_directory = Path(output_directory)
    inputs = (
        (security_nodes_path, _SECURITY_COLUMNS, "security-node data"),
        (network_summary_path, _NETWORK_SUMMARY_COLUMNS, "network summary"),
        (market_features_path, _MARKET_COLUMNS, "market features"),
        (downside_targets_path, _TARGET_COLUMNS, "downside targets"),
    )

    for path, required_columns, description in inputs:
        _require_parquet_columns(path, required_columns, description)

    output_paths = _get_output_paths(output_directory)
    existing_paths = [path for path in output_paths.values() if path.exists()]

    if existing_paths and not overwrite:
        if len(existing_paths) != len(output_paths):
            raise RuntimeError(
                "Modeling-panel outputs are incomplete. Rebuild with overwrite=True."
            )

        newest_input = max(
            *(path.stat().st_mtime for path, _, _ in inputs),
            Path(__file__).stat().st_mtime,
        )
        oldest_output = min(path.stat().st_mtime for path in output_paths.values())

        if newest_input > oldest_output:
            raise RuntimeError(
                "Modeling-panel inputs or code are newer than the outputs. "
                "Rebuild with overwrite=True."
            )

        tables, security_periods, model_eligible_periods = _validate_outputs(
            output_paths,
            security_nodes_path,
        )

        return ModelingPanelBuildResult(
            created=False,
            tables=tables,
            security_periods=security_periods,
            model_eligible_periods=model_eligible_periods,
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
            for name, path in {
                "_security_nodes": security_nodes_path,
                "_network_summary": network_summary_path,
                "_market_features": market_features_path,
                "_downside_targets": downside_targets_path,
            }.items():
                connection.read_parquet(str(path)).create_view(name)

            _create_panel(
                connection,
                temporary_paths["stock_quarter_panel"],
            )
            _create_audit(
                connection,
                temporary_paths["stock_quarter_panel"],
                temporary_paths["stock_quarter_panel_audit"],
            )
        finally:
            connection.close()

        tables, security_periods, model_eligible_periods = _validate_outputs(
            temporary_paths,
            security_nodes_path,
        )

        for name, temporary_path in temporary_paths.items():
            temporary_path.replace(output_paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    final_tables = tuple(
        ModelingPanelTableResult(
            name=table.name,
            path=output_paths[table.name],
            row_count=table.row_count,
        )
        for table in tables
    )

    return ModelingPanelBuildResult(
        created=True,
        tables=final_tables,
        security_periods=security_periods,
        model_eligible_periods=model_eligible_periods,
    )


__all__ = [
    "ModelingPanelBuildResult",
    "ModelingPanelTableResult",
    "build_stock_quarter_modeling_panel",
]
