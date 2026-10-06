"""Build the frozen stock-level ownership-network features."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


_HOLDING_COLUMNS = {
    "CIK",
    "report_period",
    "information_date",
    "permno",
    "position_value_usd",
}
_SIMILARITY_COLUMNS = {
    "report_period",
    "information_date",
    "manager_a_cik",
    "manager_b_cik",
    "cosine_similarity",
}
_COMMUNITY_COLUMNS = {
    "CIK",
    "report_period",
    "information_date",
    "similarity_degree",
    "mean_neighbor_similarity",
    "community_id",
}
_OUTPUT_COLUMNS = (
    "report_period",
    "information_date",
    "permno",
    "owner_similarity_score",
    "value_weighted_owner_degree_centrality",
    "value_weighted_owner_neighbor_similarity",
    "owner_community_count",
    "owner_community_hhi",
    "largest_owner_community_share",
    "owner_similarity_score_change",
    "owner_community_hhi_change",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StockNetworkFeatureResult:
    """Result of creating or loading the stock-network features."""

    created: bool
    path: Path
    row_count: int
    report_periods: int


def _require_parquet(
    path: Path,
    required_columns: set[str],
    description: str,
) -> None:
    """Check that an input Parquet contains the required columns."""
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


def _create_features(
    holdings_path: Path,
    similarity_edges_path: Path,
    manager_communities_path: Path,
    output_path: Path,
) -> None:
    """Construct the frozen static and quarterly-change features."""
    connection = duckdb.connect()

    try:
        connection.read_parquet(str(holdings_path)).create_view("_holdings")
        connection.read_parquet(str(similarity_edges_path)).create_view(
            "_similarity_edges"
        )
        connection.read_parquet(str(manager_communities_path)).create_view(
            "_manager_communities"
        )
        connection.execute(
            """
            COPY (
                WITH manager_counts AS (
                    SELECT
                        report_period,
                        information_date,
                        COUNT(*) AS manager_count
                    FROM _manager_communities
                    GROUP BY report_period, information_date
                ),
                owners AS (
                    SELECT
                        h.report_period,
                        h.information_date,
                        h.permno,
                        h.CIK,
                        CAST(h.position_value_usd AS DOUBLE)
                            AS position_value_usd,
                        c.similarity_degree,
                        c.mean_neighbor_similarity,
                        c.community_id,
                        m.manager_count,
                        SUM(CAST(h.position_value_usd AS DOUBLE)) OVER (
                            PARTITION BY h.report_period, h.permno
                        ) AS stock_position_value
                    FROM _holdings h
                    JOIN _manager_communities c USING (
                        report_period,
                        information_date,
                        CIK
                    )
                    JOIN manager_counts m USING (
                        report_period,
                        information_date
                    )
                ),
                owner_summary AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        COUNT(*) AS owner_count,
                        SUM(
                            position_value_usd
                            / NULLIF(stock_position_value, 0)
                            * similarity_degree
                            / NULLIF(manager_count - 1, 0)
                        ) AS value_weighted_owner_degree_centrality,
                        SUM(
                            position_value_usd
                            / NULLIF(stock_position_value, 0)
                            * mean_neighbor_similarity
                        ) AS value_weighted_owner_neighbor_similarity
                    FROM owners
                    GROUP BY report_period, information_date, permno
                ),
                community_positions AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        community_id,
                        SUM(position_value_usd) AS community_position_value,
                        MAX(stock_position_value) AS stock_position_value
                    FROM owners
                    GROUP BY
                        report_period,
                        information_date,
                        permno,
                        community_id
                ),
                community_summary AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        COUNT(*) AS owner_community_count,
                        SUM(
                            POWER(
                                community_position_value
                                / NULLIF(stock_position_value, 0),
                                2
                            )
                        ) AS owner_community_hhi,
                        MAX(
                            community_position_value
                            / NULLIF(stock_position_value, 0)
                        ) AS largest_owner_community_share
                    FROM community_positions
                    GROUP BY report_period, information_date, permno
                ),
                holding_keys AS (
                    SELECT
                        report_period,
                        information_date,
                        permno,
                        CIK
                    FROM _holdings
                ),
                linked_owner_pairs AS (
                    SELECT
                        a.report_period,
                        a.information_date,
                        a.permno,
                        SUM(e.cosine_similarity) AS linked_similarity
                    FROM _similarity_edges e
                    JOIN holding_keys a
                        ON e.report_period = a.report_period
                        AND e.information_date = a.information_date
                        AND e.manager_a_cik = a.CIK
                    JOIN holding_keys b
                        ON e.report_period = b.report_period
                        AND e.information_date = b.information_date
                        AND e.manager_b_cik = b.CIK
                        AND a.permno = b.permno
                    GROUP BY a.report_period, a.information_date, a.permno
                ),
                static_features AS (
                    SELECT
                        o.report_period,
                        o.information_date,
                        o.permno,
                        CASE
                            WHEN o.owner_count < 2 THEN 0.0
                            ELSE COALESCE(p.linked_similarity, 0.0)
                                / (o.owner_count * (o.owner_count - 1) / 2.0)
                        END AS owner_similarity_score,
                        o.value_weighted_owner_degree_centrality,
                        o.value_weighted_owner_neighbor_similarity,
                        c.owner_community_count,
                        c.owner_community_hhi,
                        c.largest_owner_community_share
                    FROM owner_summary o
                    JOIN community_summary c USING (
                        report_period,
                        information_date,
                        permno
                    )
                    LEFT JOIN linked_owner_pairs p USING (
                        report_period,
                        information_date,
                        permno
                    )
                ),
                feature_lags AS (
                    SELECT
                        *,
                        LAG(report_period) OVER security_history
                            AS previous_report_period,
                        LAG(owner_similarity_score) OVER security_history
                            AS previous_owner_similarity_score,
                        LAG(owner_community_hhi) OVER security_history
                            AS previous_owner_community_hhi
                    FROM static_features
                    WINDOW security_history AS (
                        PARTITION BY permno ORDER BY report_period
                    )
                )
                SELECT
                    report_period,
                    information_date,
                    permno,
                    owner_similarity_score,
                    value_weighted_owner_degree_centrality,
                    value_weighted_owner_neighbor_similarity,
                    owner_community_count,
                    owner_community_hhi,
                    largest_owner_community_share,
                    CASE
                        WHEN DATE_DIFF(
                            'quarter',
                            previous_report_period,
                            report_period
                        ) = 1
                        THEN owner_similarity_score
                            - previous_owner_similarity_score
                    END AS owner_similarity_score_change,
                    CASE
                        WHEN DATE_DIFF(
                            'quarter',
                            previous_report_period,
                            report_period
                        ) = 1
                        THEN owner_community_hhi
                            - previous_owner_community_hhi
                    END AS owner_community_hhi_change
                FROM feature_lags
                ORDER BY report_period, permno
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(output_path)],
        )
    finally:
        connection.close()


def _validate_output(path: Path) -> tuple[int, int]:
    """Validate the output structure and return rows and periods."""
    try:
        parquet_file = pq.ParquetFile(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid stock-network feature file: {path}") from error

    if tuple(parquet_file.schema_arrow.names) != _OUTPUT_COLUMNS:
        raise RuntimeError(f"Unexpected stock-network feature columns: {path}")

    connection = duckdb.connect()

    try:
        row_count, unique_rows, report_periods, invalid_rows = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, permno)),
                COUNT(DISTINCT report_period),
                COUNT(*) FILTER (
                    WHERE owner_similarity_score IS NULL
                        OR value_weighted_owner_degree_centrality IS NULL
                        OR value_weighted_owner_neighbor_similarity IS NULL
                        OR owner_community_count IS NULL
                        OR owner_community_hhi IS NULL
                        OR largest_owner_community_share IS NULL
                        OR owner_similarity_score NOT BETWEEN 0 AND 1
                        OR value_weighted_owner_degree_centrality NOT BETWEEN 0 AND 1
                        OR value_weighted_owner_neighbor_similarity NOT BETWEEN 0 AND 1
                        OR owner_community_count < 1
                        OR owner_community_hhi NOT BETWEEN 0 AND 1
                        OR largest_owner_community_share NOT BETWEEN 0 AND 1
                        OR owner_similarity_score_change NOT BETWEEN -1 AND 1
                        OR owner_community_hhi_change NOT BETWEEN -1 AND 1
                )
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
    finally:
        connection.close()

    if row_count == 0:
        raise RuntimeError(f"Stock-network feature file is empty: {path}")

    if row_count != unique_rows:
        raise RuntimeError("Stock-network features contain duplicate stock-quarters.")

    if invalid_rows:
        raise RuntimeError("Stock-network features contain invalid values.")

    return row_count, report_periods


def build_stock_network_features(
    holdings_path: Path,
    similarity_edges_path: Path,
    manager_communities_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> StockNetworkFeatureResult:
    """Create or load the frozen stock-level network features."""
    holdings_path = Path(holdings_path)
    similarity_edges_path = Path(similarity_edges_path)
    manager_communities_path = Path(manager_communities_path)
    output_path = Path(output_path)

    _require_parquet(holdings_path, _HOLDING_COLUMNS, "Holdings file")
    _require_parquet(
        similarity_edges_path,
        _SIMILARITY_COLUMNS,
        "Manager-similarity edge file",
    )
    _require_parquet(
        manager_communities_path,
        _COMMUNITY_COLUMNS,
        "Manager-community file",
    )

    if output_path.exists() and not overwrite:
        input_paths = (
            holdings_path,
            similarity_edges_path,
            manager_communities_path,
            Path(__file__),
        )
        newest_input = max(path.stat().st_mtime_ns for path in input_paths)

        if newest_input > output_path.stat().st_mtime_ns:
            raise RuntimeError(
                "Stock-network-feature inputs or code are newer than the "
                "existing output. Rebuild with overwrite=True."
            )

        row_count, report_periods = _validate_output(output_path)
        return StockNetworkFeatureResult(
            created=False,
            path=output_path,
            row_count=row_count,
            report_periods=report_periods,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".temporary.parquet")
    temporary_path.unlink(missing_ok=True)
    _LOGGER.info("Building the frozen stock-level network features.")

    try:
        _create_features(
            holdings_path,
            similarity_edges_path,
            manager_communities_path,
            temporary_path,
        )
        row_count, report_periods = _validate_output(temporary_path)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return StockNetworkFeatureResult(
        created=True,
        path=output_path,
        row_count=row_count,
        report_periods=report_periods,
    )


__all__ = ["StockNetworkFeatureResult", "build_stock_network_features"]
