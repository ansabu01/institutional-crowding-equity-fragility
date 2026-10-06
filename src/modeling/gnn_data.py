"""Build point-in-time quarterly inputs for graph neural networks."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .feature_sets import (
    MANAGER_GNN_FEATURES as _MANAGER_FEATURES,
    STOCK_GNN_FEATURES as _STOCK_FEATURES,
)

_TARGET = "future_max_drawdown_63d"
_OUTPUT_NAMES = (
    "stock_nodes",
    "manager_nodes",
    "ownership_edges",
    "snapshot_audit",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GnnSnapshotBuildResult:
    """Result of creating or loading the quarterly GNN inputs."""

    created: bool
    output_directory: Path
    snapshots: int
    stock_nodes: int
    manager_nodes: int
    ownership_edges: int


def _require_columns(
    path: Path,
    required_columns: set[str],
    description: str,
) -> None:
    """Require a readable Parquet file containing the requested columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing = required_columns - columns
    if missing:
        raise RuntimeError(f"Missing columns in {description}: {sorted(missing)}")


def _output_paths(output_directory: Path) -> dict[str, Path]:
    """Return the four GNN-input paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_NAMES
    }


def _validate_outputs(paths: dict[str, Path]) -> GnnSnapshotBuildResult:
    """Validate schemas, graph references, weights, and split labels."""
    required_columns = {
        "stock_nodes": {
            "report_period",
            "information_date",
            "security_index",
            "permno",
            "target_window_end",
            _TARGET,
            "sample_split",
            *_STOCK_FEATURES,
        },
        "manager_nodes": {
            "report_period",
            "information_date",
            "manager_index",
            "CIK",
            *_MANAGER_FEATURES,
        },
        "ownership_edges": {
            "report_period",
            "manager_index",
            "security_index",
            "portfolio_weight",
            "reported_ownership_share",
        },
        "snapshot_audit": {
            "report_period",
            "information_date",
            "graph_split",
            "stock_nodes",
            "labeled_stocks",
            "manager_nodes",
            "ownership_edges",
        },
    }

    for name, path in paths.items():
        _require_columns(path, required_columns[name], f"GNN {name}")

    connection = duckdb.connect()
    try:
        stock_rows, stock_keys, labeled_rows = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, security_index)),
                COUNT(*) FILTER (WHERE sample_split IS NOT NULL)
            FROM read_parquet(?)
            """,
            [str(paths["stock_nodes"])],
        ).fetchone()
        manager_rows, manager_keys = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, manager_index))
            FROM read_parquet(?)
            """,
            [str(paths["manager_nodes"])],
        ).fetchone()
        edge_rows, edge_keys, invalid_weights = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT (report_period, manager_index, security_index)),
                COUNT(*) FILTER (
                    WHERE portfolio_weight <= 0
                       OR reported_ownership_share <= 0
                       OR reported_ownership_share > 1.000001
                )
            FROM read_parquet(?)
            """,
            [str(paths["ownership_edges"])],
        ).fetchone()
        missing_references = connection.execute(
            """
            SELECT COUNT(*)
            FROM read_parquet(?) e
            LEFT JOIN read_parquet(?) m
              USING (report_period, manager_index)
            LEFT JOIN read_parquet(?) s
              USING (report_period, security_index)
            WHERE m.manager_index IS NULL OR s.security_index IS NULL
            """,
            [
                str(paths["ownership_edges"]),
                str(paths["manager_nodes"]),
                str(paths["stock_nodes"]),
            ],
        ).fetchone()[0]
        audit = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(stock_nodes),
                SUM(labeled_stocks),
                SUM(manager_nodes),
                SUM(ownership_edges),
                COUNT(*) FILTER (
                    WHERE graph_split NOT IN (
                        'train', 'validation', 'test', 'purged'
                    )
                )
            FROM read_parquet(?)
            """,
            [str(paths["snapshot_audit"])],
        ).fetchone()
    finally:
        connection.close()

    if not stock_rows or not manager_rows or not edge_rows or not audit[0]:
        raise RuntimeError("One or more GNN inputs are empty.")
    if stock_rows != stock_keys or manager_rows != manager_keys or edge_rows != edge_keys:
        raise RuntimeError("GNN node or edge keys are not unique.")
    if invalid_weights or missing_references:
        raise RuntimeError("GNN edge weights or node references are invalid.")
    if (audit[1], audit[2], audit[3], audit[4]) != (
        stock_rows,
        labeled_rows,
        manager_rows,
        edge_rows,
    ) or audit[5]:
        raise RuntimeError("GNN snapshot audit does not match the graph tables.")

    return GnnSnapshotBuildResult(
        created=False,
        output_directory=paths["stock_nodes"].parent,
        snapshots=audit[0],
        stock_nodes=stock_rows,
        manager_nodes=manager_rows,
        ownership_edges=edge_rows,
    )


def _create_outputs(
    holdings_path: Path,
    manager_nodes_path: Path,
    security_nodes_path: Path,
    panel_path: Path,
    sample_path: Path,
    paths: dict[str, Path],
) -> None:
    """Create compact node, edge, and audit tables with DuckDB."""
    connection = duckdb.connect()
    try:
        connection.read_parquet(str(holdings_path)).create_view("_holdings")
        connection.read_parquet(str(manager_nodes_path)).create_view("_managers")
        connection.read_parquet(str(security_nodes_path)).create_view("_securities")
        connection.read_parquet(str(panel_path)).create_view("_panel")
        connection.read_parquet(str(sample_path)).create_view("_sample")
        connection.execute(
            """
            CREATE TEMP TABLE _periods AS
            WITH sample_periods AS (
                SELECT DISTINCT report_period, sample_split
                FROM _sample
            ),
            sample_bounds AS (
                SELECT MIN(report_period) AS first_period,
                       MAX(report_period) AS last_period
                FROM _sample
            )
            SELECT DISTINCT
                p.report_period,
                COALESCE(s.sample_split, 'purged') AS graph_split
            FROM _panel p
            LEFT JOIN sample_periods s USING (report_period)
            CROSS JOIN sample_bounds b
            WHERE p.report_period BETWEEN b.first_period AND b.last_period
            """
        )

        _LOGGER.info("Building GNN stock-node table.")
        stock_features = ",\n                    ".join(
            f"p.{column}" for column in _STOCK_FEATURES
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    s.report_period,
                    s.information_date,
                    s.security_index,
                    s.permno,
                    {stock_features},
                    p.target_window_end,
                    p.{_TARGET},
                    l.sample_split
                FROM _securities s
                JOIN _periods q USING (report_period)
                JOIN _panel p USING (report_period, permno)
                LEFT JOIN _sample l USING (report_period, permno)
                ORDER BY s.report_period, s.security_index
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(paths["stock_nodes"])],
        )

        _LOGGER.info("Building GNN manager-node table.")
        manager_features = ",\n                    ".join(
            f"m.{column}" for column in _MANAGER_FEATURES
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    m.report_period,
                    m.information_date,
                    m.manager_index,
                    m.CIK,
                    {manager_features}
                FROM _managers m
                JOIN _periods q USING (report_period)
                ORDER BY m.report_period, m.manager_index
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(paths["manager_nodes"])],
        )

        _LOGGER.info("Building GNN ownership-edge table.")
        connection.execute(
            """
            COPY (
                SELECT
                    h.report_period,
                    m.manager_index,
                    s.security_index,
                    CAST(h.portfolio_weight AS FLOAT) AS portfolio_weight,
                    CAST(
                        h.position_value_usd / s.total_reported_value_usd
                        AS FLOAT
                    ) AS reported_ownership_share
                FROM _holdings h
                JOIN _periods q USING (report_period)
                JOIN _managers m USING (report_period, CIK)
                JOIN _securities s USING (report_period, permno)
                ORDER BY h.report_period, m.manager_index, s.security_index
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 500000
            )
            """,
            [str(paths["ownership_edges"])],
        )

        _LOGGER.info("Building GNN snapshot audit.")
        connection.read_parquet(str(paths["stock_nodes"])).create_view("_gnn_stocks")
        connection.read_parquet(str(paths["manager_nodes"])).create_view("_gnn_managers")
        connection.read_parquet(str(paths["ownership_edges"])).create_view("_gnn_edges")
        connection.execute(
            """
            COPY (
                SELECT
                    q.report_period,
                    MIN(s.information_date) AS information_date,
                    q.graph_split,
                    COUNT(DISTINCT s.security_index) AS stock_nodes,
                    COUNT(DISTINCT s.security_index) FILTER (
                        WHERE s.sample_split IS NOT NULL
                    ) AS labeled_stocks,
                    (SELECT COUNT(*) FROM _gnn_managers m WHERE m.report_period = q.report_period)
                        AS manager_nodes,
                    (SELECT COUNT(*) FROM _gnn_edges e WHERE e.report_period = q.report_period)
                        AS ownership_edges
                FROM _periods q
                JOIN _gnn_stocks s USING (report_period)
                GROUP BY q.report_period, q.graph_split
                ORDER BY q.report_period
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(paths["snapshot_audit"])],
        )
    finally:
        connection.close()


def build_gnn_snapshots(
    holdings_path: Path,
    manager_nodes_path: Path,
    security_nodes_path: Path,
    panel_path: Path,
    sample_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> GnnSnapshotBuildResult:
    """Build or validate the fixed quarterly GNN inputs."""
    input_paths = tuple(
        map(
            Path,
            (
                holdings_path,
                manager_nodes_path,
                security_nodes_path,
                panel_path,
                sample_path,
            ),
        )
    )
    output_directory = Path(output_directory)
    output_paths = _output_paths(output_directory)

    _require_columns(
        input_paths[0],
        {"report_period", "CIK", "permno", "position_value_usd", "portfolio_weight"},
        "mapped holdings",
    )
    _require_columns(
        input_paths[1],
        {"report_period", "information_date", "manager_index", "CIK", *_MANAGER_FEATURES},
        "manager nodes",
    )
    _require_columns(
        input_paths[2],
        {"report_period", "information_date", "security_index", "permno", "total_reported_value_usd"},
        "security nodes",
    )
    _require_columns(
        input_paths[3],
        {"report_period", "permno", "target_window_end", _TARGET, *_STOCK_FEATURES},
        "stock-quarter panel",
    )
    _require_columns(
        input_paths[4],
        {"report_period", "permno", "sample_split"},
        "modeling sample",
    )

    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError("GNN snapshot outputs are incomplete. Rebuild with --overwrite.")
        freshness_paths = input_paths + (
            Path(__file__),
            Path(__file__).with_name("feature_sets.py"),
        )
        if max(path.stat().st_mtime_ns for path in freshness_paths) > min(
            path.stat().st_mtime_ns for path in output_paths.values()
        ):
            raise RuntimeError(
                "A GNN snapshot input or code is newer than the outputs. "
                "Rebuild with --overwrite."
            )
        return _validate_outputs(output_paths)

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_name(f".{path.name}.part")
        for name, path in output_paths.items()
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    try:
        _create_outputs(*input_paths, temporary_paths)
        result = _validate_outputs(temporary_paths)
        for name in _OUTPUT_NAMES:
            temporary_paths[name].replace(output_paths[name])
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    return GnnSnapshotBuildResult(
        created=True,
        output_directory=output_directory,
        snapshots=result.snapshots,
        stock_nodes=result.stock_nodes,
        manager_nodes=result.manager_nodes,
        ownership_edges=result.ownership_edges,
    )


__all__ = ["GnnSnapshotBuildResult", "build_gnn_snapshots"]
