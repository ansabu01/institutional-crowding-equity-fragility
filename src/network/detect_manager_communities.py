"""Detect quarterly communities in the manager-similarity network."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import networkx as nx
import pandas as pd
import pyarrow.parquet as pq


_LOUVAIN_RESOLUTION = 1.0
_RANDOM_SEED = 13
_MANAGER_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "manager_index",
    "CIK",
    "FILINGMANAGER_NAME",
    "portfolio_value_usd",
    "security_count",
    "underlying_position_rows",
    "largest_position_weight",
    "portfolio_hhi",
}
_EDGE_COLUMNS = {
    "report_period",
    "manager_a_index",
    "manager_b_index",
    "cosine_similarity",
}
_OUTPUT_COLUMNS = {
    "manager_communities": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "manager_index",
        "CIK",
        "FILINGMANAGER_NAME",
        "portfolio_value_usd",
        "security_count",
        "underlying_position_rows",
        "largest_position_weight",
        "portfolio_hhi",
        "similarity_degree",
        "mean_neighbor_similarity",
        "community_id",
    ),
    "manager_community_summary": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "community_id",
        "manager_count",
        "manager_share",
        "total_reported_portfolio_value_usd",
        "median_reported_portfolio_value_usd",
        "mean_security_count",
        "mean_largest_position_weight",
        "mean_portfolio_hhi",
        "mean_similarity_degree",
    ),
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommunityTableResult:
    """Metadata for one manager-community output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class ManagerCommunityResult:
    """Result of creating or loading the manager-community outputs."""

    created: bool
    tables: tuple[CommunityTableResult, ...]


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
    """Return the manager-community output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _table_results(
    paths: dict[str, Path],
) -> tuple[CommunityTableResult, ...]:
    """Return basic metadata for the output tables."""
    return tuple(
        CommunityTableResult(
            name=name,
            path=paths[name],
            row_count=_require_parquet(
                paths[name],
                set(_OUTPUT_COLUMNS[name]),
                f"{name} output",
            ),
        )
        for name in _OUTPUT_COLUMNS
    )


def _existing_result(
    output_paths: dict[str, Path],
    input_paths: tuple[Path, ...],
) -> ManagerCommunityResult | None:
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
            "Manager-community outputs are incomplete. Missing files:\n  - "
            + "\n  - ".join(missing)
        )

    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    oldest_output = min(
        path.stat().st_mtime_ns for path in output_paths.values()
    )

    if newest_input > oldest_output:
        raise RuntimeError(
            "Manager-community inputs or code are newer than the existing outputs. "
            "Run script 10 with --overwrite."
        )

    return ManagerCommunityResult(
        created=False,
        tables=_table_results(output_paths),
    )


def _load_period(
    connection: duckdb.DuckDBPyConnection,
    manager_nodes_path: Path,
    edges_path: Path,
    report_period: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one quarterly manager graph."""
    managers = connection.execute(
        """
        SELECT *
        FROM read_parquet(?)
        WHERE report_period = ?
        ORDER BY manager_index
        """,
        [str(manager_nodes_path), report_period],
    ).fetch_df()
    edges = connection.execute(
        """
        SELECT manager_a_index, manager_b_index, cosine_similarity
        FROM read_parquet(?)
        WHERE report_period = ?
        """,
        [str(edges_path), report_period],
    ).fetch_df()

    if managers.empty or edges.empty:
        raise RuntimeError(
            f"Empty manager-community snapshot for {report_period.date()}."
        )

    return managers, edges


def _build_period_outputs(
    managers: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build community assignments and summaries for one quarter."""
    graph = nx.Graph()
    graph.add_nodes_from(managers["manager_index"])
    graph.add_weighted_edges_from(
        edges.itertuples(index=False, name=None),
        weight="weight",
    )
    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=_LOUVAIN_RESOLUTION,
        seed=_RANDOM_SEED,
    )
    communities = sorted(
        communities,
        key=lambda community: (-len(community), min(community)),
    )
    assignment = pd.DataFrame(
        [
            (manager_index, community_id)
            for community_id, community in enumerate(communities, start=1)
            for manager_index in community
        ],
        columns=["manager_index", "community_id"],
    )
    endpoints = pd.concat(
        [
            edges[["manager_a_index", "cosine_similarity"]].rename(
                columns={"manager_a_index": "manager_index"}
            ),
            edges[["manager_b_index", "cosine_similarity"]].rename(
                columns={"manager_b_index": "manager_index"}
            ),
        ],
        ignore_index=True,
    )
    node_statistics = (
        endpoints.groupby("manager_index", as_index=False)
        .agg(
            similarity_degree=("cosine_similarity", "size"),
            mean_neighbor_similarity=("cosine_similarity", "mean"),
        )
    )
    manager_communities = (
        managers.merge(node_statistics, on="manager_index", how="left")
        .merge(assignment, on="manager_index", how="left")
    )
    manager_communities["similarity_degree"] = (
        manager_communities["similarity_degree"].fillna(0).astype("int64")
    )
    manager_communities["mean_neighbor_similarity"] = manager_communities[
        "mean_neighbor_similarity"
    ].fillna(0.0)

    community_summary = (
        manager_communities.groupby("community_id", as_index=False)
        .agg(
            manager_count=("manager_index", "size"),
            total_reported_portfolio_value_usd=(
                "portfolio_value_usd",
                "sum",
            ),
            median_reported_portfolio_value_usd=(
                "portfolio_value_usd",
                "median",
            ),
            mean_security_count=("security_count", "mean"),
            mean_largest_position_weight=(
                "largest_position_weight",
                "mean",
            ),
            mean_portfolio_hhi=("portfolio_hhi", "mean"),
            mean_similarity_degree=("similarity_degree", "mean"),
        )
        .sort_values("community_id")
        .reset_index(drop=True)
    )
    community_summary["manager_share"] = (
        community_summary["manager_count"] / len(manager_communities)
    )

    for position, column in enumerate(
        ("PERIODOFREPORT", "report_period", "information_date")
    ):
        community_summary.insert(
            position,
            column,
            managers[column].iloc[0],
        )

    return {
        "manager_communities": manager_communities[
            list(_OUTPUT_COLUMNS["manager_communities"])
        ],
        "manager_community_summary": community_summary[
            list(_OUTPUT_COLUMNS["manager_community_summary"])
        ],
    }


def _write_parts(
    tables: dict[str, pd.DataFrame],
    temporary_paths: dict[str, Path],
    index: int,
) -> dict[str, Path]:
    """Write one quarterly part for each output."""
    part_paths = {}

    for name, table in tables.items():
        path = temporary_paths[name].with_name(
            f"{temporary_paths[name].name}.{index:03d}.quarter.parquet"
        )
        table.to_parquet(path, index=False, compression="zstd")
        part_paths[name] = path

    return part_paths


def _consolidate_parts(
    connection: duckdb.DuckDBPyConnection,
    part_paths: dict[str, list[Path]],
    temporary_paths: dict[str, Path],
) -> None:
    """Combine quarterly parts into final temporary files."""
    for index, name in enumerate(_OUTPUT_COLUMNS):
        view_name = f"_community_parts_{index}"
        connection.read_parquet(
            [str(path) for path in part_paths[name]]
        ).create_view(view_name)
        connection.execute(
            f"""
            COPY (
                SELECT * FROM {view_name}
            ) TO ? (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """,
            [str(temporary_paths[name])],
        )
        connection.execute(f"DROP VIEW {view_name}")


def _create_outputs(
    manager_nodes_path: Path,
    edges_path: Path,
    temporary_paths: dict[str, Path],
) -> None:
    """Build and consolidate every quarterly community model."""
    connection = duckdb.connect()
    part_paths: dict[str, list[Path]] = {
        name: [] for name in _OUTPUT_COLUMNS
    }

    try:
        report_periods = [
            pd.Timestamp(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT report_period
                FROM read_parquet(?)
                ORDER BY report_period
                """,
                [str(manager_nodes_path)],
            ).fetchall()
        ]

        for index, report_period in enumerate(report_periods, start=1):
            _LOGGER.info(
                "[%02d/%02d] Detecting manager communities for %s.",
                index,
                len(report_periods),
                report_period.date(),
            )
            managers, edges = _load_period(
                connection,
                manager_nodes_path,
                edges_path,
                report_period,
            )
            quarter_parts = _write_parts(
                _build_period_outputs(managers, edges),
                temporary_paths,
                index,
            )

            for name, path in quarter_parts.items():
                part_paths[name].append(path)

        _consolidate_parts(connection, part_paths, temporary_paths)
    finally:
        connection.close()

        for paths in part_paths.values():
            for path in paths:
                path.unlink(missing_ok=True)


def detect_manager_communities(
    manager_nodes_path: Path,
    edges_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> ManagerCommunityResult:
    """Create or load all quarterly manager-community outputs."""
    manager_nodes_path = Path(manager_nodes_path)
    edges_path = Path(edges_path)
    output_directory = Path(output_directory)
    input_paths = (manager_nodes_path, edges_path, Path(__file__))
    output_paths = _output_paths(output_directory)

    _require_parquet(
        manager_nodes_path,
        _MANAGER_COLUMNS,
        "Ownership-network manager nodes",
    )
    _require_parquet(
        edges_path,
        _EDGE_COLUMNS,
        "Manager-similarity edges",
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

    try:
        _create_outputs(
            manager_nodes_path,
            edges_path,
            temporary_paths,
        )
        tables = _table_results(temporary_paths)

        for name in _OUTPUT_COLUMNS:
            temporary_paths[name].replace(output_paths[name])
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

        for path in output_directory.glob("*.quarter.parquet"):
            path.unlink(missing_ok=True)

    return ManagerCommunityResult(
        created=True,
        tables=tables,
    )


__all__ = [
    "CommunityTableResult",
    "ManagerCommunityResult",
    "detect_manager_communities",
]
