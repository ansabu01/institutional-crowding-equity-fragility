"""Build quarterly manager portfolio-similarity networks."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


_NEIGHBOR_COUNTS = (10, 25, 50)
_BASELINE_NEIGHBORS = 25
_HOLDING_COLUMNS = {
    "CIK",
    "report_period",
    "permno",
    "portfolio_weight",
}
_MANAGER_COLUMNS = {
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "manager_index",
    "CIK",
}
_OUTPUT_COLUMNS = {
    "manager_similarity_edges": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "manager_a_index",
        "manager_b_index",
        "manager_a_cik",
        "manager_b_cik",
        "cosine_similarity",
        "directional_links",
        "best_neighbor_rank",
        "is_mutual_neighbor",
        "k_neighbors",
    ),
    "similarity_graph_audit": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "neighbors_per_manager",
        "undirected_edges",
        "mutual_edges",
        "mutual_edge_share",
        "connected_managers",
        "isolated_managers",
        "mean_similarity",
        "median_similarity",
        "projected_density",
    ),
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimilarityTableResult:
    """Metadata for one manager-similarity output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class ManagerSimilarityResult:
    """Result of creating or loading the similarity-network outputs."""

    created: bool
    tables: tuple[SimilarityTableResult, ...]
    report_periods: int


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
    """Return the manager-similarity output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _table_results(
    paths: dict[str, Path],
) -> tuple[SimilarityTableResult, ...]:
    """Return basic metadata for the output tables."""
    return tuple(
        SimilarityTableResult(
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
) -> ManagerSimilarityResult | None:
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
            "Manager-similarity outputs are incomplete. Missing files:\n  - "
            + "\n  - ".join(missing)
        )

    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    oldest_output = min(
        path.stat().st_mtime_ns for path in output_paths.values()
    )

    if newest_input > oldest_output:
        raise RuntimeError(
            "Manager-similarity inputs or code are newer than the existing outputs. "
            "Run script 9 with --overwrite."
        )

    tables = _table_results(output_paths)
    audit_rows = next(
        table.row_count
        for table in tables
        if table.name == "similarity_graph_audit"
    )

    return ManagerSimilarityResult(
        created=False,
        tables=tables,
        report_periods=audit_rows // len(_NEIGHBOR_COUNTS),
    )


def _load_period(
    connection: duckdb.DuckDBPyConnection,
    holdings_path: Path,
    manager_nodes_path: Path,
    report_period: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one quarterly set of holdings and manager nodes."""
    holdings = connection.execute(
        """
        SELECT CIK, permno, portfolio_weight
        FROM read_parquet(?)
        WHERE report_period = ?
        """,
        [str(holdings_path), report_period],
    ).fetch_df()
    managers = connection.execute(
        """
        SELECT *
        FROM read_parquet(?)
        WHERE report_period = ?
        ORDER BY manager_index
        """,
        [str(manager_nodes_path), report_period],
    ).fetch_df()

    if holdings.empty or managers.empty:
        raise RuntimeError(
            f"Empty manager-similarity snapshot for {report_period.date()}."
        )

    return holdings, managers


def _build_directed_neighbors(
    holdings: pd.DataFrame,
    managers: pd.DataFrame,
) -> pd.DataFrame:
    """Return each manager's 50 nearest portfolio neighbours."""
    maximum_neighbors = max(_NEIGHBOR_COUNTS)

    if len(managers) <= maximum_neighbors:
        raise RuntimeError(
            f"At least {maximum_neighbors + 1} managers are required "
            "for the similarity analysis."
        )

    manager_codes = pd.Index(managers["CIK"]).get_indexer(holdings["CIK"])
    security_codes, security_ids = pd.factorize(holdings["permno"], sort=True)
    matrix = csr_matrix(
        (
            holdings["portfolio_weight"].to_numpy(dtype=np.float64),
            (manager_codes, security_codes),
        ),
        shape=(len(managers), len(security_ids)),
    )
    matrix.sum_duplicates()

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    ).fit(matrix)
    distances, neighbor_rows = model.kneighbors(
        n_neighbors=maximum_neighbors,
        return_distance=True,
    )
    manager_indices = managers["manager_index"].to_numpy()

    return pd.DataFrame(
        {
            "source_index": np.repeat(manager_indices, maximum_neighbors),
            "target_index": manager_indices[neighbor_rows.ravel()],
            "neighbor_rank": np.tile(
                np.arange(1, maximum_neighbors + 1, dtype=np.int16),
                len(managers),
            ),
            "cosine_similarity": np.clip(
                1.0 - distances.ravel(),
                0.0,
                1.0,
            ),
        }
    )


def _undirected_edges(
    directed: pd.DataFrame,
    neighbor_count: int,
) -> pd.DataFrame:
    """Convert directed nearest neighbours to an undirected union graph."""
    selected = directed.loc[
        directed["neighbor_rank"].le(neighbor_count)
        & directed["cosine_similarity"].gt(0)
    ].copy()
    selected["manager_a_index"] = np.minimum(
        selected["source_index"],
        selected["target_index"],
    )
    selected["manager_b_index"] = np.maximum(
        selected["source_index"],
        selected["target_index"],
    )

    edges = (
        selected.groupby(
            ["manager_a_index", "manager_b_index"],
            as_index=False,
        )
        .agg(
            cosine_similarity=("cosine_similarity", "max"),
            directional_links=("neighbor_rank", "size"),
            best_neighbor_rank=("neighbor_rank", "min"),
        )
        .sort_values(["manager_a_index", "manager_b_index"])
        .reset_index(drop=True)
    )
    edges["is_mutual_neighbor"] = edges["directional_links"].eq(2)

    return edges


def _build_period_outputs(
    holdings: pd.DataFrame,
    managers: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build baseline edges and neighbour-count sensitivity for one quarter."""
    directed = _build_directed_neighbors(holdings, managers)
    manager_ids = managers.set_index("manager_index")["CIK"]
    audit_records = []
    baseline_edges = None

    for neighbor_count in _NEIGHBOR_COUNTS:
        edges = _undirected_edges(directed, neighbor_count)
        connected = pd.unique(
            pd.concat(
                [edges["manager_a_index"], edges["manager_b_index"]],
                ignore_index=True,
            )
        )
        audit_records.append(
            {
                "PERIODOFREPORT": managers["PERIODOFREPORT"].iloc[0],
                "report_period": managers["report_period"].iloc[0],
                "information_date": managers["information_date"].iloc[0],
                "neighbors_per_manager": neighbor_count,
                "undirected_edges": len(edges),
                "mutual_edges": int(edges["is_mutual_neighbor"].sum()),
                "mutual_edge_share": float(
                    edges["is_mutual_neighbor"].mean()
                ),
                "connected_managers": len(connected),
                "isolated_managers": len(managers) - len(connected),
                "mean_similarity": float(edges["cosine_similarity"].mean()),
                "median_similarity": float(
                    edges["cosine_similarity"].median()
                ),
                "projected_density": (
                    2 * len(edges) / (len(managers) * (len(managers) - 1))
                ),
            }
        )

        if neighbor_count == _BASELINE_NEIGHBORS:
            baseline_edges = edges

    if baseline_edges is None:
        raise RuntimeError("The baseline similarity graph was not created.")

    baseline_edges.insert(
        0,
        "information_date",
        managers["information_date"].iloc[0],
    )
    baseline_edges.insert(
        0,
        "report_period",
        managers["report_period"].iloc[0],
    )
    baseline_edges.insert(
        0,
        "PERIODOFREPORT",
        managers["PERIODOFREPORT"].iloc[0],
    )
    baseline_edges["manager_a_cik"] = baseline_edges[
        "manager_a_index"
    ].map(manager_ids)
    baseline_edges["manager_b_cik"] = baseline_edges[
        "manager_b_index"
    ].map(manager_ids)
    baseline_edges["k_neighbors"] = _BASELINE_NEIGHBORS

    return {
        "manager_similarity_edges": baseline_edges[
            list(_OUTPUT_COLUMNS["manager_similarity_edges"])
        ],
        "similarity_graph_audit": pd.DataFrame.from_records(
            audit_records,
            columns=_OUTPUT_COLUMNS["similarity_graph_audit"],
        ),
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
        view_name = f"_similarity_parts_{index}"
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
    holdings_path: Path,
    manager_nodes_path: Path,
    temporary_paths: dict[str, Path],
) -> None:
    """Build and consolidate every quarterly similarity graph."""
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
                "[%02d/%02d] Building manager similarity for %s.",
                index,
                len(report_periods),
                report_period.date(),
            )
            holdings, managers = _load_period(
                connection,
                holdings_path,
                manager_nodes_path,
                report_period,
            )
            quarter_parts = _write_parts(
                _build_period_outputs(holdings, managers),
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


def build_manager_similarity_network(
    holdings_path: Path,
    manager_nodes_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> ManagerSimilarityResult:
    """Create or load all quarterly manager-similarity graphs."""
    holdings_path = Path(holdings_path)
    manager_nodes_path = Path(manager_nodes_path)
    output_directory = Path(output_directory)
    input_paths = (holdings_path, manager_nodes_path, Path(__file__))
    output_paths = _output_paths(output_directory)

    _require_parquet(
        holdings_path,
        _HOLDING_COLUMNS,
        "Manager-security holdings",
    )
    _require_parquet(
        manager_nodes_path,
        _MANAGER_COLUMNS,
        "Ownership-network manager nodes",
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
            holdings_path,
            manager_nodes_path,
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

    audit_rows = next(
        table.row_count
        for table in tables
        if table.name == "similarity_graph_audit"
    )

    return ManagerSimilarityResult(
        created=True,
        tables=tables,
        report_periods=audit_rows // len(_NEIGHBOR_COUNTS),
    )


__all__ = [
    "ManagerSimilarityResult",
    "SimilarityTableResult",
    "build_manager_similarity_network",
]
