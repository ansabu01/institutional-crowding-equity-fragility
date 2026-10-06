"""Build quarterly bipartite manager-security ownership networks."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.csgraph import connected_components


_HOLDING_COLUMNS = {
    "CIK",
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
    "position_value_usd",
    "portfolio_weight",
}
_TOTAL_COLUMNS = {
    "CIK",
    "PERIODOFREPORT",
    "report_period",
    "information_date",
    "FILINGMANAGER_NAME",
    "portfolio_value_usd",
    "security_count",
    "underlying_position_rows",
}
_OUTPUT_COLUMNS = {
    "manager_nodes": (
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
    ),
    "security_nodes": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "security_index",
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
    ),
    "network_summary": (
        "PERIODOFREPORT",
        "report_period",
        "information_date",
        "managers",
        "securities",
        "ownership_edges",
        "bipartite_density",
        "connected_components",
        "largest_component_nodes",
        "largest_component_share",
    ),
}
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkTableResult:
    """Metadata for one ownership-network output."""

    name: str
    path: Path
    row_count: int


@dataclass(frozen=True)
class OwnershipNetworkResult:
    """Result of creating or loading the ownership-network outputs."""

    created: bool
    tables: tuple[NetworkTableResult, ...]
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
    """Return the three ownership-network output paths."""
    return {
        name: output_directory / f"{name}.parquet"
        for name in _OUTPUT_COLUMNS
    }


def _table_results(
    paths: dict[str, Path],
) -> tuple[NetworkTableResult, ...]:
    """Return basic metadata for the output tables."""
    return tuple(
        NetworkTableResult(
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
) -> OwnershipNetworkResult | None:
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
            "Ownership-network outputs are incomplete. Missing files:\n  - "
            + "\n  - ".join(missing)
        )

    newest_input = max(path.stat().st_mtime_ns for path in input_paths)
    oldest_output = min(
        path.stat().st_mtime_ns for path in output_paths.values()
    )

    if newest_input > oldest_output:
        raise RuntimeError(
            "Ownership-network inputs or code are newer than the existing outputs. "
            "Run script 8 with --overwrite."
        )

    tables = _table_results(output_paths)
    report_periods = next(
        table.row_count
        for table in tables
        if table.name == "network_summary"
    )

    return OwnershipNetworkResult(
        created=False,
        tables=tables,
        report_periods=report_periods,
    )


def _load_period(
    connection: duckdb.DuckDBPyConnection,
    holdings_path: Path,
    totals_path: Path,
    report_period: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one quarterly holdings snapshot."""
    holdings = connection.execute(
        "SELECT * FROM read_parquet(?) WHERE report_period = ?",
        [str(holdings_path), report_period],
    ).fetch_df()
    totals = connection.execute(
        "SELECT * FROM read_parquet(?) WHERE report_period = ?",
        [str(totals_path), report_period],
    ).fetch_df()

    if holdings.empty or totals.empty:
        raise RuntimeError(f"Empty ownership snapshot for {report_period.date()}.")

    return holdings, totals


def _build_manager_nodes(
    holdings: pd.DataFrame,
    totals: pd.DataFrame,
) -> pd.DataFrame:
    """Build manager nodes and portfolio concentration measures."""
    manager_statistics = (
        holdings.assign(
            squared_portfolio_weight=holdings["portfolio_weight"] ** 2
        )
        .groupby("CIK", as_index=False)
        .agg(
            largest_position_weight=("portfolio_weight", "max"),
            portfolio_hhi=("squared_portfolio_weight", "sum"),
        )
    )
    manager_nodes = (
        totals.merge(manager_statistics, on="CIK", how="inner")
        .sort_values("CIK")
        .reset_index(drop=True)
    )
    manager_nodes.insert(
        3,
        "manager_index",
        np.arange(len(manager_nodes), dtype=np.int32),
    )

    return manager_nodes[list(_OUTPUT_COLUMNS["manager_nodes"])]


def _build_security_nodes(holdings: pd.DataFrame) -> pd.DataFrame:
    """Build security nodes and reported-ownership concentration measures."""
    rows = holdings.copy()
    rows["reported_holding_share"] = (
        rows["position_value_usd"]
        / rows.groupby("permno")["position_value_usd"].transform("sum")
    )
    rows["reported_holding_share_squared"] = (
        rows["reported_holding_share"] ** 2
    )
    rows = rows.sort_values(
        ["permno", "reported_holding_share", "CIK"],
        ascending=[True, False, True],
    )
    rows["holder_rank"] = rows.groupby("permno").cumcount() + 1

    security_nodes = (
        rows.groupby("permno", as_index=False)
        .agg(
            PERIODOFREPORT=("PERIODOFREPORT", "first"),
            report_period=("report_period", "first"),
            information_date=("information_date", "first"),
            permco=("permco", "first"),
            CUSIP=("CUSIP", "first"),
            ticker=("ticker", "first"),
            securitynm=("securitynm", "first"),
            primaryexch=("primaryexch", "first"),
            issuertype=("issuertype", "first"),
            institutional_holder_count=("CIK", "nunique"),
            total_reported_value_usd=("position_value_usd", "sum"),
            reported_ownership_hhi=(
                "reported_holding_share_squared",
                "sum",
            ),
            mean_owner_portfolio_weight=("portfolio_weight", "mean"),
            maximum_owner_portfolio_weight=("portfolio_weight", "max"),
        )
        .sort_values("permno")
        .reset_index(drop=True)
    )
    top_five_share = (
        rows.loc[rows["holder_rank"].le(5)]
        .groupby("permno")["reported_holding_share"]
        .sum()
        .rename("top_five_reported_ownership_share")
    )
    security_nodes = security_nodes.merge(
        top_five_share,
        on="permno",
        how="left",
    )
    security_nodes.insert(
        3,
        "security_index",
        np.arange(len(security_nodes), dtype=np.int32),
    )

    return security_nodes[list(_OUTPUT_COLUMNS["security_nodes"])]


def _build_ownership_matrix(
    holdings: pd.DataFrame,
    manager_nodes: pd.DataFrame,
    security_nodes: pd.DataFrame,
) -> csr_matrix:
    """Return the manager-by-security portfolio-weight matrix."""
    manager_codes = pd.Index(manager_nodes["CIK"]).get_indexer(holdings["CIK"])
    security_codes = pd.Index(security_nodes["permno"]).get_indexer(
        holdings["permno"]
    )
    matrix = csr_matrix(
        (
            holdings["portfolio_weight"].to_numpy(dtype=np.float64),
            (manager_codes, security_codes),
        ),
        shape=(len(manager_nodes), len(security_nodes)),
    )
    matrix.sum_duplicates()

    return matrix


def _build_network_summary(
    matrix: csr_matrix,
    manager_nodes: pd.DataFrame,
    security_nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Build one summary row for the quarterly bipartite graph."""
    binary_matrix = matrix.copy()
    binary_matrix.data = np.ones_like(binary_matrix.data, dtype=np.int8)
    adjacency = bmat(
        [[None, binary_matrix], [binary_matrix.T, None]],
        format="csr",
    )
    component_count, component_labels = connected_components(
        adjacency,
        directed=False,
    )
    largest_component_nodes = int(np.bincount(component_labels).max())
    record = {
        "PERIODOFREPORT": manager_nodes["PERIODOFREPORT"].iloc[0],
        "report_period": manager_nodes["report_period"].iloc[0],
        "information_date": manager_nodes["information_date"].iloc[0],
        "managers": len(manager_nodes),
        "securities": len(security_nodes),
        "ownership_edges": matrix.nnz,
        "bipartite_density": (
            matrix.nnz / (len(manager_nodes) * len(security_nodes))
        ),
        "connected_components": component_count,
        "largest_component_nodes": largest_component_nodes,
        "largest_component_share": (
            largest_component_nodes / adjacency.shape[0]
        ),
    }

    return pd.DataFrame.from_records(
        [record],
        columns=_OUTPUT_COLUMNS["network_summary"],
    )


def _write_period_parts(
    tables: dict[str, pd.DataFrame],
    temporary_paths: dict[str, Path],
    index: int,
) -> dict[str, Path]:
    """Write one quarterly part for each output."""
    part_paths = {}

    for name, table in tables.items():
        part_path = temporary_paths[name].with_name(
            f"{temporary_paths[name].name}.{index:03d}.quarter.parquet"
        )
        table.to_parquet(
            part_path,
            index=False,
            compression="zstd",
        )
        part_paths[name] = part_path

    return part_paths


def _consolidate_parts(
    connection: duckdb.DuckDBPyConnection,
    part_paths: dict[str, list[Path]],
    temporary_paths: dict[str, Path],
) -> None:
    """Combine quarterly parts into the three final temporary files."""
    for index, name in enumerate(_OUTPUT_COLUMNS):
        view_name = f"_network_parts_{index}"
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
    totals_path: Path,
    temporary_paths: dict[str, Path],
) -> None:
    """Build each quarterly graph and consolidate its outputs."""
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
                [str(holdings_path)],
            ).fetchall()
        ]

        for index, report_period in enumerate(report_periods, start=1):
            _LOGGER.info(
                "[%02d/%02d] Building ownership network for %s.",
                index,
                len(report_periods),
                report_period.date(),
            )
            holdings, totals = _load_period(
                connection,
                holdings_path,
                totals_path,
                report_period,
            )
            manager_nodes = _build_manager_nodes(holdings, totals)
            security_nodes = _build_security_nodes(holdings)
            matrix = _build_ownership_matrix(
                holdings,
                manager_nodes,
                security_nodes,
            )
            network_summary = _build_network_summary(
                matrix,
                manager_nodes,
                security_nodes,
            )
            quarter_parts = _write_period_parts(
                {
                    "manager_nodes": manager_nodes,
                    "security_nodes": security_nodes,
                    "network_summary": network_summary,
                },
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


def build_ownership_networks(
    holdings_path: Path,
    totals_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> OwnershipNetworkResult:
    """Create or load the quarterly bipartite ownership networks."""
    holdings_path = Path(holdings_path)
    totals_path = Path(totals_path)
    output_directory = Path(output_directory)
    input_paths = (holdings_path, totals_path, Path(__file__))
    output_paths = _output_paths(output_directory)

    _require_parquet(
        holdings_path,
        _HOLDING_COLUMNS,
        "Mapped manager-security holdings",
    )
    _require_parquet(
        totals_path,
        _TOTAL_COLUMNS,
        "Mapped manager portfolio totals",
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
            totals_path,
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

    report_periods = next(
        table.row_count
        for table in tables
        if table.name == "network_summary"
    )

    return OwnershipNetworkResult(
        created=True,
        tables=tables,
        report_periods=report_periods,
    )


__all__ = [
    "NetworkTableResult",
    "OwnershipNetworkResult",
    "build_ownership_networks",
]
