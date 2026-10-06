"""Build point-in-time quarterly inputs for graph neural networks."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.gnn_data import build_gnn_snapshots  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build quarterly inputs for the graph models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the existing GNN inputs.",
    )
    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the quarterly GNN inputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    paths = get_paths_config()
    ownership_directory = paths.interim / "networks" / "ownership"
    modeling_directory = paths.processed / "modeling"
    result = build_gnn_snapshots(
        holdings_path=(
            paths.interim
            / "sec_13f_crsp"
            / "manager_security_holdings.parquet"
        ),
        manager_nodes_path=ownership_directory / "manager_nodes.parquet",
        security_nodes_path=ownership_directory / "security_nodes.parquet",
        panel_path=modeling_directory / "stock_quarter_panel.parquet",
        sample_path=modeling_directory / "modeling_sample.parquet",
        output_directory=modeling_directory / "gnn",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"GNN snapshots: {status}")
    print(f"Quarterly snapshots: {result.snapshots:,}")
    print(f"Stock nodes: {result.stock_nodes:,}")
    print(f"Manager nodes: {result.manager_nodes:,}")
    print(f"Ownership edges: {result.ownership_edges:,}")
    print("GNN snapshot outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
