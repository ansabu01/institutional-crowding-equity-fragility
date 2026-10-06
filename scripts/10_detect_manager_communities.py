"""Detect quarterly communities in the manager-similarity network."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.network.detect_manager_communities import (  # noqa: E402
    detect_manager_communities,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional community-model rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Detect quarterly manager portfolio communities."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing manager-community outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or load the manager-community outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    ownership_directory = paths.interim / "networks" / "ownership"
    similarity_directory = (
        paths.interim / "networks" / "manager_similarity"
    )
    result = detect_manager_communities(
        manager_nodes_path=ownership_directory / "manager_nodes.parquet",
        edges_path=(
            similarity_directory / "manager_similarity_edges.parquet"
        ),
        output_directory=(
            paths.interim / "networks" / "manager_communities"
        ),
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Manager communities: {status}")
    print(f"{'Table':<40} | Rows")
    print(f"{'-' * 40}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<40} | {table.row_count:>12,}")

    print()
    print("Manager-community outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
