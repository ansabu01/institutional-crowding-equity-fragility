"""Build quarterly manager portfolio-similarity networks."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.network.build_manager_similarity import (  # noqa: E402
    build_manager_similarity_network,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional similarity-network rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build quarterly manager portfolio-similarity networks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing manager-similarity outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or load the manager-similarity outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    result = build_manager_similarity_network(
        holdings_path=(
            paths.interim
            / "sec_13f_crsp"
            / "manager_security_holdings.parquet"
        ),
        manager_nodes_path=(
            paths.interim
            / "networks"
            / "ownership"
            / "manager_nodes.parquet"
        ),
        output_directory=(
            paths.interim / "networks" / "manager_similarity"
        ),
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Manager similarity: {status}")
    print(f"{'Table':<30} | Rows")
    print(f"{'-' * 30}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<30} | {table.row_count:>12,}")

    print()
    print(f"Report periods: {result.report_periods:,}")
    print("Manager-similarity outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
