"""Build the frozen stock-level ownership-network features."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.network.build_stock_features import (  # noqa: E402
    build_stock_network_features,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional network-feature rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build the frozen stock-level network features."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the existing network-feature output.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or load the frozen stock-level network features."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    network_directory = paths.interim / "networks"
    result = build_stock_network_features(
        holdings_path=(
            paths.interim
            / "sec_13f_crsp"
            / "manager_security_holdings.parquet"
        ),
        similarity_edges_path=(
            network_directory
            / "manager_similarity"
            / "manager_similarity_edges.parquet"
        ),
        manager_communities_path=(
            network_directory
            / "manager_communities"
            / "manager_communities.parquet"
        ),
        output_path=network_directory / "stock_network_features.parquet",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Stock-level network features: {status}")
    print(f"Rows: {result.row_count:,}")
    print(f"Report periods: {result.report_periods:,}")
    print("Stock-level network-feature output: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
