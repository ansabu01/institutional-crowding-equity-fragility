"""Build quarterly SEC Form 13F bipartite ownership networks."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.network.build_ownership_network import (  # noqa: E402
    build_ownership_networks,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional network-rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build all quarterly SEC Form 13F ownership networks."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing ownership-network outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or load all quarterly ownership-network outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    holdings_directory = paths.interim / "sec_13f_crsp"
    result = build_ownership_networks(
        holdings_path=holdings_directory / "manager_security_holdings.parquet",
        totals_path=holdings_directory / "manager_portfolio_totals.parquet",
        output_directory=paths.interim / "networks" / "ownership",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Ownership networks: {status}")
    print(f"{'Table':<30} | Rows")
    print(f"{'-' * 30}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<30} | {table.row_count:>12,}")

    print()
    print(f"Report periods: {result.report_periods:,}")
    print("Ownership-network outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
