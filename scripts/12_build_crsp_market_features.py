"""Build point-in-time CRSP market features for the 13F stock universe."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.features.market_features import (  # noqa: E402
    build_crsp_market_features,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional market-feature rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build point-in-time CRSP market features.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing CRSP market-feature outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate all CRSP market-feature outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    result = build_crsp_market_features(
        daily_stock_path=paths.raw / "crsp" / "daily_stock.parquet",
        market_index_path=paths.raw / "crsp" / "daily_market_index.parquet",
        security_nodes_path=(
            paths.interim / "networks" / "ownership" / "security_nodes.parquet"
        ),
        output_directory=paths.interim / "crsp",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"CRSP market features: {status}")
    print(f"{'Table':<30} | Rows")
    print(f"{'-' * 30}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<30} | {table.row_count:>12,}")

    print()
    print(f"Security-periods: {result.security_periods:,}")
    print(f"Usable security-periods: {result.usable_security_periods:,}")
    print("CRSP market-feature outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
