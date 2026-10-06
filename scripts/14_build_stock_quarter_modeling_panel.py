"""Build the final point-in-time stock-quarter modeling panel."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.panel.modeling import (  # noqa: E402
    build_stock_quarter_modeling_panel,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional modeling-panel rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build the point-in-time stock-quarter modeling panel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing modeling-panel outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the final stock-quarter modeling panel."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    ownership_directory = paths.interim / "networks" / "ownership"
    crsp_directory = paths.interim / "crsp"
    result = build_stock_quarter_modeling_panel(
        security_nodes_path=ownership_directory / "security_nodes.parquet",
        network_summary_path=ownership_directory / "network_summary.parquet",
        market_features_path=crsp_directory / "market_features.parquet",
        downside_targets_path=crsp_directory / "downside_targets.parquet",
        output_directory=paths.processed / "modeling",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Stock-quarter modeling panel: {status}")
    print(f"{'Table':<32} | Rows")
    print(f"{'-' * 32}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<32} | {table.row_count:>12,}")

    print()
    print(f"Security-periods: {result.security_periods:,}")
    print(f"Model-eligible periods: {result.model_eligible_periods:,}")
    print("Stock-quarter modeling-panel outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
