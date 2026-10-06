"""Build the analysis-ready SEC Form 13F-CRSP holdings panel."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.panel.holdings import (  # noqa: E402
    build_sec_13f_crsp_holdings,
)
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional mapped-holdings rebuild flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the point-in-time U.S. common-equity holdings panel."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the mapped-holdings tables.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or load the mapped SEC Form 13F-CRSP holdings tables."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    sec_directory = paths.interim / "sec_13f"
    result = build_sec_13f_crsp_holdings(
        positions_path=(
            sec_directory
            / "position_universe"
            / "manager_security_positions.parquet"
        ),
        mapping_path=paths.interim / "crsp" / "security_mapping.parquet",
        selected_filings_path=(
            sec_directory
            / "filing_selection"
            / "selected_filings.parquet"
        ),
        output_directory=paths.interim / "sec_13f_crsp",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Mapped holdings: {status}")
    print(f"{'Table':<32} | Rows")
    print(f"{'-' * 32}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<32} | {table.row_count:>12,}")

    print()
    print("Mapped-holdings outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
