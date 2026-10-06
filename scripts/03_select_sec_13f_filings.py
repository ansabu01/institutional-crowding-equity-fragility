"""Select the historical SEC Form 13F filings used in the analysis."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.sec_13f.select_filings import (  # noqa: E402
    select_13f_filings,
)
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_sec_13f_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional filing-selection rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Select point-in-time SEC Form 13F filings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the filing-selection tables.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the historical filing-selection tables."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_sec_13f_config()
    paths = get_paths_config()
    archive_directories = {
        archive.archive_id: (paths.interim / "sec_13f" / "tables" / archive.archive_id)
        for archive in source.archives
    }
    output_directory = paths.interim / "sec_13f" / "filing_selection"

    result = select_13f_filings(
        archive_directories=archive_directories,
        filing_decisions_path=(output_directory / "filing_decisions.parquet"),
        selected_filings_path=(output_directory / "selected_filings.parquet"),
        report_period_start_date=source.report_period_start_date,
        report_period_end_date=source.report_period_end_date,
        overwrite=overwrite,
    )

    status = "created" if result.created else "already present"
    reporting_periods = pd.to_datetime(
        result.selected_filings["report_period"],
    )
    manager_period_count = (
        result.filing_decisions[["CIK", "PERIODOFREPORT"]].drop_duplicates().shape[0]
    )
    decision_counts = (
        result.filing_decisions["selection_decision"].value_counts().sort_index()
    )
    selected_manager_periods = (
        result.selected_filings[["CIK", "PERIODOFREPORT"]].drop_duplicates().shape[0]
    )

    print()
    print(f"Filing-selection tables: {status}")
    print(
        "Reporting periods: "
        f"{reporting_periods.min().date()} to "
        f"{reporting_periods.max().date()}"
    )
    print(f"Filings assessed: {len(result.filing_decisions):,}")
    print(f"Manager-period groups: {manager_period_count:,}")
    print(f"Selected filings: {len(result.selected_filings):,}")
    print(f"Selected manager-periods: {selected_manager_periods:,}")

    print()
    print(f"{'Decision':<30} | Filings")
    print(f"{'-' * 30}-+-{'-' * 12}")

    for decision, count in decision_counts.items():
        print(f"{decision:<30} | {count:>12,}")

    print()
    print("SEC Form 13F filing selection: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
