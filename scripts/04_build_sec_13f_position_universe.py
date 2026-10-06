"""Build SEC Form 13F positions, the filter audit, and the filing audit."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.sec_13f.build_universe import (  # noqa: E402
    build_13f_position_universe,
)
from src.utils.configuration import (  # noqa: E402
    get_sec_13f_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional universe-rebuild flag."""
    parser = argparse.ArgumentParser(
        description=("Build the historical SEC Form 13F position universe.")
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing position-universe tables.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build or check the three SEC position-universe outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_sec_13f_config()
    paths = get_paths_config()
    sec_directory = paths.interim / "sec_13f"
    archive_directories = {
        archive.archive_id: (sec_directory / "tables" / archive.archive_id)
        for archive in source.archives
    }

    result = build_13f_position_universe(
        selected_filings_path=(
            sec_directory / "filing_selection" / "selected_filings.parquet"
        ),
        archive_directories=archive_directories,
        output_directory=(sec_directory / "position_universe"),
        overwrite=overwrite,
    )

    status = "created" if result.created else "already present"

    print()
    print(f"{'Table':<28} | {'Status':<15} | Rows")
    print(f"{'-' * 28}-+-{'-' * 15}-+-{'-' * 12}")

    for table in result.tables:
        print(f"{table.name:<28} | {status:<15} | {table.row_count:>12,}")

    print()
    print("SEC Form 13F position universe: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
