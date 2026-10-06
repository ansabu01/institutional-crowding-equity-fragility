"""Extract the configured SEC Form 13F archives to Parquet tables."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.sec_13f.extract import (  # noqa: E402
    extract_sec_13f_tables,
)
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_sec_13f_config,
)


_LOGGER = logging.getLogger(__name__)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional table-extraction rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Extract the configured SEC Form 13F tables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing extracted tables.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Extract all configured Form 13F archive tables."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_sec_13f_config()
    paths = get_paths_config()

    archive_count = len(source.archives)
    results = []

    for number, archive in enumerate(source.archives, start=1):
        _LOGGER.info(
            "[%02d/%02d] Processing archive: %s",
            number,
            archive_count,
            archive.archive_id,
        )

        result = extract_sec_13f_tables(
            archive_path=paths.raw / "sec_13f" / archive.filename,
            output_directory=(
                paths.interim / "sec_13f" / "tables" / archive.archive_id
            ),
            overwrite=overwrite,
        )
        results.append((archive.archive_id, result))

    table_names = [Path(table.source_name).stem for table in results[0][1].tables]
    column_width = max(len(name) for name in table_names)

    print()
    print(
        f"{'Archive':<17} | "
        + " | ".join(f"{name:<{column_width}}" for name in table_names)
    )
    print(f"{'-' * 17}-+-" + "-+-".join("-" * column_width for _ in table_names))

    for archive_id, result in results:
        statuses = [
            "created" if table.created else "existing" for table in result.tables
        ]
        print(
            f"{archive_id:<17} | "
            + " | ".join(f"{status:<{column_width}}" for status in statuses)
        )

    table_count = sum(len(result.tables) for _, result in results)
    created_count = sum(
        table.created for _, result in results for table in result.tables
    )

    print()
    print(f"Archives processed: {archive_count}")
    print(f"Tables verified: {table_count}")
    print(f"Tables created: {created_count}")
    print("SEC Form 13F tables: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
