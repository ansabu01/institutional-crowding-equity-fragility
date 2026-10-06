"""Download and verify the configured SEC Form 13F archives."""

import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.sec_13f.download import (  # noqa: E402
    ArchiveDownloadResult,
    download_archive,
)
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_sec_13f_config,
)


_LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Download or verify all configured SEC Form 13F archives."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_sec_13f_config()
    paths = get_paths_config()
    manifest_directory = paths.manifest / "sec_13f"
    output_directory = paths.raw / "sec_13f"

    archive_count = len(source.archives)
    results: list[ArchiveDownloadResult] = []

    for number, archive in enumerate(source.archives, start=1):
        _LOGGER.info(
            "[%02d/%02d] Processing archive: %s",
            number,
            archive_count,
            archive.archive_id,
        )

        result = download_archive(
            archive=archive,
            manifest_path=(
                manifest_directory / f"{archive.archive_id}.json"
            ),
            output_directory=output_directory,
        )
        results.append(result)

    print()
    print(f"{'No.':>3} | {'Archive':<17} | Status")
    print(f"{'-' * 3}-+-{'-' * 17}-+-{'-' * 15}")

    for number, result in enumerate(results, start=1):
        status = "downloaded" if result.downloaded else "already present"
        print(f"{number:>3} | {result.archive_id:<17} | {status}")

    downloaded_count = sum(result.downloaded for result in results)

    print(f"{'-' * 23}-+-{'-' * 15}")
    print(f"Archives verified:      | {archive_count}")
    print(f"Archives downloaded:    | {downloaded_count}")
    print("SEC Form 13F archives:  | PASS")


if __name__ == "__main__":
    main()
