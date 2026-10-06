"""Download the CRSP security history required for identifier mapping."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.crsp.download import (  # noqa: E402
    download_crsp_security_history,
)
from src.utils.configuration import (  # noqa: E402
    get_crsp_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional CRSP redownload flag."""
    parser = argparse.ArgumentParser(
        description="Download CRSP security history through WRDS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and replace the existing CRSP security history.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Download or verify the configured CRSP security history."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_crsp_config()
    paths = get_paths_config()
    result = download_crsp_security_history(
        source=source,
        manifest_path=paths.manifest / "crsp" / "security_history.json",
        output_path=paths.raw / "crsp" / "security_history.parquet",
        overwrite=overwrite,
    )
    status = "downloaded" if result.downloaded else "already present"

    print()
    print(f"{'Dataset':<24} | {'Status':<15} | {'Rows':>12}")
    print(f"{'-' * 24}-+-{'-' * 15}-+-{'-' * 12}")
    print(f"{result.dataset:<24} | {status:<15} | {result.row_count:>12,}")

    print()
    print(f"Source table: {result.source_table}")
    print("CRSP download: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
