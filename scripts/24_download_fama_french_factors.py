"""Download the fixed daily Fama--French factor snapshot."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.fama_french.download import (  # noqa: E402
    download_fama_french_factors,
)
from src.utils.configuration import (  # noqa: E402
    get_fama_french_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional factor-data redownload flag."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the fixed daily factor data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload the archives and replace the factor table.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Download, prepare, and summarize the daily factor data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    source = get_fama_french_config()
    paths = get_paths_config()
    result = download_fama_french_factors(
        source=source,
        manifest_directory=paths.manifest / "fama_french",
        raw_directory=paths.raw / "fama_french",
        factors_path=paths.interim / "fama_french" / "daily_factors.parquet",
        overwrite=overwrite,
    )

    print()
    print(f"{'Dataset':<20} | Status")
    print(f"{'-' * 20}-+-{'-' * 15}")
    print(
        f"{'Five factors':<20} | "
        f"{'downloaded' if result.five_factors_downloaded else 'already present'}"
    )
    print(
        f"{'Momentum':<20} | "
        f"{'downloaded' if result.momentum_downloaded else 'already present'}"
    )
    print(
        f"{'Daily factor table':<20} | "
        f"{'created' if result.factors_created else 'already present'}"
    )
    print()
    print(f"Daily factor observations: {result.factor_rows:,}")
    print("Fama--French factor data: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
