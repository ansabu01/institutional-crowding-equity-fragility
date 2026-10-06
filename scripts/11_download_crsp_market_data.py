"""Download the fixed CRSP daily stock and market-index snapshots."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.data.crsp.download_daily_stock import (  # noqa: E402
    download_crsp_daily_stock,
)
from src.data.crsp.download_market_index import (  # noqa: E402
    download_crsp_market_index,
)
from src.utils.configuration import (  # noqa: E402
    get_crsp_config,
    get_paths_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional CRSP market-data redownload flag."""
    parser = argparse.ArgumentParser(
        description="Download the fixed CRSP daily market data through WRDS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and replace the existing CRSP market-data files.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Download or validate the configured CRSP daily market data."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    source = get_crsp_config()
    paths = get_paths_config()
    stock_result = download_crsp_daily_stock(
        source=source,
        universe_path=(
            paths.interim
            / "sec_13f_crsp"
            / "manager_security_holdings.parquet"
        ),
        manifest_path=paths.manifest / "crsp" / "daily_stock.json",
        output_path=paths.raw / "crsp" / "daily_stock.parquet",
        overwrite=overwrite,
    )
    index_result = download_crsp_market_index(
        source=source,
        manifest_path=paths.manifest / "crsp" / "daily_market_index.json",
        output_path=paths.raw / "crsp" / "daily_market_index.parquet",
        overwrite=overwrite,
    )

    print()
    print(f"{'Dataset':<24} | {'Status':<15} | {'Rows':>12}")
    print(f"{'-' * 24}-+-{'-' * 15}-+-{'-' * 12}")

    for result in (stock_result, index_result):
        status = "downloaded" if result.downloaded else "already present"
        print(f"{result.dataset:<24} | {status:<15} | {result.row_count:>12,}")

    print()
    print("CRSP market-data download: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
