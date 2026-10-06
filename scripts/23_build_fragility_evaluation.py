"""Build the held-out economic evaluation of the Fragility Score."""

import argparse
import logging
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.evaluation.fragility import build_fragility_evaluation  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional evaluation rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build the held-out Fragility Score evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing evaluation outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build and summarize the frozen economic-evaluation outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    paths = get_paths_config()
    result = build_fragility_evaluation(
        predictions_path=(
            paths.tables / "modeling" / "selected_model_test_predictions.parquet"
        ),
        sample_path=paths.processed / "modeling" / "modeling_sample.parquet",
        daily_path=paths.raw / "crsp" / "daily_stock.parquet",
        output_directory=paths.tables / "evaluation",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    print()
    print(f"Fragility Score evaluation: {status}")
    print(f"Stock-quarters: {result.stock_quarters:,}")
    print(f"Test quarters: {result.quarters}")
    print("Economic-evaluation outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
