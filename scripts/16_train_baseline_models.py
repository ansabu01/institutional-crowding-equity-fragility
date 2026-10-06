"""Run the market-only model-class comparison."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.baselines import train_baseline_models  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_project_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional baseline-model rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Compare prespecified market-only Ridge and XGBoost models.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the stage-one validation outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the stage-one model-class outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    project = get_project_config()
    result = train_baseline_models(
        sample_path=paths.processed / "modeling" / "modeling_sample.parquet",
        results_directory=paths.tables / "modeling",
        model_directory=paths.models / "baseline",
        figures_directory=paths.figures,
        seed=project.seed,
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"
    validation = pd.read_csv(result.validation_path)

    print()
    print(f"Stage-one model-class comparison: {status}")
    print("Market-only validation performance")
    print(
        validation[["model", "mae", "rmse", "r2", "mean_quarterly_spearman"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print(f"Selected model class: {result.selected_model}")
    print("Stage-one validation outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
