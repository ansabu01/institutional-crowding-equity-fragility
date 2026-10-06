"""Run the nested tabular feature-set comparison."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.network_models import train_network_models  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_project_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional network-model rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Tune the winning model class across nested feature sets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace the stage-two validation outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Create or validate the stage-two feature-set outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    paths = get_paths_config()
    project = get_project_config()
    modeling_directory = paths.processed / "modeling"
    results_directory = paths.tables / "modeling"
    result = train_network_models(
        sample_path=modeling_directory / "network_modeling_sample.parquet",
        stage1_selection_path=(
            paths.models / "baseline" / "stage1_selected_model_class.json"
        ),
        results_directory=results_directory,
        model_directory=paths.models / "network",
        figures_directory=paths.figures,
        seed=project.seed,
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"

    validation_comparison = pd.read_csv(result.validation_path)
    tuned_validation = pd.read_csv(result.tuned_validation_path)
    performance_columns = [
        "model",
        "mae",
        "rmse",
        "r2",
        "mean_quarterly_spearman",
    ]

    print()
    print(f"Stage-two tabular comparison: {status}")
    print("Nested feature-set validation performance")
    print(
        validation_comparison[performance_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("Tuned selected tabular specification")
    print(
        tuned_validation[performance_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Selected tabular finalist: {result.selected_model}")
    print("Stage-two validation outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
