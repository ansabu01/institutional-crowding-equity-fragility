"""Evaluate the overall selected model on robustness outcomes."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.robustness import build_prediction_robustness  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_project_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional robustness-model rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build prediction robustness results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refit and replace the prediction-robustness outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build and summarize the prediction-robustness outputs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    paths = get_paths_config()
    project = get_project_config()
    result = build_prediction_robustness(
        sample_path=(
            paths.processed / "modeling" / "network_modeling_sample.parquet"
        ),
        robustness_targets_path=(
            paths.processed / "modeling" / "robustness_targets.parquet"
        ),
        selected_model_path=(
            paths.models / "selected" / "selected_model.joblib"
        ),
        graph_directory=paths.processed / "modeling" / "gnn",
        output_directory=paths.tables / "robustness",
        seed=project.seed,
        overwrite=overwrite,
    )
    performance = pd.read_csv(result.performance_path)

    print()
    print(
        f"Prediction robustness: "
        f"{'created' if result.created else 'already present'}"
    )
    print(
        performance[
            [
                "target",
                "test_observations",
                "test_quarters",
                "mae",
                "r2",
                "mean_quarterly_spearman",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("Prediction-robustness outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
