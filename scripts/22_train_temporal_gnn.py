"""Complete the GNN comparison and select the overall predictive model."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.gnn_training import train_gnn_family  # noqa: E402
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_project_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional retraining flag."""
    parser = argparse.ArgumentParser(
        description="Tune the GNN winner and select the overall model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the stage-three and canonical final outputs.",
    )
    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Run or validate stage three and the sole held-out test evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    paths = get_paths_config()
    project = get_project_config()
    results_directory = paths.tables / "modeling"
    result = train_gnn_family(
        graph_directory=paths.processed / "modeling" / "gnn",
        tabular_sample_path=(
            paths.processed / "modeling" / "network_modeling_sample.parquet"
        ),
        stage2_validation_path=(
            results_directory / "stage2_tuned_tabular_validation.csv"
        ),
        stage2_selection_path=(
            paths.models / "network" / "stage2_selected_tabular.json"
        ),
        results_directory=results_directory,
        model_directory=paths.models / "selected",
        figures_directory=paths.figures,
        seed=project.seed,
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"
    validation = pd.read_csv(result.validation_path)
    overall = pd.read_csv(result.overall_validation_path)
    test = pd.read_csv(result.test_path)
    columns = [
        "model",
        "best_epoch",
        "mae",
        "rmse",
        "r2",
        "mean_quarterly_spearman",
    ]

    print()
    print(f"Stage-three and final-model results: {status}")
    print("Initial GNN architecture comparison")
    print(
        validation[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print()
    print("Selected tabular versus tuned GNN")
    print(
        overall[["model", "mae", "rmse", "r2", "mean_quarterly_spearman"]]
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Overall validation winner: {result.selected_model}")
    print()
    print("Sole held-out test performance")
    print(
        test[["model", "mae", "rmse", "r2", "mean_quarterly_spearman"]].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print(f"Test predictions: {result.test_observations:,}")
    print("Overall model-selection outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
