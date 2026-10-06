"""Train the common starting configuration for all GNN architectures."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.modeling.gnn_training import (  # noqa: E402
    GNN_INITIAL_PARAMETERS,
    train_gnn_architectures,
)
from src.utils.configuration import (  # noqa: E402
    get_paths_config,
    get_project_config,
)


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional retraining flag."""
    parser = argparse.ArgumentParser(
        description="Train all GNN candidates on the frozen graph sample.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain and replace the GNN architecture validation results.",
    )
    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Train or validate the common-configuration GNN candidates."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    paths = get_paths_config()
    project = get_project_config()
    results_directory = paths.tables / "modeling"
    result = train_gnn_architectures(
        graph_directory=paths.processed / "modeling" / "gnn",
        results_directory=results_directory,
        seed=project.seed,
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"
    performance = pd.read_csv(result.validation_path)

    print()
    print(f"GNN architecture results: {status}")
    print(f"Starting configuration: {GNN_INITIAL_PARAMETERS}")
    print(
        performance[
            ["model", "best_epoch", "mae", "rmse", "r2", "mean_quarterly_spearman"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"Selected GNN architecture: {result.selected_architecture}")
    print(f"Training history: {result.history_path}")
    print("GNN architecture validation outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
