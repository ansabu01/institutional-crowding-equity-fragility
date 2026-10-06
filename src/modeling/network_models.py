"""Stage-two tabular feature comparison and winner-only tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .selection_plots import plot_stage_two
from .tabular_selection import (
    PERFORMANCE_COLUMNS,
    XGBOOST_HISTORY_COLUMNS,
    evaluate_fixed_model,
    load_sample,
    load_selection,
    save_selection,
    split_sample,
    tune_model,
)


@dataclass(frozen=True)
class NetworkTrainingResult:
    """Stage-two paths and tuned selected tabular specification."""

    created: bool
    selected_model: str
    validation_path: Path
    tuning_path: Path
    tuning_history_path: Path
    tuned_validation_path: Path
    selection_path: Path


def _output_paths(
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
) -> tuple[Path, ...]:
    """Return all stage-two output paths."""
    return (
        results_directory / "stage2_tabular_feature_validation.csv",
        results_directory / "stage2_selected_tabular_tuning.csv",
        results_directory / "stage2_selected_tabular_tuning_history.csv",
        results_directory / "stage2_tuned_tabular_validation.csv",
        model_directory / "stage2_selected_tabular.json",
        figures_directory / "04_02_tabular_feature_validation.pdf",
    )


def _validate_outputs(paths: tuple[Path, ...], model_class: str) -> str:
    """Validate the feature comparison and winner-only tuning outputs."""
    validation = pd.read_csv(paths[0])
    tuning = pd.read_csv(paths[1])
    history = pd.read_csv(paths[2])
    tuned = pd.read_csv(paths[3])
    if tuple(validation.columns) != PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected stage-two feature-comparison columns.")
    if set(validation["feature_group"]) != {
        "market",
        "augmented",
        "network_augmented",
    } or not validation["model_class"].eq(model_class).all():
        raise RuntimeError("Stage two does not contain the required feature sets.")
    if validation["selected"].sum() != 1 or not bool(
        validation.loc[validation["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("Stage-two feature-set selection is invalid.")
    expected = 5 if model_class == "ridge" else 12
    if len(tuning) != expected or tuning["selected"].sum() != 1 or not bool(
        tuning.loc[tuning["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("Stage-two winner-only tuning is invalid.")
    if model_class == "xgboost":
        if tuple(history.columns) != XGBOOST_HISTORY_COLUMNS or history.empty:
            raise RuntimeError("Invalid selected-XGBoost tuning history.")
    elif not history.empty:
        raise RuntimeError("Selected Ridge tuning should not have boosting history.")
    if tuple(tuned.columns) != PERFORMANCE_COLUMNS or len(tuned) != 1:
        raise RuntimeError("The tuned tabular finalist must contain one row.")
    selected_group = str(validation.loc[validation["selected"], "feature_group"].iloc[0])
    if str(tuned.loc[0, "feature_group"]) != selected_group:
        raise RuntimeError("The tuned tabular finalist changed feature set.")
    if not paths[4].is_file() or not paths[5].is_file():
        raise RuntimeError("A stage-two artifact is missing.")
    return str(tuned.loc[0, "model"])


def train_network_models(
    sample_path: Path,
    stage1_selection_path: Path,
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> NetworkTrainingResult:
    """Compare feature sets, then tune only the selected tabular specification."""
    sample_path = Path(sample_path)
    stage1_selection_path = Path(stage1_selection_path)
    results_directory = Path(results_directory)
    model_directory = Path(model_directory)
    figures_directory = Path(figures_directory)
    stage1 = load_selection(stage1_selection_path)
    model_class = str(stage1["model_class"])
    paths = _output_paths(results_directory, model_directory, figures_directory)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(paths):
            raise RuntimeError("Stage-two outputs are incomplete; rebuild with overwrite.")
        selected = _validate_outputs(paths, model_class)
        return NetworkTrainingResult(False, selected, *paths[:5])

    sample = load_sample(sample_path, "network_augmented")
    train, validation, _ = split_sample(sample)
    rows = [
        evaluate_fixed_model(
            model_class,
            train,
            validation,
            feature_group,
            seed,
            stage="tabular_features",
        )
        for feature_group in ("market", "augmented", "network_augmented")
    ]
    selected_index = min(range(3), key=lambda index: float(rows[index]["mae"]))
    rows[selected_index]["selected"] = True
    validation_table = pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)
    selected_group = str(rows[selected_index]["feature_group"])
    tuned = tune_model(
        model_class,
        train,
        validation,
        selected_group,
        seed,
        stage="tabular_tuning",
    )
    tuned_row = dict(tuned.performance)
    tuned_row["selected"] = True
    tuned_validation = pd.DataFrame([tuned_row], columns=PERFORMANCE_COLUMNS)
    selection = {
        "stage": 2,
        "model": str(tuned_row["model"]),
        "model_class": model_class,
        "feature_group": selected_group,
        "parameters": tuned.parameters,
        "validation_mae": float(tuned_row["mae"]),
    }

    results_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    validation_table.to_csv(paths[0], index=False)
    tuned.tuning.to_csv(paths[1], index=False)
    tuned.history.to_csv(paths[2], index=False)
    tuned_validation.to_csv(paths[3], index=False)
    save_selection(paths[4], selection)
    plot_stage_two(validation_table, paths[5])
    selected_model = _validate_outputs(paths, model_class)
    return NetworkTrainingResult(True, selected_model, *paths[:5])


__all__ = ["NetworkTrainingResult", "train_network_models"]
