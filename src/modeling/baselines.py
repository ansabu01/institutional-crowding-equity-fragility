"""Stage-one validation: prespecified market-only Ridge versus XGBoost."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .selection_plots import plot_stage_one
from .tabular_selection import (
    PERFORMANCE_COLUMNS,
    evaluate_fixed_model,
    load_sample,
    mean_predictions,
    performance,
    save_selection,
    split_sample,
)


@dataclass(frozen=True)
class BaselineTrainingResult:
    """Stage-one paths and selected market-only model class."""

    created: bool
    selected_model: str
    validation_path: Path
    selection_path: Path


def _output_paths(
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
) -> tuple[Path, Path, Path]:
    """Return the complete stage-one output set."""
    return (
        results_directory / "stage1_model_class_validation.csv",
        model_directory / "stage1_selected_model_class.json",
        figures_directory / "04_01_model_class_validation.pdf",
    )


def _validate_outputs(paths: tuple[Path, Path, Path]) -> str:
    """Validate stage-one artifacts and return the selected class."""
    validation = pd.read_csv(paths[0])
    if tuple(validation.columns) != PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected stage-one validation columns.")
    if set(validation["model"]) != {
        "naive_mean",
        "ridge_market",
        "xgboost_market",
    }:
        raise RuntimeError("Unexpected stage-one candidate models.")
    fitted = validation.loc[validation["model_class"].isin(["ridge", "xgboost"])]
    if fitted["selected"].sum() != 1 or not bool(
        fitted.loc[fitted["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("Stage-one model-class selection is invalid.")
    if not paths[1].is_file() or not paths[2].is_file():
        raise RuntimeError("A stage-one artifact is missing.")
    return str(fitted.loc[fitted["selected"], "model"].iloc[0])


def train_baseline_models(
    sample_path: Path,
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> BaselineTrainingResult:
    """Compare prespecified market-only Ridge and XGBoost specifications."""
    sample_path = Path(sample_path)
    results_directory = Path(results_directory)
    model_directory = Path(model_directory)
    figures_directory = Path(figures_directory)
    paths = _output_paths(results_directory, model_directory, figures_directory)
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(paths):
            raise RuntimeError("Stage-one outputs are incomplete; rebuild with overwrite.")
        selected = _validate_outputs(paths)
        return BaselineTrainingResult(False, selected, paths[0], paths[1])

    sample = load_sample(sample_path, "market")
    train, validation, _ = split_sample(sample)
    ridge = evaluate_fixed_model(
        "ridge", train, validation, "market", seed, stage="model_class"
    )
    xgboost = evaluate_fixed_model(
        "xgboost", train, validation, "market", seed, stage="model_class"
    )
    naive = performance(
        stage="model_class",
        model="naive_mean",
        model_class="none",
        feature_group="none",
        sample=validation,
        predictions=mean_predictions(train, validation),
    )
    rows = [naive, ridge, xgboost]
    selected_index = min((1, 2), key=lambda index: float(rows[index]["mae"]))
    rows[selected_index]["selected"] = True
    table = pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)
    selected_model = str(rows[selected_index]["model"])
    selection = {
        "stage": 1,
        "model": selected_model,
        "model_class": str(rows[selected_index]["model_class"]),
        "feature_group": "market",
        "validation_mae": float(rows[selected_index]["mae"]),
    }
    results_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths[0], index=False)
    save_selection(paths[1], selection)
    plot_stage_one(table, paths[2])
    selected_model = _validate_outputs(paths)
    return BaselineTrainingResult(True, selected_model, paths[0], paths[1])


__all__ = ["BaselineTrainingResult", "train_baseline_models"]
