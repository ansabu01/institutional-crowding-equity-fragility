"""Evaluate the overall selected model on alternative prediction targets."""

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .tabular_selection import (
    feature_names,
    fit_preprocessor,
    make_xgboost,
    transform_features,
)

_BASE_SAMPLE_COLUMNS = {
    "report_period",
    "information_date",
    "target_window_end",
    "permno",
    "sample_split",
    "future_downside_volatility_63d",
    "future_worst_five_day_return_63d",
}
_ROBUSTNESS_TARGET_COLUMNS = {
    "report_period",
    "information_date",
    "permno",
    "horizon_market_days",
    "target_window_end",
    "future_max_drawdown",
    "target_usable",
}
_TARGETS = (
    "future_downside_volatility_63d",
    "future_worst_five_day_loss_63d",
    "future_max_drawdown_21d",
    "future_max_drawdown_126d",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobustnessModelResult:
    """Paths and dimensions of the robustness-model outputs."""

    created: bool
    performance_path: Path
    predictions_path: Path
    deciles_path: Path
    targets: int


def _require_columns(
    path: Path,
    required_columns: set[str],
    description: str,
) -> None:
    """Require a readable Parquet file with the expected columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        columns = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing = required_columns - columns

    if missing:
        raise RuntimeError(f"Missing columns in {description}: {sorted(missing)}")


def _target_samples(
    sample_path: Path,
    robustness_targets_path: Path,
    model_features: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    """Return chronologically valid samples for all robustness outcomes."""
    sample = pd.read_parquet(
        sample_path,
        columns=list(_BASE_SAMPLE_COLUMNS) + list(model_features),
    )
    alternative = pd.read_parquet(
        robustness_targets_path,
        columns=list(_ROBUSTNESS_TARGET_COLUMNS),
    )
    validation_start = sample.loc[
        sample["sample_split"] == "validation", "information_date"
    ].min()
    test_start = sample.loc[
        sample["sample_split"] == "test", "information_date"
    ].min()
    samples = {}

    downside = sample.copy()
    downside["target_value"] = downside["future_downside_volatility_63d"]
    downside["robustness_window_end"] = downside["target_window_end"]
    samples[_TARGETS[0]] = downside

    worst_loss = sample.copy()
    worst_loss["target_value"] = (
        -worst_loss["future_worst_five_day_return_63d"]
    ).clip(lower=0)
    worst_loss["robustness_window_end"] = worst_loss["target_window_end"]
    samples[_TARGETS[1]] = worst_loss

    for horizon in (21, 126):
        target = alternative.loc[
            (alternative["horizon_market_days"] == horizon)
            & alternative["target_usable"],
            [
                "report_period",
                "information_date",
                "permno",
                "target_window_end",
                "future_max_drawdown",
            ],
        ].rename(
            columns={
                "target_window_end": "robustness_window_end",
                "future_max_drawdown": "target_value",
            }
        )
        merged = sample.drop(
            columns=["target_window_end"],
        ).merge(
            target,
            on=["report_period", "information_date", "permno"],
            how="inner",
            validate="one_to_one",
        )
        samples[f"future_max_drawdown_{horizon}d"] = merged

    for target_name, target_sample in samples.items():
        crosses_boundary = (
            (target_sample["sample_split"] == "train")
            & (target_sample["robustness_window_end"] >= validation_start)
        ) | (
            (target_sample["sample_split"] == "validation")
            & (target_sample["robustness_window_end"] >= test_start)
        )
        valid_target = np.isfinite(target_sample["target_value"])
        samples[target_name] = target_sample.loc[
            ~crosses_boundary & valid_target
        ].copy()

    return samples


def _spearman(first: pd.Series, second: pd.Series) -> float:
    """Return a Spearman rank correlation."""
    return float(first.rank(method="average").corr(second.rank(method="average")))


def _load_selected_bundle(path: Path) -> dict[str, object]:
    """Load the canonical overall selected-model bundle."""
    if not path.is_file():
        raise FileNotFoundError(f"Selected model not found: {path}")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("overall_selection") not in {
        "selected_tabular",
        "selected_gnn",
    }:
        raise RuntimeError("The selected-model bundle is invalid.")
    return bundle


def _fit_tabular_target(
    target_name: str,
    sample: pd.DataFrame,
    seed: int,
    bundle: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit and evaluate one frozen robustness specification."""
    train = sample.loc[sample["sample_split"] == "train"]
    validation = sample.loc[sample["sample_split"] == "validation"]
    test = sample.loc[sample["sample_split"] == "test"]
    fitting = pd.concat([train, validation], ignore_index=True)
    feature_group = str(bundle["feature_group"])
    model_class = str(bundle["model_class"])
    parameters = dict(bundle["hyperparameters"])
    preprocessing = fit_preprocessor(
        fitting,
        feature_group,
        standardize=model_class == "ridge",
    )
    fitting_features = transform_features(fitting, preprocessing)
    test_features = transform_features(test, preprocessing)
    if model_class == "ridge":
        model = Ridge(alpha=float(parameters["alpha"]))
    elif model_class == "xgboost":
        model = make_xgboost(seed, **parameters)
    else:
        raise RuntimeError(f"Unsupported selected tabular class: {model_class}")
    model.fit(fitting_features, fitting["target_value"])
    prediction = np.maximum(model.predict(test_features), 0)

    if target_name != "future_downside_volatility_63d":
        prediction = np.minimum(prediction, 1)

    return _evaluate_target_predictions(
        target_name,
        sample,
        test[
            ["report_period", "information_date", "permno", "target_value"]
        ],
        prediction,
    )


def _evaluate_target_predictions(
    target_name: str,
    full_sample: pd.DataFrame,
    test: pd.DataFrame,
    prediction: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Build common robustness diagnostics from held-out predictions."""
    train = full_sample.loc[full_sample["sample_split"] == "train"]
    validation = full_sample.loc[full_sample["sample_split"] == "validation"]
    output = test[
        ["report_period", "information_date", "permno", "target_value"]
    ].copy()
    output["target"] = target_name
    output["prediction"] = prediction
    ranks = output.groupby("report_period")["prediction"].rank(method="average")
    sizes = output.groupby("report_period")["prediction"].transform("size")
    output["score"] = (100 * (ranks - 1) / (sizes - 1)).fillna(50)
    output["decile"] = np.clip(
        np.floor(output["score"] / 10).astype(int) + 1,
        1,
        10,
    )
    quarterly_correlations = []
    quarterly_deciles = []

    for report_period, quarter in output.groupby("report_period", sort=True):
        quarterly_correlations.append(
            _spearman(quarter["prediction"], quarter["target_value"])
        )
        decile = (
            quarter.groupby("decile", as_index=False)
            .agg(
                observations=("permno", "size"),
                mean_prediction=("prediction", "mean"),
                mean_actual=("target_value", "mean"),
            )
            .assign(report_period=report_period, target=target_name)
        )
        quarterly_deciles.append(decile)

    deciles = pd.concat(quarterly_deciles, ignore_index=True)
    decile_correlations = [
        _spearman(group["decile"], group["mean_actual"])
        for _, group in deciles.groupby("report_period")
    ]
    spread = deciles.pivot(
        index="report_period",
        columns="decile",
        values="mean_actual",
    )
    actual = output["target_value"]
    performance = {
        "target": target_name,
        "train_observations": len(train),
        "validation_observations": len(validation),
        "test_observations": len(test),
        "test_quarters": test["report_period"].nunique(),
        "mae": mean_absolute_error(actual, prediction),
        "rmse": mean_squared_error(actual, prediction) ** 0.5,
        "r2": r2_score(actual, prediction),
        "mean_quarterly_spearman": float(np.mean(quarterly_correlations)),
        "mean_decile_spearman": float(np.mean(decile_correlations)),
        "mean_top_bottom_spread": float((spread[10] - spread[1]).mean()),
    }

    return performance, output, deciles


def _fit_gnn_target(
    target_name: str,
    sample: pd.DataFrame,
    graph_directory: Path,
    seed: int,
    bundle: dict[str, object],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Refit and evaluate the selected GNN for one alternative target."""
    from .gnn_training import refit_selected_gnn_for_target

    raw = refit_selected_gnn_for_target(
        graph_directory,
        sample,
        bundle,
        seed,
    )
    prediction = np.maximum(raw["prediction"].to_numpy(dtype=float), 0)
    if target_name != "future_downside_volatility_63d":
        prediction = np.minimum(prediction, 1)
    test = raw[["report_period", "information_date", "permno", "target_value"]]
    return _evaluate_target_predictions(target_name, sample, test, prediction)


def _output_paths(output_directory: Path) -> tuple[Path, Path, Path]:
    """Return the robustness-model output paths."""
    return (
        output_directory / "prediction_robustness_performance.csv",
        output_directory / "prediction_robustness_test_predictions.parquet",
        output_directory / "prediction_robustness_deciles.csv",
    )


def _validate_outputs(paths: tuple[Path, Path, Path]) -> int:
    """Validate completed robustness-model outputs."""
    performance = pd.read_csv(paths[0])
    predictions = pd.read_parquet(paths[1])
    deciles = pd.read_csv(paths[2])

    if set(performance["target"]) != set(_TARGETS) or len(performance) != 4:
        raise RuntimeError("Invalid prediction-robustness performance output.")

    if (
        predictions.empty
        or set(predictions["target"]) != set(_TARGETS)
        or predictions.duplicated(["target", "report_period", "permno"]).any()
        or predictions.isna().any().any()
        or not predictions["score"].between(0, 100).all()
        or not predictions["decile"].between(1, 10).all()
    ):
        raise RuntimeError("Invalid prediction-robustness predictions.")

    expected_deciles = int(performance["test_quarters"].sum()) * 10

    if len(deciles) != expected_deciles or deciles.isna().any().any():
        raise RuntimeError("Invalid prediction-robustness decile output.")

    return len(performance)


def build_prediction_robustness(
    sample_path: Path,
    robustness_targets_path: Path,
    selected_model_path: Path,
    graph_directory: Path,
    output_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> RobustnessModelResult:
    """Build or validate all frozen prediction-robustness results."""
    sample_path = Path(sample_path)
    robustness_targets_path = Path(robustness_targets_path)
    selected_model_path = Path(selected_model_path)
    graph_directory = Path(graph_directory)
    output_directory = Path(output_directory)
    bundle = _load_selected_bundle(selected_model_path)
    if bundle.get("family") == "tabular":
        selected_features = tuple(bundle.get("features", ()))
        if selected_features != feature_names(str(bundle["feature_group"])):
            raise RuntimeError("The selected-model feature definition is inconsistent.")
    elif bundle.get("family") == "gnn":
        selected_features = feature_names("augmented")
    else:
        raise RuntimeError("Unknown selected-model family.")
    _require_columns(
        sample_path,
        _BASE_SAMPLE_COLUMNS.union(selected_features),
        "modeling sample",
    )
    _require_columns(
        robustness_targets_path,
        _ROBUSTNESS_TARGET_COLUMNS,
        "robustness targets",
    )
    output_paths = _output_paths(output_directory)
    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError(
                "Prediction-robustness outputs are incomplete. "
                "Rebuild with overwrite=True."
            )

        newest_input = max(
            sample_path.stat().st_mtime,
            robustness_targets_path.stat().st_mtime,
            selected_model_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
            Path(__file__).with_name("feature_sets.py").stat().st_mtime,
        )

        if newest_input > min(path.stat().st_mtime for path in output_paths):
            raise RuntimeError(
                "Prediction-robustness inputs or code are newer than the "
                "outputs. Rebuild with overwrite=True."
            )

        targets = _validate_outputs(output_paths)
        return RobustnessModelResult(False, *output_paths, targets)

    samples = _target_samples(
        sample_path,
        robustness_targets_path,
        selected_features,
    )
    performance_rows = []
    prediction_tables = []
    decile_tables = []

    for target_name in _TARGETS:
        _LOGGER.info("Refitting the selected specification for %s.", target_name)
        if bundle["family"] == "tabular":
            performance, predictions, deciles = _fit_tabular_target(
                target_name,
                samples[target_name],
                seed,
                bundle,
            )
        else:
            performance, predictions, deciles = _fit_gnn_target(
                target_name,
                samples[target_name],
                graph_directory,
                seed,
                bundle,
            )
        performance_rows.append(performance)
        prediction_tables.append(predictions)
        decile_tables.append(deciles)

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(performance_rows).to_csv(output_paths[0], index=False)
    pd.concat(prediction_tables, ignore_index=True).to_parquet(
        output_paths[1],
        index=False,
        compression="zstd",
    )
    pd.concat(decile_tables, ignore_index=True).to_csv(output_paths[2], index=False)
    targets = _validate_outputs(output_paths)

    return RobustnessModelResult(True, *output_paths, targets)


__all__ = ["RobustnessModelResult", "build_prediction_robustness"]
