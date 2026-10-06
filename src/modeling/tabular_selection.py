"""Shared tabular-model tuning and evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .feature_sets import (
    MARKET_FEATURES,
    OWNERSHIP_FEATURES,
    RANKED_NETWORK_FEATURES,
    STOCK_LOG_FEATURES,
)


TARGET = "future_max_drawdown_63d"
PREDICTION = "predicted_future_max_drawdown_63d"
RIDGE_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
XGBOOST_GRID = tuple(product((2, 3, 4), (1, 5), (0.8, 1.0)))
XGBOOST_FIXED = {
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
}
XGBOOST_MAX_ROUNDS = 1000
XGBOOST_PATIENCE = 30
INITIAL_PARAMETERS = {
    "ridge": {"alpha": 1.0},
    "xgboost": {
        "n_estimators": 400,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
    },
}

PERFORMANCE_COLUMNS = (
    "stage",
    "model",
    "model_class",
    "feature_group",
    "observations",
    "quarters",
    "mae",
    "rmse",
    "r2",
    "mean_quarterly_spearman",
    "selected",
)
TUNING_COLUMNS = (
    "model",
    "model_class",
    "feature_group",
    "candidate",
    "alpha",
    "n_estimators",
    "max_depth",
    "min_child_weight",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "mae",
    "rmse",
    "r2",
    "mean_quarterly_spearman",
    "selected",
)
XGBOOST_HISTORY_COLUMNS = (
    "model",
    "feature_group",
    "candidate",
    "boosting_round",
    "training_mae",
    "validation_mae",
)
PREDICTION_COLUMNS = (
    "report_period",
    "information_date",
    "permno",
    TARGET,
    PREDICTION,
    "fragility_score",
)


@dataclass(frozen=True)
class TuningResult:
    """Validation results and selected settings for one specification."""

    performance: dict[str, Any]
    tuning: pd.DataFrame
    history: pd.DataFrame
    parameters: dict[str, float | int]


def feature_names(feature_group: str) -> tuple[str, ...]:
    """Return the ordered predictors for one nested information set."""
    groups = {
        "market": MARKET_FEATURES,
        "augmented": MARKET_FEATURES + OWNERSHIP_FEATURES,
        "network_augmented": (
            MARKET_FEATURES + OWNERSHIP_FEATURES + RANKED_NETWORK_FEATURES
        ),
    }
    try:
        return groups[feature_group]
    except KeyError as error:
        raise ValueError(f"Unknown feature group: {feature_group}") from error


def load_sample(path: Path, feature_group: str) -> pd.DataFrame:
    """Load and validate the columns needed for one tabular feature set."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modeling sample not found: {path}")
    columns = [
        "report_period",
        "information_date",
        "permno",
        "sample_split",
        TARGET,
        *feature_names(feature_group),
    ]
    sample = pd.read_parquet(path, columns=columns)
    if set(sample["sample_split"].unique()) != {"train", "validation", "test"}:
        raise RuntimeError("The modeling sample has unexpected split labels.")
    if sample.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("The modeling sample contains duplicate stock-quarters.")
    if sample[TARGET].isna().any() or not sample[TARGET].between(0, 1).all():
        raise RuntimeError("The target must be nonmissing and lie within [0, 1].")
    values = sample[list(feature_names(feature_group))].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise RuntimeError("The modeling features contain infinite values.")
    return sample


def split_sample(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological training, validation, and test samples."""
    return tuple(
        sample.loc[sample["sample_split"].eq(split)].copy()
        for split in ("train", "validation", "test")
    )


def fit_preprocessor(
    frame: pd.DataFrame,
    feature_group: str,
    *,
    standardize: bool,
) -> dict[str, Any]:
    """Estimate all preprocessing quantities from one fitting sample."""
    features = feature_names(feature_group)
    transformed = frame.loc[:, features].astype(float).copy()
    for feature in STOCK_LOG_FEATURES.intersection(features):
        if (transformed[feature].dropna() < 0).any():
            raise RuntimeError(f"Cannot apply log1p to negative values in {feature}.")
        transformed[feature] = np.log1p(transformed[feature])
    lower = transformed.quantile(0.01)
    upper = transformed.quantile(0.99)
    clipped = transformed.clip(lower=lower, upper=upper, axis="columns")
    medians = clipped.median()
    if medians.isna().any():
        raise RuntimeError("At least one modeling feature is entirely missing.")
    filled = clipped.fillna(medians)
    means = filled.mean() if standardize else None
    deviations = filled.std(ddof=0).replace(0, 1) if standardize else None
    return {
        "feature_group": feature_group,
        "features": features,
        "lower": lower,
        "upper": upper,
        "medians": medians,
        "means": means,
        "standard_deviations": deviations,
    }


def transform_features(frame: pd.DataFrame, parameters: dict[str, Any]) -> np.ndarray:
    """Apply fitting-sample preprocessing parameters without re-estimation."""
    features = tuple(parameters["features"])
    transformed = frame.loc[:, features].astype(float).copy()
    for feature in STOCK_LOG_FEATURES.intersection(features):
        transformed[feature] = np.log1p(transformed[feature])
    transformed = transformed.clip(
        lower=parameters["lower"],
        upper=parameters["upper"],
        axis="columns",
    ).fillna(parameters["medians"])
    if parameters["means"] is not None:
        transformed = (
            transformed - parameters["means"]
        ) / parameters["standard_deviations"]
    values = transformed.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("Preprocessing produced non-finite feature values.")
    return values


def make_xgboost(seed: int, **parameters: float | int) -> XGBRegressor:
    """Construct the common XGBoost regression estimator."""
    return XGBRegressor(
        **XGBOOST_FIXED,
        **parameters,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        verbosity=0,
    )


def performance(
    *,
    stage: str,
    model: str,
    model_class: str,
    feature_group: str,
    sample: pd.DataFrame,
    predictions: np.ndarray,
    selected: bool = False,
) -> dict[str, Any]:
    """Calculate the common magnitude and ranking metrics."""
    actual = sample[TARGET].to_numpy(dtype=float)
    comparison = sample[["report_period", TARGET]].copy()
    comparison[PREDICTION] = predictions
    correlations = [
        quarter[TARGET].corr(quarter[PREDICTION], method="spearman")
        for _, quarter in comparison.groupby("report_period")
        if quarter[TARGET].nunique() > 1 and quarter[PREDICTION].nunique() > 1
    ]
    return {
        "stage": stage,
        "model": model,
        "model_class": model_class,
        "feature_group": feature_group,
        "observations": len(sample),
        "quarters": sample["report_period"].nunique(),
        "mae": mean_absolute_error(actual, predictions),
        "rmse": mean_squared_error(actual, predictions) ** 0.5,
        "r2": r2_score(actual, predictions),
        "mean_quarterly_spearman": (
            float(np.mean(correlations)) if correlations else np.nan
        ),
        "selected": selected,
    }


def tune_ridge(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_group: str,
    *,
    stage: str,
) -> TuningResult:
    """Tune Ridge regularization using validation MAE only."""
    preprocessing = fit_preprocessor(train, feature_group, standardize=True)
    train_x = transform_features(train, preprocessing)
    validation_x = transform_features(validation, preprocessing)
    train_y = train[TARGET].to_numpy(dtype=float)
    model_name = f"ridge_{feature_group}"
    rows = []
    for index, alpha in enumerate(RIDGE_ALPHA_GRID, start=1):
        estimator = Ridge(alpha=alpha)
        estimator.fit(train_x, train_y)
        predictions = np.clip(estimator.predict(validation_x), 0, 1)
        metrics = performance(
            stage=stage,
            model=model_name,
            model_class="ridge",
            feature_group=feature_group,
            sample=validation,
            predictions=predictions,
        )
        rows.append(
            {
                "model": model_name,
                "model_class": "ridge",
                "feature_group": feature_group,
                "candidate": f"ridge_{index:02d}",
                "alpha": alpha,
                "n_estimators": np.nan,
                "max_depth": np.nan,
                "min_child_weight": np.nan,
                "learning_rate": np.nan,
                "subsample": np.nan,
                "colsample_bytree": np.nan,
                **{name: metrics[name] for name in (
                    "mae", "rmse", "r2", "mean_quarterly_spearman"
                )},
                "selected": False,
            }
        )
    tuning = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    selected_index = tuning["mae"].idxmin()
    tuning.loc[selected_index, "selected"] = True
    selected = tuning.loc[selected_index]
    parameters = {"alpha": float(selected["alpha"])}
    metrics = performance(
        stage=stage,
        model=model_name,
        model_class="ridge",
        feature_group=feature_group,
        sample=validation,
        predictions=np.clip(
            Ridge(**parameters).fit(train_x, train_y).predict(validation_x),
            0,
            1,
        ),
    )
    return TuningResult(
        metrics,
        tuning,
        pd.DataFrame(columns=XGBOOST_HISTORY_COLUMNS),
        parameters,
    )


def tune_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_group: str,
    seed: int,
    *,
    stage: str,
) -> TuningResult:
    """Tune the prespecified XGBoost grid using validation MAE."""
    preprocessing = fit_preprocessor(train, feature_group, standardize=False)
    train_x = transform_features(train, preprocessing)
    validation_x = transform_features(validation, preprocessing)
    train_y = train[TARGET].to_numpy(dtype=float)
    model_name = f"xgboost_{feature_group}"
    rows = []
    histories = []
    for index, (depth, child_weight, subsample) in enumerate(XGBOOST_GRID, start=1):
        candidate = f"xgboost_{index:02d}"
        estimator = make_xgboost(
            seed,
            n_estimators=XGBOOST_MAX_ROUNDS,
            max_depth=depth,
            min_child_weight=child_weight,
            subsample=subsample,
            early_stopping_rounds=XGBOOST_PATIENCE,
        )
        estimator.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (validation_x, validation[TARGET])],
            verbose=False,
        )
        evaluation = estimator.evals_result()
        rounds = len(evaluation["validation_0"]["mae"])
        histories.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "feature_group": feature_group,
                    "candidate": candidate,
                    "boosting_round": range(1, rounds + 1),
                    "training_mae": evaluation["validation_0"]["mae"],
                    "validation_mae": evaluation["validation_1"]["mae"],
                }
            )
        )
        predictions = np.clip(estimator.predict(validation_x), 0, 1)
        metrics = performance(
            stage=stage,
            model=model_name,
            model_class="xgboost",
            feature_group=feature_group,
            sample=validation,
            predictions=predictions,
        )
        rows.append(
            {
                "model": model_name,
                "model_class": "xgboost",
                "feature_group": feature_group,
                "candidate": candidate,
                "alpha": np.nan,
                "n_estimators": int(estimator.best_iteration) + 1,
                "max_depth": depth,
                "min_child_weight": child_weight,
                "learning_rate": XGBOOST_FIXED["learning_rate"],
                "subsample": subsample,
                "colsample_bytree": XGBOOST_FIXED["colsample_bytree"],
                **{name: metrics[name] for name in (
                    "mae", "rmse", "r2", "mean_quarterly_spearman"
                )},
                "selected": False,
            }
        )
    tuning = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    selected_index = tuning["mae"].idxmin()
    tuning.loc[selected_index, "selected"] = True
    selected = tuning.loc[selected_index]
    parameters = {
        "n_estimators": int(selected["n_estimators"]),
        "max_depth": int(selected["max_depth"]),
        "min_child_weight": int(selected["min_child_weight"]),
        "subsample": float(selected["subsample"]),
    }
    selected_history = pd.concat(histories, ignore_index=True)
    metrics = {
        "stage": stage,
        "model": model_name,
        "model_class": "xgboost",
        "feature_group": feature_group,
        "observations": len(validation),
        "quarters": validation["report_period"].nunique(),
        **{name: selected[name] for name in (
            "mae", "rmse", "r2", "mean_quarterly_spearman"
        )},
        "selected": False,
    }
    return TuningResult(metrics, tuning, selected_history, parameters)


def tune_model(
    model_class: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_group: str,
    seed: int,
    *,
    stage: str,
) -> TuningResult:
    """Dispatch validation-only tuning for one tabular model class."""
    if model_class == "ridge":
        return tune_ridge(train, validation, feature_group, stage=stage)
    if model_class == "xgboost":
        return tune_xgboost(
            train,
            validation,
            feature_group,
            seed,
            stage=stage,
        )
    raise ValueError(f"Unknown tabular model class: {model_class}")


def evaluate_fixed_model(
    model_class: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_group: str,
    seed: int,
    *,
    stage: str,
) -> dict[str, Any]:
    """Evaluate one prespecified tabular model without hyperparameter search."""
    parameters = dict(INITIAL_PARAMETERS[model_class])
    predictions, _ = fit_tabular_model(
        model_class,
        feature_group,
        parameters,
        train,
        validation,
        seed,
    )
    return performance(
        stage=stage,
        model=f"{model_class}_{feature_group}",
        model_class=model_class,
        feature_group=feature_group,
        sample=validation,
        predictions=predictions,
    )


def fit_tabular_model(
    model_class: str,
    feature_group: str,
    parameters: dict[str, float | int],
    fitting_sample: pd.DataFrame,
    prediction_sample: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refit one frozen tabular specification and predict another sample."""
    preprocessing = fit_preprocessor(
        fitting_sample,
        feature_group,
        standardize=model_class == "ridge",
    )
    fitting_x = transform_features(fitting_sample, preprocessing)
    prediction_x = transform_features(prediction_sample, preprocessing)
    if model_class == "ridge":
        estimator: Any = Ridge(alpha=float(parameters["alpha"]))
    elif model_class == "xgboost":
        estimator = make_xgboost(seed, **parameters)
    else:
        raise ValueError(f"Unknown tabular model class: {model_class}")
    estimator.fit(fitting_x, fitting_sample[TARGET].to_numpy(dtype=float))
    predictions = np.clip(estimator.predict(prediction_x), 0, 1)
    return predictions, {
        "family": "tabular",
        "model_class": model_class,
        "model_name": f"{model_class}_{feature_group}",
        "feature_group": feature_group,
        "features": feature_names(feature_group),
        "hyperparameters": parameters,
        "preprocessing": preprocessing,
        "model": estimator,
    }


def mean_predictions(fitting_sample: pd.DataFrame, sample: pd.DataFrame) -> np.ndarray:
    """Return the no-information fitting-sample mean forecast."""
    return np.full(len(sample), float(fitting_sample[TARGET].mean()))


def prediction_table(sample: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Create canonical predictions and within-quarter Fragility Scores."""
    output = sample[["report_period", "information_date", "permno", TARGET]].copy()
    output[PREDICTION] = predictions
    ranks = output.groupby("report_period")[PREDICTION].rank(method="average")
    sizes = output.groupby("report_period")[PREDICTION].transform("size")
    output["fragility_score"] = (100 * (ranks - 1) / (sizes - 1)).fillna(50)
    return output[list(PREDICTION_COLUMNS)]


def save_selection(path: Path, selection: dict[str, Any]) -> None:
    """Write a small human-readable model-selection artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2, sort_keys=True)


def load_selection(path: Path) -> dict[str, Any]:
    """Read a model-selection artifact."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model-selection artifact not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        selection = json.load(file)
    if not isinstance(selection, dict):
        raise RuntimeError(f"Invalid model-selection artifact: {path}")
    return selection


def save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    """Persist the canonical selected-model bundle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


__all__ = [
    "INITIAL_PARAMETERS",
    "PERFORMANCE_COLUMNS",
    "PREDICTION",
    "PREDICTION_COLUMNS",
    "RIDGE_ALPHA_GRID",
    "TARGET",
    "TUNING_COLUMNS",
    "TuningResult",
    "XGBOOST_HISTORY_COLUMNS",
    "feature_names",
    "evaluate_fixed_model",
    "fit_tabular_model",
    "load_sample",
    "load_selection",
    "mean_predictions",
    "performance",
    "prediction_table",
    "save_model_bundle",
    "save_selection",
    "split_sample",
    "tune_model",
]
