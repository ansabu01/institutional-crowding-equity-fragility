"""Train, select, and evaluate the frozen graph-model specifications."""

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from .feature_sets import (
    MANAGER_GNN_FEATURES as _MANAGER_FEATURES,
    MANAGER_LOG_FEATURES as _MANAGER_LOG_FEATURES,
    STOCK_GNN_FEATURES as _STOCK_FEATURES,
    STOCK_LOG_FEATURES as _STOCK_LOG_FEATURES,
)
from .gnn_models import build_gnn
from .selection_plots import (
    plot_gnn_architectures,
    plot_stage_three,
    plot_test_quarterly,
)
from .tabular_selection import (
    PERFORMANCE_COLUMNS as TABULAR_PERFORMANCE_COLUMNS,
    fit_tabular_model,
    load_sample as load_tabular_sample,
    load_selection,
    mean_predictions,
    performance as tabular_performance,
    prediction_table as tabular_prediction_table,
    save_model_bundle,
    save_selection,
    split_sample as split_tabular_sample,
)


_TARGET = "future_max_drawdown_63d"
_PREDICTION = "predicted_future_max_drawdown_63d"
GNN_INITIAL_PARAMETERS = {
    "hidden_size": 16,
    "dropout": 0.20,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 50,
    "patience": 5,
    "minimum_improvement": 0.0001,
}
_HIDDEN_SIZE = int(GNN_INITIAL_PARAMETERS["hidden_size"])
_DROPOUT = float(GNN_INITIAL_PARAMETERS["dropout"])
_LEARNING_RATE = float(GNN_INITIAL_PARAMETERS["learning_rate"])
_WEIGHT_DECAY = float(GNN_INITIAL_PARAMETERS["weight_decay"])
_MAX_EPOCHS = int(GNN_INITIAL_PARAMETERS["max_epochs"])
_PATIENCE = int(GNN_INITIAL_PARAMETERS["patience"])
_MIN_IMPROVEMENT = float(GNN_INITIAL_PARAMETERS["minimum_improvement"])
_ARCHITECTURES = ("graphsage", "gat", "temporal_graphsage")
_TUNING_GRID = ((_HIDDEN_SIZE, _DROPOUT, _LEARNING_RATE),)
_PERFORMANCE_COLUMNS = (
    "model",
    "feature_group",
    "observations",
    "quarters",
    "mae",
    "rmse",
    "r2",
    "mean_quarterly_spearman",
    "best_epoch",
    "selected",
)
_TRAINING_HISTORY_COLUMNS = (
    "model",
    "epoch",
    "training_l1",
    "validation_mae",
)
_TUNING_COLUMNS = (
    "architecture",
    "candidate",
    "hidden_size",
    "dropout",
    "learning_rate",
    "weight_decay",
    "best_epoch",
    "observations",
    "quarters",
    "mae",
    "rmse",
    "r2",
    "mean_quarterly_spearman",
    "selected",
)
_TUNING_HISTORY_COLUMNS = (
    "architecture",
    "candidate",
    "hidden_size",
    "dropout",
    "learning_rate",
    "epoch",
    "training_l1",
    "validation_mae",
)
_CODE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("feature_sets.py"),
    Path(__file__).with_name("gnn_models.py"),
)
_PREDICTION_COLUMNS = (
    "report_period",
    "information_date",
    "permno",
    _TARGET,
    _PREDICTION,
    "fragility_score",
)
_LOGGER = logging.getLogger(__name__)


@dataclass
class _GraphSnapshot:
    """Tensor representation of one quarterly bipartite graph."""

    report_period: pd.Timestamp
    graph_split: str
    quarter_number: int
    stock_features: torch.Tensor
    manager_features: torch.Tensor
    edge_index: torch.Tensor
    portfolio_weight: torch.Tensor
    ownership_share: torch.Tensor
    edge_attributes: torch.Tensor
    target: torch.Tensor
    label_mask: torch.Tensor
    permno: np.ndarray
    information_date: np.ndarray
    global_stock_index: torch.Tensor


@dataclass
class _GraphDataset:
    """Loaded graph sequence and its fitted preprocessing parameters."""

    snapshots: list[_GraphSnapshot]
    global_stocks: int
    stock_preprocessing: dict[str, object]
    manager_preprocessing: dict[str, object]
    edge_mean: torch.Tensor
    edge_standard_deviation: torch.Tensor


@dataclass(frozen=True)
class StaticGnnTrainingResult:
    """Result of training or loading the static GNN validation outputs."""

    created: bool
    validation_path: Path
    history_path: Path
    selected_static_model: str


@dataclass(frozen=True)
class GnnArchitectureTrainingResult:
    """Saved common-configuration results for all three GNN architectures."""

    created: bool
    validation_path: Path
    history_path: Path
    selected_architecture: str


@dataclass(frozen=True)
class GnnTrainingResult:
    """Result of the GNN comparison and overall model selection."""

    created: bool
    selected_model: str
    validation_path: Path
    tuning_path: Path
    tuning_history_path: Path
    overall_validation_path: Path
    test_path: Path
    predictions_path: Path
    model_path: Path
    test_observations: int


def _set_seed(seed: int) -> None:
    """Set deterministic random seeds for CPU training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _require_graph_inputs(directory: Path) -> dict[str, Path]:
    """Return the required graph input paths after basic validation."""
    paths = {
        name: directory / f"{name}.parquet"
        for name in (
            "stock_nodes",
            "manager_nodes",
            "ownership_edges",
            "snapshot_audit",
        )
    }
    for description, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"GNN {description} not found: {path}")
        try:
            if pq.ParquetFile(path).metadata.num_rows == 0:
                raise RuntimeError(f"GNN {description} is empty: {path}")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Invalid GNN {description}: {path}") from error
    return paths


def _apply_logs(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    log_features: set[str],
) -> pd.DataFrame:
    """Apply the fixed non-negative log transformations."""
    transformed = frame.loc[:, features].astype(float).copy()
    for feature in log_features:
        if (transformed[feature].dropna() < 0).any():
            raise RuntimeError(f"Cannot apply log1p to negative values in {feature}.")
        transformed[feature] = np.log1p(transformed[feature])
    return transformed


def _fit_preprocessor(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    log_features: set[str],
) -> dict[str, object]:
    """Fit winsorization, imputation, and scaling on fitting observations."""
    transformed = _apply_logs(frame, features, log_features)
    lower = transformed.quantile(0.01)
    upper = transformed.quantile(0.99)
    clipped = transformed.clip(lower=lower, upper=upper, axis="columns")
    medians = clipped.median()
    if medians.isna().any():
        raise RuntimeError("At least one GNN node feature is entirely missing.")
    filled = clipped.fillna(medians)
    means = filled.mean()
    deviations = filled.std(ddof=0).replace(0, 1)
    return {
        "features": features,
        "log_features": tuple(sorted(log_features)),
        "lower": torch.tensor(lower.to_numpy(), dtype=torch.float32),
        "upper": torch.tensor(upper.to_numpy(), dtype=torch.float32),
        "medians": torch.tensor(medians.to_numpy(), dtype=torch.float32),
        "means": torch.tensor(means.to_numpy(), dtype=torch.float32),
        "standard_deviations": torch.tensor(
            deviations.to_numpy(), dtype=torch.float32
        ),
    }


def _transform_features(
    frame: pd.DataFrame,
    parameters: dict[str, object],
) -> torch.Tensor:
    """Apply fitted node preprocessing and return a float tensor."""
    features = tuple(parameters["features"])
    transformed = _apply_logs(
        frame,
        features,
        set(parameters["log_features"]),
    )
    values = torch.tensor(transformed.to_numpy(), dtype=torch.float32)
    lower = parameters["lower"]
    upper = parameters["upper"]
    medians = parameters["medians"]
    values = torch.maximum(torch.minimum(values, upper), lower)
    values = torch.where(torch.isnan(values), medians, values)
    values = (values - parameters["means"]) / parameters["standard_deviations"]
    if not torch.isfinite(values).all():
        raise RuntimeError("GNN preprocessing produced non-finite node features.")
    return values


def _fit_edge_scaling(
    edge_path: Path,
    audit_path: Path,
    fitting_splits: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit log-edge scaling on the requested graph splits."""
    connection = duckdb.connect()
    placeholders = ", ".join("?" for _ in fitting_splits)
    try:
        row = connection.execute(
            f"""
            SELECT
                AVG(LN(portfolio_weight)),
                STDDEV_POP(LN(portfolio_weight)),
                AVG(LN(reported_ownership_share)),
                STDDEV_POP(LN(reported_ownership_share))
            FROM read_parquet(?) e
            JOIN read_parquet(?) a USING (report_period)
            WHERE a.graph_split IN ({placeholders})
            """,
            [str(edge_path), str(audit_path), *sorted(fitting_splits)],
        ).fetchone()
    finally:
        connection.close()
    mean = torch.tensor([row[0], row[2]], dtype=torch.float32)
    deviation = torch.tensor([row[1], row[3]], dtype=torch.float32)
    if not torch.isfinite(mean).all() or not torch.isfinite(deviation).all():
        raise RuntimeError("Edge preprocessing parameters are invalid.")
    return mean, deviation


def _load_graph_dataset(
    directory: Path,
    fitting_splits: set[str],
    last_labeled_split: str,
) -> _GraphDataset:
    """Load and preprocess the graph sequence through one labeled split."""
    paths = _require_graph_inputs(directory)
    audit = pd.read_parquet(paths["snapshot_audit"]).sort_values("report_period")
    last_period = audit.loc[
        audit["graph_split"].eq(last_labeled_split), "report_period"
    ].max()
    audit = audit.loc[audit["report_period"].le(last_period)].copy()
    periods = set(audit["report_period"])
    stocks = pd.read_parquet(paths["stock_nodes"])
    managers = pd.read_parquet(paths["manager_nodes"])
    stocks = stocks.loc[stocks["report_period"].isin(periods)].copy()
    managers = managers.loc[managers["report_period"].isin(periods)].copy()
    period_splits = audit.set_index("report_period")["graph_split"]

    fitting_stocks = stocks.loc[stocks["sample_split"].isin(fitting_splits)]
    fitting_manager_periods = set(
        audit.loc[audit["graph_split"].isin(fitting_splits), "report_period"]
    )
    fitting_managers = managers.loc[
        managers["report_period"].isin(fitting_manager_periods)
    ]
    stock_preprocessing = _fit_preprocessor(
        fitting_stocks,
        _STOCK_FEATURES,
        _STOCK_LOG_FEATURES,
    )
    manager_preprocessing = _fit_preprocessor(
        fitting_managers,
        _MANAGER_FEATURES,
        _MANAGER_LOG_FEATURES,
    )
    edge_mean, edge_deviation = _fit_edge_scaling(
        paths["ownership_edges"],
        paths["snapshot_audit"],
        fitting_splits,
    )
    unique_permnos = pd.Index(sorted(stocks["permno"].unique()))
    snapshots = []

    for index, report_period in enumerate(audit["report_period"], start=1):
        _LOGGER.info(
            "[%02d/%02d] Loading graph for %s.",
            index,
            len(audit),
            report_period.date(),
        )
        stock = (
            stocks.loc[stocks["report_period"].eq(report_period)]
            .sort_values("security_index")
            .reset_index(drop=True)
        )
        manager = (
            managers.loc[managers["report_period"].eq(report_period)]
            .sort_values("manager_index")
            .reset_index(drop=True)
        )
        edges = pd.read_parquet(
            paths["ownership_edges"],
            filters=[("report_period", "==", report_period)],
        )

        if not np.array_equal(stock["security_index"], np.arange(len(stock))):
            raise RuntimeError(f"Non-contiguous stock indices for {report_period.date()}.")
        if not np.array_equal(manager["manager_index"], np.arange(len(manager))):
            raise RuntimeError(f"Non-contiguous manager indices for {report_period.date()}.")

        edge_values = torch.tensor(
            edges[["portfolio_weight", "reported_ownership_share"]].to_numpy(),
            dtype=torch.float32,
        )
        edge_attributes = (torch.log(edge_values) - edge_mean) / edge_deviation
        graph_split = str(period_splits.loc[report_period])
        label_mask = torch.tensor(
            stock["sample_split"].eq(graph_split).to_numpy(),
            dtype=torch.bool,
        )
        snapshots.append(
            _GraphSnapshot(
                report_period=report_period,
                graph_split=graph_split,
                quarter_number=report_period.to_period("Q").ordinal,
                stock_features=_transform_features(stock, stock_preprocessing),
                manager_features=_transform_features(
                    manager, manager_preprocessing
                ),
                edge_index=torch.tensor(
                    edges[["manager_index", "security_index"]].to_numpy().T,
                    dtype=torch.long,
                ),
                portfolio_weight=edge_values[:, 0],
                ownership_share=edge_values[:, 1],
                edge_attributes=edge_attributes,
                target=torch.tensor(stock[_TARGET].to_numpy(), dtype=torch.float32),
                label_mask=label_mask,
                permno=stock["permno"].to_numpy(),
                information_date=stock["information_date"].to_numpy(),
                global_stock_index=torch.tensor(
                    unique_permnos.get_indexer(stock["permno"]),
                    dtype=torch.long,
                ),
            )
        )

    return _GraphDataset(
        snapshots=snapshots,
        global_stocks=len(unique_permnos),
        stock_preprocessing=stock_preprocessing,
        manager_preprocessing=manager_preprocessing,
        edge_mean=edge_mean,
        edge_standard_deviation=edge_deviation,
    )


def _new_model(
    architecture: str,
    hidden_size: int = _HIDDEN_SIZE,
    dropout: float = _DROPOUT,
) -> nn.Module:
    """Create one graph architecture under the requested configuration."""
    return build_gnn(
        architecture,
        stock_features=len(_STOCK_FEATURES),
        manager_features=len(_MANAGER_FEATURES),
        hidden_size=hidden_size,
        dropout=dropout,
    )


def _graph_inputs(snapshot: _GraphSnapshot) -> tuple[torch.Tensor, ...]:
    """Return the ordered tensors consumed by each graph encoder."""
    return (
        snapshot.stock_features,
        snapshot.manager_features,
        snapshot.edge_index,
        snapshot.portfolio_weight,
        snapshot.ownership_share,
        snapshot.edge_attributes,
    )


def _train_static_epoch(
    model: nn.Module,
    snapshots: list[_GraphSnapshot],
    training_splits: set[str],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train a static model for one pass over labeled quarterly graphs."""
    model.train()
    losses = []
    for snapshot in snapshots:
        if (
            snapshot.graph_split not in training_splits
            or not bool(snapshot.label_mask.any())
        ):
            continue
        optimizer.zero_grad()
        predictions = model(*_graph_inputs(snapshot))
        loss = nn.functional.l1_loss(
            predictions[snapshot.label_mask],
            snapshot.target[snapshot.label_mask],
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _empty_temporal_state(
    dataset: _GraphDataset,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero hidden states and last-seen quarters."""
    state = torch.zeros((dataset.global_stocks, hidden_size))
    last_seen = torch.full((dataset.global_stocks,), -1, dtype=torch.long)
    return state, last_seen


def _previous_temporal_state(
    snapshot: _GraphSnapshot,
    state: torch.Tensor,
    last_seen: torch.Tensor,
) -> torch.Tensor:
    """Return prior states, resetting stocks absent last quarter."""
    previous = state[snapshot.global_stock_index].clone()
    continuing = (
        last_seen[snapshot.global_stock_index] == snapshot.quarter_number - 1
    )
    previous[~continuing] = 0
    return previous


def _update_temporal_state(
    snapshot: _GraphSnapshot,
    current: torch.Tensor,
    state: torch.Tensor,
    last_seen: torch.Tensor,
) -> None:
    """Store detached current states for the next quarter."""
    state[snapshot.global_stock_index] = current.detach()
    last_seen[snapshot.global_stock_index] = snapshot.quarter_number


def _train_temporal_epoch(
    model: nn.Module,
    dataset: _GraphDataset,
    training_splits: set[str],
    optimizer: torch.optim.Optimizer,
    hidden_size: int,
) -> float:
    """Train the temporal model through the fitting graph sequence."""
    state, last_seen = _empty_temporal_state(dataset, hidden_size)
    last_training_period = max(
        snapshot.report_period
        for snapshot in dataset.snapshots
        if snapshot.graph_split in training_splits
    )
    losses = []
    for snapshot in dataset.snapshots:
        if snapshot.report_period > last_training_period:
            break
        previous = _previous_temporal_state(snapshot, state, last_seen)
        if snapshot.graph_split in training_splits and bool(snapshot.label_mask.any()):
            model.train()
            optimizer.zero_grad()
            predictions, current = model(
                previous,
                *_graph_inputs(snapshot),
            )
            loss = nn.functional.l1_loss(
                predictions[snapshot.label_mask],
                snapshot.target[snapshot.label_mask],
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        else:
            model.eval()
            with torch.no_grad():
                _, current = model(previous, *_graph_inputs(snapshot))
        _update_temporal_state(snapshot, current, state, last_seen)
    return float(np.mean(losses))


def _prediction_rows(
    snapshot: _GraphSnapshot,
    predictions: torch.Tensor,
) -> pd.DataFrame:
    """Return labeled predictions for one snapshot."""
    mask = snapshot.label_mask.numpy()
    return pd.DataFrame(
        {
            "report_period": snapshot.report_period,
            "information_date": snapshot.information_date[mask],
            "permno": snapshot.permno[mask],
            _TARGET: snapshot.target[mask].numpy(),
            _PREDICTION: predictions[mask].numpy(),
        }
    )


def _predict_static(
    model: nn.Module,
    dataset: _GraphDataset,
    prediction_split: str,
) -> pd.DataFrame:
    """Predict labeled nodes in one split with a static model."""
    model.eval()
    rows = []
    with torch.no_grad():
        for snapshot in dataset.snapshots:
            if snapshot.graph_split == prediction_split and bool(
                snapshot.label_mask.any()
            ):
                predictions = model(*_graph_inputs(snapshot)).clamp(0, 1).cpu()
                rows.append(_prediction_rows(snapshot, predictions))
    return pd.concat(rows, ignore_index=True)


def _predict_temporal(
    model: nn.Module,
    dataset: _GraphDataset,
    prediction_split: str,
    hidden_size: int,
) -> pd.DataFrame:
    """Predict one split while advancing states through earlier graphs."""
    model.eval()
    state, last_seen = _empty_temporal_state(dataset, hidden_size)
    rows = []
    with torch.no_grad():
        for snapshot in dataset.snapshots:
            previous = _previous_temporal_state(snapshot, state, last_seen)
            predictions, current = model(previous, *_graph_inputs(snapshot))
            _update_temporal_state(snapshot, current, state, last_seen)
            if snapshot.graph_split == prediction_split and bool(
                snapshot.label_mask.any()
            ):
                rows.append(
                    _prediction_rows(snapshot, predictions.clamp(0, 1).cpu())
                )
    return pd.concat(rows, ignore_index=True)


def _performance(
    model_name: str,
    predictions: pd.DataFrame,
    best_epoch: int,
) -> dict[str, object]:
    """Calculate the frozen regression and within-quarter ranking metrics."""
    actual = predictions[_TARGET]
    predicted = predictions[_PREDICTION]
    correlations = [
        quarter[_TARGET].corr(quarter[_PREDICTION], method="spearman")
        for _, quarter in predictions.groupby("report_period")
        if quarter[_TARGET].nunique() > 1 and quarter[_PREDICTION].nunique() > 1
    ]
    return {
        "model": model_name,
        "feature_group": "direct_graph",
        "observations": len(predictions),
        "quarters": predictions["report_period"].nunique(),
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "r2": r2_score(actual, predicted),
        "mean_quarterly_spearman": float(np.mean(correlations)),
        "best_epoch": best_epoch,
        "selected": False,
    }


def _fit_with_validation(
    architecture: str,
    dataset: _GraphDataset,
    seed: int,
    *,
    hidden_size: int = _HIDDEN_SIZE,
    dropout: float = _DROPOUT,
    learning_rate: float = _LEARNING_RATE,
    weight_decay: float = _WEIGHT_DECAY,
) -> tuple[dict[str, object], dict[str, torch.Tensor], pd.DataFrame]:
    """Fit one architecture with validation-only early stopping."""
    _set_seed(seed)
    model = _new_model(architecture, hidden_size, dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    best_mae = np.inf
    patience_reference = np.inf
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history_rows = []

    for epoch in range(1, _MAX_EPOCHS + 1):
        if architecture == "temporal_graphsage":
            training_loss = _train_temporal_epoch(
                model,
                dataset,
                {"train"},
                optimizer,
                hidden_size,
            )
            validation_predictions = _predict_temporal(
                model,
                dataset,
                "validation",
                hidden_size,
            )
        else:
            training_loss = _train_static_epoch(
                model,
                dataset.snapshots,
                {"train"},
                optimizer,
            )
            validation_predictions = _predict_static(
                model,
                dataset,
                "validation",
            )
        validation_mae = mean_absolute_error(
            validation_predictions[_TARGET],
            validation_predictions[_PREDICTION],
        )
        history_rows.append(
            {
                "model": architecture,
                "epoch": epoch,
                "training_l1": training_loss,
                "validation_mae": validation_mae,
            }
        )
        _LOGGER.info(
            "%s epoch %02d: train L1 %.4f, validation MAE %.4f",
            architecture,
            epoch,
            training_loss,
            validation_mae,
        )
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if validation_mae < patience_reference - _MIN_IMPROVEMENT:
            patience_reference = validation_mae
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= _PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    if architecture == "temporal_graphsage":
        predictions = _predict_temporal(
            model, dataset, "validation", hidden_size
        )
    else:
        predictions = _predict_static(model, dataset, "validation")
    history = pd.DataFrame(history_rows, columns=_TRAINING_HISTORY_COLUMNS)
    return _performance(architecture, predictions, best_epoch), best_state, history


def _fit_fixed_epochs(
    architecture: str,
    dataset: _GraphDataset,
    training_splits: set[str],
    epochs: int,
    seed: int,
    *,
    hidden_size: int = _HIDDEN_SIZE,
    dropout: float = _DROPOUT,
    learning_rate: float = _LEARNING_RATE,
    weight_decay: float = _WEIGHT_DECAY,
) -> nn.Module:
    """Refit a selected architecture for its validation-chosen epoch count."""
    _set_seed(seed)
    model = _new_model(architecture, hidden_size, dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for epoch in range(1, epochs + 1):
        if architecture == "temporal_graphsage":
            loss = _train_temporal_epoch(
                model,
                dataset,
                training_splits,
                optimizer,
                hidden_size,
            )
        else:
            loss = _train_static_epoch(
                model,
                dataset.snapshots,
                training_splits,
                optimizer,
            )
        _LOGGER.info(
            "%s refit epoch %02d/%02d: train L1 %.4f",
            architecture,
            epoch,
            epochs,
            loss,
        )
    return model


def _static_validation_path(results_directory: Path) -> Path:
    """Return the static-candidate validation path."""
    return results_directory / "static_gnn_validation_performance.csv"


def _static_history_path(results_directory: Path) -> Path:
    """Return the static-candidate training-history path."""
    return results_directory / "static_gnn_training_history.csv"


def _validate_static_history(path: Path) -> pd.DataFrame:
    """Load and validate the static-candidate epoch histories."""
    history = pd.read_csv(path)
    if tuple(history.columns) != _TRAINING_HISTORY_COLUMNS:
        raise RuntimeError("Unexpected static-GNN training-history columns.")
    if set(history["model"]) != {"graphsage", "gat"}:
        raise RuntimeError("Unexpected models in the static-GNN training history.")
    if history[list(_TRAINING_HISTORY_COLUMNS[1:])].isna().any().any():
        raise RuntimeError("Static-GNN training history contains missing values.")
    if (history[["training_l1", "validation_mae"]] <= 0).any().any():
        raise RuntimeError("Static-GNN training errors must be positive.")
    for model, rows in history.groupby("model", sort=False):
        expected_epochs = list(range(1, len(rows) + 1))
        if rows["epoch"].tolist() != expected_epochs:
            raise RuntimeError(f"Nonconsecutive training epochs for {model}.")
    return history


def _validate_static_output(path: Path) -> pd.DataFrame:
    """Load and validate the two static-candidate results."""
    performance = pd.read_csv(path)
    if tuple(performance.columns) != _PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected static-GNN validation columns.")
    if set(performance["model"]) != {"graphsage", "gat"}:
        raise RuntimeError("Unexpected static-GNN models.")
    if performance["selected"].sum() != 1:
        raise RuntimeError("Exactly one static GNN must be selected.")
    if not bool(performance.loc[performance["mae"].idxmin(), "selected"]):
        raise RuntimeError("The selected static GNN does not minimize validation MAE.")
    return performance


def train_static_gnns(
    graph_directory: Path,
    results_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> StaticGnnTrainingResult:
    """Train GraphSAGE and GAT without opening the test sample."""
    graph_directory = Path(graph_directory)
    results_directory = Path(results_directory)
    graph_paths = _require_graph_inputs(graph_directory)
    output_path = _static_validation_path(results_directory)
    history_path = _static_history_path(results_directory)
    output_paths = (output_path, history_path)
    if all(path.exists() for path in output_paths) and not overwrite:
        input_paths = tuple(graph_paths.values())
        if max(path.stat().st_mtime_ns for path in input_paths) > min(
            path.stat().st_mtime_ns for path in output_paths
        ):
            raise RuntimeError(
                "GNN inputs or code are newer than the static results. "
                "Rebuild with --overwrite."
            )
        performance = _validate_static_output(output_path)
        _validate_static_history(history_path)
        selected = str(performance.loc[performance["selected"], "model"].iloc[0])
        return StaticGnnTrainingResult(False, output_path, history_path, selected)
    if any(path.exists() for path in output_paths) and not overwrite:
        raise RuntimeError("Static-GNN outputs are incomplete. Rebuild with --overwrite.")

    dataset = _load_graph_dataset(graph_directory, {"train"}, "validation")
    rows = []
    histories = []
    for architecture in ("graphsage", "gat"):
        _LOGGER.info("Training static %s candidate.", architecture)
        performance, _, history = _fit_with_validation(architecture, dataset, seed)
        rows.append(performance)
        histories.append(history)
    table = pd.DataFrame(rows, columns=_PERFORMANCE_COLUMNS)
    history = pd.concat(histories, ignore_index=True)
    table.loc[table["mae"].idxmin(), "selected"] = True
    results_directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    history.to_csv(history_path, index=False)
    performance = _validate_static_output(output_path)
    _validate_static_history(history_path)
    selected = str(performance.loc[performance["selected"], "model"].iloc[0])
    return StaticGnnTrainingResult(True, output_path, history_path, selected)


def _architecture_validation_path(results_directory: Path) -> Path:
    """Return the common-configuration architecture-comparison path."""
    return results_directory / "gnn_architecture_validation_performance.csv"


def _architecture_history_path(results_directory: Path) -> Path:
    """Return the common-configuration history path for all architectures."""
    return results_directory / "gnn_architecture_training_history.csv"


def _validate_architecture_output(path: Path) -> pd.DataFrame:
    """Load and validate the three-architecture validation comparison."""
    performance = pd.read_csv(path)
    if tuple(performance.columns) != _PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected GNN architecture-validation columns.")
    if set(performance["model"]) != set(_ARCHITECTURES):
        raise RuntimeError("The GNN architecture comparison is incomplete.")
    if performance["selected"].sum() != 1 or not bool(
        performance.loc[performance["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("The selected GNN architecture does not minimize MAE.")
    return performance


def _validate_architecture_history(path: Path) -> pd.DataFrame:
    """Load and validate epoch histories for all three architectures."""
    history = pd.read_csv(path)
    if tuple(history.columns) != _TRAINING_HISTORY_COLUMNS:
        raise RuntimeError("Unexpected GNN architecture-history columns.")
    if set(history["model"]) != set(_ARCHITECTURES):
        raise RuntimeError("The GNN architecture histories are incomplete.")
    if history[list(_TRAINING_HISTORY_COLUMNS[1:])].isna().any().any():
        raise RuntimeError("The GNN architecture histories contain missing values.")
    if (history[["training_l1", "validation_mae"]] <= 0).any().any():
        raise RuntimeError("GNN training errors must be positive.")
    for model, rows in history.groupby("model", sort=False):
        expected_epochs = list(range(1, len(rows) + 1))
        if rows["epoch"].tolist() != expected_epochs:
            raise RuntimeError(f"Nonconsecutive training epochs for {model}.")
    return history


def train_gnn_architectures(
    graph_directory: Path,
    results_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> GnnArchitectureTrainingResult:
    """Train and save the common starting configuration for all GNNs."""
    graph_directory = Path(graph_directory)
    results_directory = Path(results_directory)
    _require_graph_inputs(graph_directory)
    validation_path = _architecture_validation_path(results_directory)
    history_path = _architecture_history_path(results_directory)
    output_paths = (validation_path, history_path)
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError(
                "GNN architecture outputs are incomplete; rebuild with overwrite."
            )
        performance = _validate_architecture_output(validation_path)
        _validate_architecture_history(history_path)
        selected = str(performance.loc[performance["selected"], "model"].iloc[0])
        return GnnArchitectureTrainingResult(
            False, validation_path, history_path, selected
        )

    dataset = _load_graph_dataset(graph_directory, {"train"}, "validation")
    rows = []
    histories = []
    for architecture in _ARCHITECTURES:
        _LOGGER.info("Training common-configuration %s candidate.", architecture)
        result, _, history = _fit_with_validation(architecture, dataset, seed)
        rows.append(result)
        histories.append(history)

    performance = pd.DataFrame(rows, columns=_PERFORMANCE_COLUMNS)
    performance.loc[performance["mae"].idxmin(), "selected"] = True
    history = pd.concat(histories, ignore_index=True)
    results_directory.mkdir(parents=True, exist_ok=True)
    performance.to_csv(validation_path, index=False)
    history.to_csv(history_path, index=False)
    performance = _validate_architecture_output(validation_path)
    _validate_architecture_history(history_path)
    selected = str(performance.loc[performance["selected"], "model"].iloc[0])
    return GnnArchitectureTrainingResult(True, validation_path, history_path, selected)


def _family_output_paths(
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
) -> tuple[Path, ...]:
    """Return stage-three and canonical final output paths."""
    return (
        results_directory / "stage3_gnn_architecture_validation.csv",
        results_directory / "stage3_gnn_tuning.csv",
        results_directory / "stage3_gnn_tuning_history.csv",
        results_directory / "overall_validation_comparison.csv",
        results_directory / "selected_model_test_performance.csv",
        results_directory / "selected_model_test_predictions.parquet",
        model_directory / "selected_model.joblib",
        model_directory / "overall_selected_model.json",
        results_directory / "predictive_benchmark_performance.csv",
        results_directory / "selected_model_quarterly_test_performance.csv",
        figures_directory / "04_03_gnn_architecture_validation.pdf",
        figures_directory / "04_04_tabular_gnn_validation.pdf",
        figures_directory / "04_05_quarterly_test_mae.pdf",
        figures_directory / "04_05_quarterly_test_spearman.pdf",
    )


def _add_fragility_score(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add the standard within-quarter Fragility Score."""
    output = predictions.copy()
    ranks = output.groupby("report_period")[_PREDICTION].rank(method="average")
    sizes = output.groupby("report_period")[_PREDICTION].transform("size")
    output["fragility_score"] = (100 * (ranks - 1) / (sizes - 1)).fillna(50)
    return output[list(_PREDICTION_COLUMNS)]


def _tune_architecture(
    architecture: str,
    dataset: _GraphDataset,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Tune the validation-selected GNN architecture."""
    rows = []
    histories = []
    for index, (hidden_size, dropout, learning_rate) in enumerate(
        _TUNING_GRID, start=1
    ):
        candidate = f"gnn_{index:02d}"
        _LOGGER.info(
            "Tuning %s [%02d/%02d]: hidden=%d, dropout=%.1f, learning_rate=%.4f",
            architecture,
            index,
            len(_TUNING_GRID),
            hidden_size,
            dropout,
            learning_rate,
        )
        result, _, history = _fit_with_validation(
            architecture,
            dataset,
            seed,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
        )
        rows.append(
            {
                "architecture": architecture,
                "candidate": candidate,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "weight_decay": _WEIGHT_DECAY,
                "best_epoch": int(result["best_epoch"]),
                "observations": int(result["observations"]),
                "quarters": int(result["quarters"]),
                "mae": float(result["mae"]),
                "rmse": float(result["rmse"]),
                "r2": float(result["r2"]),
                "mean_quarterly_spearman": float(
                    result["mean_quarterly_spearman"]
                ),
                "selected": False,
            }
        )
        history = history.assign(
            architecture=architecture,
            candidate=candidate,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
        ).loc[:, _TUNING_HISTORY_COLUMNS]
        histories.append(history)
    tuning = pd.DataFrame(rows, columns=_TUNING_COLUMNS)
    selected_index = tuning["mae"].idxmin()
    tuning.loc[selected_index, "selected"] = True
    selected = tuning.loc[selected_index]
    parameters = {
        "hidden_size": int(selected["hidden_size"]),
        "dropout": float(selected["dropout"]),
        "learning_rate": float(selected["learning_rate"]),
        "weight_decay": float(selected["weight_decay"]),
        "best_epoch": int(selected["best_epoch"]),
    }
    return tuning, pd.concat(histories, ignore_index=True), parameters


def _validate_family_outputs(paths: tuple[Path, ...]) -> tuple[str, int]:
    """Validate the staged selection and sole held-out test evaluation."""
    architectures = pd.read_csv(paths[0])
    tuning = pd.read_csv(paths[1])
    tuning_history = pd.read_csv(paths[2])
    overall = pd.read_csv(paths[3])
    test = pd.read_csv(paths[4])
    predictions = pd.read_parquet(paths[5])
    bundle = joblib.load(paths[6])
    if tuple(architectures.columns) != _PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected GNN architecture-comparison columns.")
    if set(architectures["model"]) != {"graphsage", "gat", "temporal_graphsage"}:
        raise RuntimeError("Unexpected GNN architectures.")
    if architectures["selected"].sum() != 1 or not bool(
        architectures.loc[architectures["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("The initial GNN architecture selection is invalid.")
    if tuple(tuning.columns) != _TUNING_COLUMNS or len(tuning) != len(_TUNING_GRID):
        raise RuntimeError("Unexpected selected-GNN tuning table.")
    if tuning["selected"].sum() != 1 or not bool(
        tuning.loc[tuning["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("The selected GNN hyperparameters do not minimize MAE.")
    if tuple(tuning_history.columns) != _TUNING_HISTORY_COLUMNS:
        raise RuntimeError("Unexpected selected-GNN tuning history.")
    if tuple(overall.columns) != TABULAR_PERFORMANCE_COLUMNS:
        raise RuntimeError("Unexpected overall validation-comparison columns.")
    if set(overall["model"]) != {"selected_tabular", "selected_gnn"}:
        raise RuntimeError("Overall comparison must contain tabular and GNN rows.")
    if overall["selected"].sum() != 1 or not bool(
        overall.loc[overall["mae"].idxmin(), "selected"]
    ):
        raise RuntimeError("Overall model selection does not minimize validation MAE.")
    selected = str(overall.loc[overall["selected"], "model"].iloc[0])
    if len(test) != 1 or not bool(test.loc[0, "selected"]):
        raise RuntimeError("The test table must contain only the overall winner.")
    if bundle.get("overall_selection") != selected:
        raise RuntimeError("The saved model bundle does not match overall selection.")
    if tuple(predictions.columns) != _PREDICTION_COLUMNS:
        raise RuntimeError("Unexpected canonical prediction columns.")
    if predictions.isna().any().any() or not predictions[_PREDICTION].between(0, 1).all():
        raise RuntimeError("Canonical test predictions are invalid.")
    if not predictions["fragility_score"].between(0, 100).all():
        raise RuntimeError("Canonical Fragility Scores are invalid.")
    for path in paths[7:]:
        if not path.is_file():
            raise RuntimeError(f"Missing final model artifact: {path}")
    return selected, len(predictions)


def train_gnn_family(
    graph_directory: Path,
    tabular_sample_path: Path,
    stage2_validation_path: Path,
    stage2_selection_path: Path,
    results_directory: Path,
    model_directory: Path,
    figures_directory: Path,
    seed: int,
    *,
    overwrite: bool = False,
) -> GnnTrainingResult:
    """Tune the GNN winner, select overall, and evaluate the winner once."""
    graph_directory = Path(graph_directory)
    results_directory = Path(results_directory)
    model_directory = Path(model_directory)
    figures_directory = Path(figures_directory)
    tabular_sample_path = Path(tabular_sample_path)
    stage2_validation_path = Path(stage2_validation_path)
    stage2_selection_path = Path(stage2_selection_path)
    graph_paths = _require_graph_inputs(graph_directory)
    architecture_path = _architecture_validation_path(results_directory)
    architecture_history_path = _architecture_history_path(results_directory)
    if not architecture_path.is_file() or not architecture_history_path.is_file():
        raise FileNotFoundError(
            "GNN architecture results are missing. Run script 21 first."
        )
    architecture_performance = _validate_architecture_output(architecture_path)
    _validate_architecture_history(architecture_history_path)
    output_paths = _family_output_paths(
        results_directory, model_directory, figures_directory
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError("Final GNN outputs are incomplete. Rebuild with --overwrite.")
        input_paths = (
            *graph_paths.values(),
            architecture_path,
            architecture_history_path,
            tabular_sample_path,
            stage2_validation_path,
            stage2_selection_path,
            *_CODE_PATHS,
        )
        if max(path.stat().st_mtime_ns for path in input_paths) > min(
            path.stat().st_mtime_ns for path in output_paths
        ):
            raise RuntimeError(
                "A final-GNN input or the training code is newer than the "
                "outputs. Rebuild with --overwrite."
            )
        selected, observations = _validate_family_outputs(output_paths)
        return GnnTrainingResult(False, selected, *output_paths[:7], observations)

    validation_dataset = _load_graph_dataset(
        graph_directory,
        {"train"},
        "validation",
    )
    validation = architecture_performance.copy()
    selected_architecture = str(validation.loc[validation["selected"], "model"].iloc[0])
    tuning, tuning_history, gnn_parameters = _tune_architecture(
        selected_architecture, validation_dataset, seed
    )
    selected_gnn = tuning.loc[tuning["selected"]].iloc[0]

    if not stage2_validation_path.is_file():
        raise FileNotFoundError("Stage-two validation results are missing.")
    stage2_validation = pd.read_csv(stage2_validation_path)
    tabular_row = stage2_validation.loc[stage2_validation["selected"]]
    if len(tabular_row) != 1:
        raise RuntimeError("Stage-two tabular winner is not unique.")
    tabular_row = tabular_row.iloc[0]
    overall_rows = [
        {
            "stage": "overall",
            "model": "selected_tabular",
            "model_class": str(tabular_row["model_class"]),
            "feature_group": str(tabular_row["feature_group"]),
            "observations": int(tabular_row["observations"]),
            "quarters": int(tabular_row["quarters"]),
            "mae": float(tabular_row["mae"]),
            "rmse": float(tabular_row["rmse"]),
            "r2": float(tabular_row["r2"]),
            "mean_quarterly_spearman": float(tabular_row["mean_quarterly_spearman"]),
            "selected": False,
        },
        {
            "stage": "overall",
            "model": "selected_gnn",
            "model_class": selected_architecture,
            "feature_group": "direct_graph",
            "observations": int(selected_gnn["observations"]),
            "quarters": int(selected_gnn["quarters"]),
            "mae": float(selected_gnn["mae"]),
            "rmse": float(selected_gnn["rmse"]),
            "r2": float(selected_gnn["r2"]),
            "mean_quarterly_spearman": float(selected_gnn["mean_quarterly_spearman"]),
            "selected": False,
        },
    ]
    selected_overall_index = min(
        range(2), key=lambda index: overall_rows[index]["mae"]
    )
    overall_rows[selected_overall_index]["selected"] = True
    overall = pd.DataFrame(overall_rows, columns=TABULAR_PERFORMANCE_COLUMNS)
    overall_selection = str(overall_rows[selected_overall_index]["model"])

    tabular_sample = load_tabular_sample(tabular_sample_path, "network_augmented")
    train, tabular_validation, test_sample = split_tabular_sample(tabular_sample)
    fitting_sample = pd.concat([train, tabular_validation], ignore_index=True)
    if overall_selection == "selected_tabular":
        tabular_selection = load_selection(stage2_selection_path)
        predictions, bundle = fit_tabular_model(
            str(tabular_selection["model_class"]),
            str(tabular_selection["feature_group"]),
            dict(tabular_selection["parameters"]),
            fitting_sample,
            test_sample,
            seed,
        )
        prediction_table = tabular_prediction_table(test_sample, predictions)
        test_evaluation_sample = test_sample
        test_model_class = str(tabular_selection["model_class"])
        test_feature_group = str(tabular_selection["feature_group"])
        test_model_name = str(tabular_selection["model"])
    else:
        del validation_dataset
        final_dataset = _load_graph_dataset(
            graph_directory,
            {"train", "validation"},
            "test",
        )
        model = _fit_fixed_epochs(
            selected_architecture,
            final_dataset,
            {"train", "validation"},
            int(gnn_parameters["best_epoch"]),
            seed,
            hidden_size=int(gnn_parameters["hidden_size"]),
            dropout=float(gnn_parameters["dropout"]),
            learning_rate=float(gnn_parameters["learning_rate"]),
            weight_decay=float(gnn_parameters["weight_decay"]),
        )
        if selected_architecture == "temporal_graphsage":
            raw_predictions = _predict_temporal(
                model,
                final_dataset,
                "test",
                int(gnn_parameters["hidden_size"]),
            )
        else:
            raw_predictions = _predict_static(model, final_dataset, "test")
        prediction_table = _add_fragility_score(raw_predictions)
        bundle = {
            "family": "gnn",
            "architecture": selected_architecture,
            "hyperparameters": gnn_parameters,
            "stock_features": _STOCK_FEATURES,
            "manager_features": _MANAGER_FEATURES,
            "state_dict": model.state_dict(),
            "stock_preprocessing": final_dataset.stock_preprocessing,
            "manager_preprocessing": final_dataset.manager_preprocessing,
            "edge_mean": final_dataset.edge_mean,
            "edge_standard_deviation": final_dataset.edge_standard_deviation,
        }
        predictions = prediction_table[_PREDICTION].to_numpy()
        test_evaluation_sample = prediction_table
        test_model_class = selected_architecture
        test_feature_group = "direct_graph"
        test_model_name = selected_architecture

    bundle["overall_selection"] = overall_selection
    test_metrics = tabular_performance(
        stage="test",
        model=test_model_name,
        model_class=test_model_class,
        feature_group=test_feature_group,
        sample=test_evaluation_sample,
        predictions=predictions,
        selected=True,
    )
    test = pd.DataFrame([test_metrics], columns=TABULAR_PERFORMANCE_COLUMNS)
    validation_naive = mean_predictions(train, tabular_validation)
    test_naive = mean_predictions(fitting_sample, test_sample)
    benchmarks = pd.DataFrame(
        [
            {
                "sample_split": "validation",
                **tabular_performance(
                    stage="benchmark",
                    model="naive_mean",
                    model_class="none",
                    feature_group="none",
                    sample=tabular_validation,
                    predictions=validation_naive,
                ),
            },
            {
                "sample_split": "test",
                **tabular_performance(
                    stage="benchmark",
                    model="naive_mean",
                    model_class="none",
                    feature_group="none",
                    sample=test_sample,
                    predictions=test_naive,
                ),
            },
        ]
    )
    selection = {
        "stage": 3,
        "overall_selection": overall_selection,
        "selected_model": test_model_name,
        "model_class": test_model_class,
        "feature_group": test_feature_group,
        "validation_mae": float(overall_rows[selected_overall_index]["mae"]),
    }

    results_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_paths[0], index=False)
    tuning.to_csv(output_paths[1], index=False)
    tuning_history.to_csv(output_paths[2], index=False)
    overall.to_csv(output_paths[3], index=False)
    test.to_csv(output_paths[4], index=False)
    prediction_table.to_parquet(output_paths[5], index=False, compression="zstd")
    save_model_bundle(output_paths[6], bundle)
    save_selection(output_paths[7], selection)
    benchmarks.to_csv(output_paths[8], index=False)
    plot_gnn_architectures(validation, output_paths[10])
    plot_stage_three(overall, output_paths[11])
    quarterly = plot_test_quarterly(
        prediction_table,
        test_naive,
        output_paths[12],
        output_paths[13],
    )
    quarterly.to_csv(output_paths[9], index=False)
    selected_model, observations = _validate_family_outputs(output_paths)
    return GnnTrainingResult(
        True,
        selected_model,
        *output_paths[:7],
        observations,
    )


def refit_selected_gnn_for_target(
    graph_directory: Path,
    target_sample: pd.DataFrame,
    selected_bundle: dict[str, object],
    seed: int,
) -> pd.DataFrame:
    """Refit the selected GNN with fixed settings for one alternative target."""
    if selected_bundle.get("family") != "gnn":
        raise ValueError("The selected-model bundle is not a GNN.")
    required = {"report_period", "permno", "sample_split", "target_value"}
    if missing := required - set(target_sample.columns):
        raise RuntimeError(f"Alternative GNN target is missing columns: {sorted(missing)}")
    dataset = _load_graph_dataset(
        Path(graph_directory),
        {"train", "validation"},
        "test",
    )
    for snapshot in dataset.snapshots:
        period = target_sample.loc[
            target_sample["report_period"].eq(snapshot.report_period),
            ["permno", "sample_split", "target_value"],
        ]
        if period.duplicated("permno").any():
            raise RuntimeError("Alternative GNN target contains duplicate stock-quarters.")
        period = period.loc[period["sample_split"].eq(snapshot.graph_split)]
        values = pd.Series(snapshot.permno).map(
            period.set_index("permno")["target_value"]
        )
        mask = values.notna().to_numpy()
        snapshot.label_mask = torch.tensor(mask, dtype=torch.bool)
        snapshot.target = torch.tensor(
            values.fillna(0).to_numpy(), dtype=torch.float32
        )
    parameters = dict(selected_bundle["hyperparameters"])
    architecture = str(selected_bundle["architecture"])
    model = _fit_fixed_epochs(
        architecture,
        dataset,
        {"train", "validation"},
        int(parameters["best_epoch"]),
        seed,
        hidden_size=int(parameters["hidden_size"]),
        dropout=float(parameters["dropout"]),
        learning_rate=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    if architecture == "temporal_graphsage":
        predictions = _predict_temporal(
            model,
            dataset,
            "test",
            int(parameters["hidden_size"]),
        )
    else:
        predictions = _predict_static(model, dataset, "test")
    return predictions.rename(
        columns={_TARGET: "target_value", _PREDICTION: "prediction"}
    )


__all__ = [
    "GNN_INITIAL_PARAMETERS",
    "GnnArchitectureTrainingResult",
    "GnnTrainingResult",
    "StaticGnnTrainingResult",
    "train_gnn_architectures",
    "train_gnn_family",
    "train_static_gnns",
    "refit_selected_gnn_for_target",
]
