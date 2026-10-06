"""Figures documenting the sequential predictive-model comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.style import (
    COLOR_PALETTE,
    SINGLE_PANEL_SIZE,
    apply_matplotlib_layout,
    set_matplotlib_style,
)

from .tabular_selection import PREDICTION, TARGET


_LABELS = {
    "naive_mean": "Naive mean",
    "ridge_market": "Ridge",
    "xgboost_market": "XGBoost",
    "ridge_augmented": "Market + ownership",
    "ridge_network_augmented": "Market + ownership + network",
    "xgboost_augmented": "Market + ownership",
    "xgboost_network_augmented": "Market + ownership + network",
    "selected_tabular": "Selected tabular",
    "selected_gnn": "Selected GNN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "temporal_graphsage": "Temporal GraphSAGE",
}


def _save_validation_bars(
    performance: pd.DataFrame,
    order: list[str],
    output_path: Path,
    *,
    labels: dict[str, str] | None = None,
) -> None:
    """Save one validation-MAE comparison in a common visual format."""
    set_matplotlib_style()
    indexed = performance.set_index("model").loc[order]
    names = [
        (labels or {}).get(model, _LABELS.get(model, model.replace("_", " ").title()))
        for model in order
    ]
    colors = [COLOR_PALETTE[index % len(COLOR_PALETTE)] for index in range(len(order))]
    figure, axis = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    positions = np.arange(len(order))
    values = indexed["mae"].to_numpy(dtype=float)
    axis.scatter(values, positions, color=colors, s=75, zorder=3)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel("Validation MAE")
    axis.set_ylabel("Specification")
    span = max(float(values.max() - values.min()), 0.0005)
    axis.set_xlim(float(values.min() - 0.15 * span), float(values.max() + 0.15 * span))
    apply_matplotlib_layout(figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def plot_stage_one(performance: pd.DataFrame, output_path: Path) -> None:
    """Plot the market-only model-class comparison."""
    _save_validation_bars(
        performance,
        ["ridge_market", "xgboost_market"],
        output_path,
    )


def plot_stage_two(performance: pd.DataFrame, output_path: Path) -> None:
    """Plot the nested tabular information-set comparison."""
    order_map = {"market": 0, "augmented": 1, "network_augmented": 2}
    ordered = performance.assign(
        feature_order=performance["feature_group"].map(order_map)
    ).sort_values("feature_order")
    labels = {
        row.model: {
            "market": "Market",
            "augmented": "Market + ownership",
            "network_augmented": "Market + ownership + network",
        }[row.feature_group]
        for row in ordered.itertuples()
    }
    _save_validation_bars(
        performance,
        ordered["model"].tolist(),
        output_path,
        labels=labels,
    )


def plot_stage_three(performance: pd.DataFrame, output_path: Path) -> None:
    """Plot the selected-tabular versus selected-GNN comparison."""
    _save_validation_bars(
        performance,
        ["selected_tabular", "selected_gnn"],
        output_path,
    )


def plot_gnn_architectures(performance: pd.DataFrame, output_path: Path) -> None:
    """Plot the common-configuration GNN architecture comparison."""
    _save_validation_bars(
        performance,
        ["graphsage", "gat", "temporal_graphsage"],
        output_path,
    )


def plot_test_quarterly(
    predictions: pd.DataFrame,
    naive_predictions: np.ndarray,
    mae_path: Path,
    spearman_path: Path,
) -> pd.DataFrame:
    """Save quarterly test MAE and ranking performance for the final model."""
    comparison = predictions.copy()
    comparison["naive_prediction"] = naive_predictions
    rows = []
    for report_period, quarter in comparison.groupby("report_period", sort=True):
        rows.append(
            {
                "report_period": report_period,
                "selected_model_mae": np.mean(
                    np.abs(quarter[TARGET] - quarter[PREDICTION])
                ),
                "naive_mean_mae": np.mean(
                    np.abs(quarter[TARGET] - quarter["naive_prediction"])
                ),
                "spearman": quarter[TARGET].corr(
                    quarter[PREDICTION], method="spearman"
                ),
            }
        )
    quarterly = pd.DataFrame(rows)
    set_matplotlib_style()
    figure, axis = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    axis.plot(
        quarterly["report_period"],
        quarterly["selected_model_mae"],
        marker="o",
        color=COLOR_PALETTE[0],
        label="Selected model",
    )
    axis.plot(
        quarterly["report_period"],
        quarterly["naive_mean_mae"],
        marker="o",
        linestyle="--",
        color=COLOR_PALETTE[1],
        label="Naive mean",
    )
    axis.set_xlabel("Report quarter")
    axis.set_ylabel("MAE")
    axis.legend()
    apply_matplotlib_layout(figure)
    mae_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(mae_path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    axis.plot(
        quarterly["report_period"],
        quarterly["spearman"],
        marker="o",
        color=COLOR_PALETTE[0],
    )
    axis.axhline(0, color=COLOR_PALETTE[1], linestyle="--")
    axis.set_xlabel("Report quarter")
    axis.set_ylabel("Spearman correlation")
    apply_matplotlib_layout(figure)
    figure.savefig(spearman_path)
    plt.close(figure)
    return quarterly


__all__ = [
    "plot_stage_one",
    "plot_gnn_architectures",
    "plot_stage_three",
    "plot_stage_two",
    "plot_test_quarterly",
]
