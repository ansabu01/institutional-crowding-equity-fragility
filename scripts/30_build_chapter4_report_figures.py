"""Build Chapter 4 figures that combine saved model-comparison outputs."""

import sys
from pathlib import Path

import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.style import (  # noqa: E402
    COLOR_PALETTE,
    SINGLE_PANEL_SIZE,
    apply_matplotlib_layout,
    set_matplotlib_style,
)


TARGET = "future_max_drawdown_63d"
PREDICTION = "predicted_future_max_drawdown_63d"


def main() -> None:
    predictions = pd.read_parquet(
        ROOT
        / "results"
        / "tables"
        / "modeling"
        / "final_models_test_predictions.parquet"
    )
    predictions["realized_decile"] = predictions.groupby(
        ["model", "report_period"], sort=False
    )[TARGET].transform(
        lambda values: pd.qcut(values, 10, labels=False, duplicates="drop") + 1
    )
    predictions["absolute_error"] = (
        predictions[TARGET] - predictions[PREDICTION]
    ).abs()
    error_by_decile = (
        predictions.groupby(["model", "realized_decile"], as_index=False)
        .agg(mean_absolute_error=("absolute_error", "mean"))
    )

    model_order = ["Ridge", "XGBoost", "Temporal GraphSAGE"]
    linestyles = ["-", "--", ":"]

    set_matplotlib_style()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    for index, model in enumerate(model_order):
        group = error_by_decile.loc[error_by_decile["model"].eq(model)]
        axis.plot(
            group["realized_decile"],
            group["mean_absolute_error"],
            marker="o",
            color=COLOR_PALETTE[index],
            linestyle=linestyles[index],
            label=model,
        )
    axis.set_xlabel("Realized test drawdown decile")
    axis.set_ylabel("Mean absolute error")
    axis.set_ylim(bottom=-0.002)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(frameon=False)
    apply_matplotlib_layout(figure)
    figure.savefig(
        ROOT
        / "results"
        / "figures"
        / "04_08_test_mae_by_realized_drawdown_decile.pdf"
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
