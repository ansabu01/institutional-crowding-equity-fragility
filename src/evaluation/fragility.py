"""Build the held-out economic evaluation of the Fragility Score."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .metrics import downside_volatility


_PREDICTION_COLUMNS = {
    "report_period",
    "information_date",
    "permno",
    "future_max_drawdown_63d",
    "predicted_future_max_drawdown_63d",
    "fragility_score",
}
_SAMPLE_COLUMNS = {
    "report_period",
    "information_date",
    "target_window_start",
    "target_window_end",
    "permno",
    "sample_split",
    "future_max_drawdown_63d",
    "future_cumulative_return_63d",
    "future_downside_volatility_63d",
    "future_worst_five_day_return_63d",
}
_DAILY_COLUMNS = {"permno", "dlycaldt", "dlyret"}
_PORTFOLIOS = (
    "all_stocks",
    "exclude_most_fragile_decile",
    "least_fragile_decile",
    "most_fragile_decile",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FragilityEvaluationResult:
    """Paths and dimensions of the economic-evaluation outputs."""

    created: bool
    stock_quarters_path: Path
    deciles_path: Path
    correlations_path: Path
    portfolios_path: Path
    stock_quarters: int
    quarters: int


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


def _output_paths(output_directory: Path) -> tuple[Path, Path, Path, Path]:
    """Return the four economic-evaluation output paths."""
    return (
        output_directory / "fragility_stock_quarters.parquet",
        output_directory / "fragility_decile_quarterly.csv",
        output_directory / "fragility_rank_correlations.csv",
        output_directory / "fragility_portfolio_quarterly.csv",
    )


def _build_stock_quarters(
    predictions_path: Path,
    sample_path: Path,
) -> pd.DataFrame:
    """Combine held-out predictions with their realized forward outcomes."""
    predictions = pd.read_parquet(predictions_path)
    sample = pd.read_parquet(sample_path, columns=list(_SAMPLE_COLUMNS))
    sample = sample.loc[sample["sample_split"] == "test"].drop(columns=["sample_split"])

    if predictions.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("Baseline predictions contain duplicate stock-quarters.")

    if sample.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("The test sample contains duplicate stock-quarters.")

    panel = predictions.merge(
        sample,
        on=["report_period", "information_date", "permno"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_sample"),
    )

    if len(panel) != len(predictions) or len(panel) != len(sample):
        raise RuntimeError("Predictions and test-sample observations do not match.")

    sample_target = panel.pop("future_max_drawdown_63d_sample")

    if not np.allclose(panel["future_max_drawdown_63d"], sample_target):
        raise RuntimeError("Prediction and modeling-sample targets do not match.")

    panel["fragility_decile"] = np.clip(
        np.floor(panel["fragility_score"] / 10).astype(int) + 1,
        1,
        10,
    )
    panel["future_worst_five_day_loss_63d"] = (
        -panel.pop("future_worst_five_day_return_63d")
    ).clip(lower=0)

    columns = [
        "report_period",
        "information_date",
        "target_window_start",
        "target_window_end",
        "permno",
        "predicted_future_max_drawdown_63d",
        "fragility_score",
        "fragility_decile",
        "future_max_drawdown_63d",
        "future_downside_volatility_63d",
        "future_worst_five_day_loss_63d",
        "future_cumulative_return_63d",
    ]

    return panel[columns].sort_values(["report_period", "permno"])


def _build_decile_results(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarterly realized outcomes for each score decile."""
    return (
        panel.groupby(
            ["report_period", "information_date", "fragility_decile"],
            as_index=False,
        )
        .agg(
            stocks=("permno", "size"),
            mean_predicted_drawdown=(
                "predicted_future_max_drawdown_63d",
                "mean",
            ),
            mean_realized_drawdown=("future_max_drawdown_63d", "mean"),
            median_realized_drawdown=("future_max_drawdown_63d", "median"),
            mean_downside_volatility=("future_downside_volatility_63d", "mean"),
            mean_worst_five_day_loss=(
                "future_worst_five_day_loss_63d",
                "mean",
            ),
            mean_cumulative_return=("future_cumulative_return_63d", "mean"),
        )
        .sort_values(["report_period", "fragility_decile"])
    )


def _spearman(first: pd.Series, second: pd.Series) -> float:
    """Return a Spearman rank correlation without an external dependency."""
    return float(first.rank(method="average").corr(second.rank(method="average")))


def _build_rank_correlations(
    panel: pd.DataFrame,
    deciles: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize stock-level ranking and decile-level monotonicity by quarter."""
    rows = []

    for report_period, quarter in panel.groupby("report_period", sort=True):
        quarter_deciles = deciles.loc[deciles["report_period"] == report_period]
        mean_drawdown = quarter_deciles.set_index("fragility_decile")[
            "mean_realized_drawdown"
        ]
        rows.append(
            {
                "report_period": report_period,
                "information_date": quarter["information_date"].iloc[0],
                "stocks": len(quarter),
                "stock_spearman": _spearman(
                    quarter["fragility_score"],
                    quarter["future_max_drawdown_63d"],
                ),
                "decile_spearman": _spearman(
                    quarter_deciles["fragility_decile"],
                    quarter_deciles["mean_realized_drawdown"],
                ),
                "top_minus_bottom_mean_drawdown": (
                    mean_drawdown.loc[10] - mean_drawdown.loc[1]
                ),
            }
        )

    return pd.DataFrame(rows)


def _portfolio_members(quarter: pd.DataFrame, portfolio: str) -> pd.Index:
    """Return the frozen constituent set for one portfolio."""
    if portfolio == "all_stocks":
        members = quarter
    elif portfolio == "exclude_most_fragile_decile":
        members = quarter.loc[quarter["fragility_decile"] != 10]
    elif portfolio == "least_fragile_decile":
        members = quarter.loc[quarter["fragility_decile"] == 1]
    elif portfolio == "most_fragile_decile":
        members = quarter.loc[quarter["fragility_decile"] == 10]
    else:
        raise ValueError(f"Unknown portfolio: {portfolio}")

    return pd.Index(members["permno"].unique())


def _portfolio_metrics(
    returns: pd.DataFrame,
    members: pd.Index,
) -> dict[str, float]:
    """Calculate buy-and-hold metrics from equal initial stock weights."""
    stock_returns = returns.reindex(columns=members, fill_value=0).fillna(0)
    stock_wealth = (1 + stock_returns).cumprod()
    portfolio_wealth = stock_wealth.mean(axis="columns")
    previous_wealth = np.concatenate(([1.0], portfolio_wealth.to_numpy()[:-1]))
    portfolio_returns = portfolio_wealth.to_numpy() / previous_wealth - 1
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], portfolio_wealth.to_numpy()))
    )[1:]
    five_day_returns = (
        pd.Series(1 + portfolio_returns).rolling(5).apply(np.prod, raw=True) - 1
    )

    return {
        "maximum_drawdown": float(
            np.max(1 - portfolio_wealth.to_numpy() / running_peak)
        ),
        "downside_volatility": downside_volatility(portfolio_returns),
        "worst_five_day_loss": float(max(0, -five_day_returns.min())),
        "cumulative_return": float(portfolio_wealth.iloc[-1] - 1),
    }


def _load_daily_returns(
    daily_path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the daily stock returns required by the test event windows."""
    connection = duckdb.connect()

    try:
        daily = connection.execute(
            """
            SELECT permno, dlycaldt, dlyret
            FROM read_parquet(?)
            WHERE dlycaldt BETWEEN ? AND ?
            ORDER BY dlycaldt, permno
            """,
            [str(daily_path), start_date, end_date],
        ).fetchdf()
    finally:
        connection.close()

    if daily.duplicated(["dlycaldt", "permno"]).any():
        raise RuntimeError("CRSP daily data contain duplicate security-date rows.")

    return daily


def _build_portfolio_results(
    panel: pd.DataFrame,
    daily_path: Path,
) -> pd.DataFrame:
    """Evaluate the four fixed portfolios in separate quarterly event windows."""
    daily = _load_daily_returns(
        daily_path,
        panel["target_window_start"].min(),
        panel["target_window_end"].max(),
    )
    rows = []

    for report_period, quarter in panel.groupby("report_period", sort=True):
        date_columns = [
            "information_date",
            "target_window_start",
            "target_window_end",
        ]

        if not quarter[date_columns].nunique().eq(1).all():
            raise RuntimeError(
                f"Inconsistent event-window dates for {report_period:%Y-%m-%d}."
            )

        information_date = quarter["information_date"].iloc[0]
        window_start = quarter["target_window_start"].iloc[0]
        window_end = quarter["target_window_end"].iloc[0]
        window = daily.loc[
            daily["dlycaldt"].between(window_start, window_end)
            & daily["permno"].isin(quarter["permno"])
        ]
        market_dates = pd.date_range(window_start, window_end, freq="B")
        observed_dates = pd.Index(window["dlycaldt"].drop_duplicates().sort_values())
        market_dates = market_dates.intersection(observed_dates)

        if len(market_dates) != 63:
            raise RuntimeError(
                f"Expected 63 market days for {report_period:%Y-%m-%d}, "
                f"found {len(market_dates)}."
            )

        returns = window.pivot(index="dlycaldt", columns="permno", values="dlyret")
        returns = returns.reindex(market_dates)

        for portfolio in _PORTFOLIOS:
            members = _portfolio_members(quarter, portfolio)

            if members.empty:
                raise RuntimeError(f"Portfolio {portfolio} has no constituents.")

            metrics = _portfolio_metrics(returns, members)
            expected_return = quarter.loc[
                quarter["permno"].isin(members),
                "future_cumulative_return_63d",
            ].mean()

            if not np.isclose(metrics["cumulative_return"], expected_return):
                raise RuntimeError(
                    f"Portfolio return does not match its constituents for "
                    f"{report_period:%Y-%m-%d}."
                )

            rows.append(
                {
                    "report_period": report_period,
                    "information_date": information_date,
                    "portfolio": portfolio,
                    "stocks": len(members),
                    "window_start": window_start,
                    "window_end": window_end,
                    "market_days": len(market_dates),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def _validate_outputs(paths: tuple[Path, Path, Path, Path]) -> tuple[int, int]:
    """Validate the completed evaluation tables and core design invariants."""
    panel = pd.read_parquet(paths[0])
    deciles = pd.read_csv(paths[1], parse_dates=["report_period", "information_date"])
    correlations = pd.read_csv(
        paths[2],
        parse_dates=["report_period", "information_date"],
    )
    portfolios = pd.read_csv(
        paths[3],
        parse_dates=[
            "report_period",
            "information_date",
            "window_start",
            "window_end",
        ],
    )
    quarters = panel["report_period"].nunique()

    if panel.empty or panel.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("Invalid Fragility Score stock-quarter output.")

    if not panel["fragility_decile"].between(1, 10).all():
        raise RuntimeError("Fragility deciles must be between 1 and 10.")

    if len(deciles) != quarters * 10:
        raise RuntimeError("Every test quarter must contain ten score deciles.")

    if len(correlations) != quarters or correlations.isna().any().any():
        raise RuntimeError("Invalid quarterly rank-correlation output.")

    if (
        len(portfolios) != quarters * len(_PORTFOLIOS)
        or set(portfolios["portfolio"]) != set(_PORTFOLIOS)
        or not portfolios["market_days"].eq(63).all()
        or portfolios.isna().any().any()
    ):
        raise RuntimeError("Invalid quarterly portfolio output.")

    bounded = portfolios[["maximum_drawdown", "worst_five_day_loss"]]

    if not ((bounded >= 0) & (bounded <= 1)).all().all():
        raise RuntimeError("Portfolio loss measures must be between zero and one.")

    if (portfolios["downside_volatility"] < 0).any():
        raise RuntimeError("Portfolio downside volatility cannot be negative.")

    return len(panel), quarters


def build_fragility_evaluation(
    predictions_path: Path,
    sample_path: Path,
    daily_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> FragilityEvaluationResult:
    """Build or validate the frozen held-out economic evaluation."""
    predictions_path = Path(predictions_path)
    sample_path = Path(sample_path)
    daily_path = Path(daily_path)
    output_directory = Path(output_directory)
    _require_columns(
        predictions_path,
        _PREDICTION_COLUMNS,
        "selected baseline predictions",
    )
    _require_columns(sample_path, _SAMPLE_COLUMNS, "modeling sample")
    _require_columns(daily_path, _DAILY_COLUMNS, "CRSP daily data")
    output_paths = _output_paths(output_directory)
    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError(
                "Economic-evaluation outputs are incomplete. "
                "Rebuild with overwrite=True."
            )

        newest_input = max(
            predictions_path.stat().st_mtime,
            sample_path.stat().st_mtime,
            daily_path.stat().st_mtime,
            Path(__file__).stat().st_mtime,
            Path(__file__).with_name("metrics.py").stat().st_mtime,
        )

        if newest_input > min(path.stat().st_mtime for path in output_paths):
            raise RuntimeError(
                "Economic-evaluation inputs or code are newer than the outputs. "
                "Rebuild with overwrite=True."
            )

        stock_quarters, quarters = _validate_outputs(output_paths)
        return FragilityEvaluationResult(
            False,
            *output_paths,
            stock_quarters,
            quarters,
        )

    _LOGGER.info("Building held-out stock-level Fragility Score results.")
    panel = _build_stock_quarters(predictions_path, sample_path)
    deciles = _build_decile_results(panel)
    correlations = _build_rank_correlations(panel, deciles)
    _LOGGER.info("Building quarterly buy-and-hold portfolio event windows.")
    portfolios = _build_portfolio_results(panel, daily_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_paths[0], index=False, compression="zstd")
    deciles.to_csv(output_paths[1], index=False)
    correlations.to_csv(output_paths[2], index=False)
    portfolios.to_csv(output_paths[3], index=False)
    stock_quarters, quarters = _validate_outputs(output_paths)

    return FragilityEvaluationResult(
        True,
        *output_paths,
        stock_quarters,
        quarters,
    )


__all__ = ["FragilityEvaluationResult", "build_fragility_evaluation"]
