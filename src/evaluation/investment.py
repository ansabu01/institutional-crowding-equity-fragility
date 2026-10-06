"""Build the continuous Fragility Score strategy and alpha tests."""

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from .metrics import downside_volatility, maximum_drawdown


_SIGNAL_COLUMNS = {
    "report_period",
    "information_date",
    "target_window_end",
    "permno",
    "fragility_decile",
}
_STOCK_COLUMNS = {"permno", "dlycaldt", "dlyret"}
_MARKET_COLUMNS = {"dlycaldt", "dlytotret"}
_FACTOR_COLUMNS = {
    "date",
    "market_excess_return",
    "smb",
    "hml",
    "rmw",
    "cma",
    "momentum",
    "risk_free_rate",
}
_DAILY_OUTPUT_COLUMNS = (
    "date",
    "report_period",
    "all_stocks_count",
    "screened_stocks_count",
    "all_stocks_return",
    "screened_return",
    "active_return",
    "crsp_market_return",
    "market_excess_return",
    "smb",
    "hml",
    "rmw",
    "cma",
    "momentum",
    "risk_free_rate",
)
_FACTOR_MODELS = {
    "capm": ["market_excess_return"],
    "ff5_momentum": [
        "market_excess_return",
        "smb",
        "hml",
        "rmw",
        "cma",
        "momentum",
    ],
}
_HAC_LAGS = 5
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvestmentEvaluationResult:
    """Paths and dimensions of the supplementary investment outputs."""

    created: bool
    daily_returns_path: Path
    performance_path: Path
    alpha_path: Path
    trading_days: int


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


def _output_paths(output_directory: Path) -> tuple[Path, Path, Path]:
    """Return the three investment-evaluation output paths."""
    return (
        output_directory / "fragility_strategy_daily_returns.parquet",
        output_directory / "fragility_strategy_performance.csv",
        output_directory / "fragility_strategy_alpha.csv",
    )


def _load_stock_returns(
    daily_stock_path: Path,
    signals: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load only the stock returns needed by the test strategy."""
    members = signals[["permno"]].drop_duplicates()
    connection = duckdb.connect()
    connection.register("_members", members)

    try:
        daily = connection.execute(
            """
            SELECT d.permno, d.dlycaldt, d.dlyret
            FROM read_parquet(?) d
            JOIN _members m USING (permno)
            WHERE d.dlycaldt BETWEEN ? AND ?
            ORDER BY d.dlycaldt, d.permno
            """,
            [str(daily_stock_path), start_date, end_date],
        ).fetchdf()
    finally:
        connection.close()

    if daily.duplicated(["dlycaldt", "permno"]).any():
        raise RuntimeError("CRSP stock data contain duplicate security-date rows.")

    return daily


def _portfolio_returns(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    members: pd.Index,
) -> np.ndarray:
    """Return one equal-initial-weight buy-and-hold portfolio path."""
    window = daily.loc[daily["dlycaldt"].isin(dates) & daily["permno"].isin(members)]
    returns = window.pivot(index="dlycaldt", columns="permno", values="dlyret")
    returns = returns.reindex(index=dates, columns=members).fillna(0)
    stock_wealth = (1 + returns).cumprod()
    portfolio_wealth = stock_wealth.mean(axis="columns").to_numpy()
    previous_wealth = np.concatenate(([1.0], portfolio_wealth[:-1]))

    return portfolio_wealth / previous_wealth - 1


def _build_daily_returns(
    signals_path: Path,
    daily_stock_path: Path,
    market_index_path: Path,
    factors_path: Path,
) -> pd.DataFrame:
    """Construct continuous strategy, benchmark, and factor returns."""
    signals = pd.read_parquet(signals_path, columns=list(_SIGNAL_COLUMNS))
    market = pd.read_parquet(market_index_path).rename(
        columns={"dlycaldt": "date", "dlytotret": "crsp_market_return"}
    )
    factors = pd.read_parquet(factors_path)

    if signals.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("Fragility Score signals contain duplicate stock-quarters.")

    if market["date"].duplicated().any() or factors["date"].duplicated().any():
        raise RuntimeError("Market-index and factor dates must be unique.")

    calendar = pd.DatetimeIndex(market["date"].sort_values())
    quarters = list(signals.groupby("report_period", sort=True))
    rebalance_dates = []

    for report_period, quarter in quarters:
        information_dates = quarter["information_date"].drop_duplicates()

        if len(information_dates) != 1:
            raise RuntimeError(
                f"Inconsistent information date for {report_period:%Y-%m-%d}."
            )

        future_dates = calendar[calendar > information_dates.iloc[0]]

        if future_dates.empty:
            raise RuntimeError(f"No rebalance date for {report_period:%Y-%m-%d}.")

        rebalance_dates.append(future_dates[0])

    final_end_dates = quarters[-1][1]["target_window_end"].drop_duplicates()

    if len(final_end_dates) != 1:
        raise RuntimeError("The final strategy window has inconsistent end dates.")

    final_date = final_end_dates.iloc[0]
    daily_stock = _load_stock_returns(
        daily_stock_path,
        signals,
        rebalance_dates[0],
        final_date,
    )
    rows = []

    for index, (report_period, quarter) in enumerate(quarters):
        start_date = rebalance_dates[index]
        end_date = (
            calendar[calendar < rebalance_dates[index + 1]][-1]
            if index + 1 < len(rebalance_dates)
            else final_date
        )
        dates = calendar[(calendar >= start_date) & (calendar <= end_date)]
        all_members = pd.Index(quarter["permno"].unique())
        screened_members = pd.Index(
            quarter.loc[quarter["fragility_decile"] != 10, "permno"].unique()
        )

        if dates.empty or screened_members.empty:
            raise RuntimeError(f"Empty strategy period for {report_period:%Y-%m-%d}.")

        period = pd.DataFrame(
            {
                "date": dates,
                "report_period": report_period,
                "all_stocks_count": len(all_members),
                "screened_stocks_count": len(screened_members),
                "all_stocks_return": _portfolio_returns(
                    daily_stock,
                    dates,
                    all_members,
                ),
                "screened_return": _portfolio_returns(
                    daily_stock,
                    dates,
                    screened_members,
                ),
            }
        )
        rows.append(period)

    strategy = pd.concat(rows, ignore_index=True)
    strategy["active_return"] = (
        strategy["screened_return"] - strategy["all_stocks_return"]
    )
    strategy = strategy.merge(
        market[["date", "crsp_market_return"]],
        on="date",
        how="left",
        validate="one_to_one",
    ).merge(
        factors,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if strategy[list(_DAILY_OUTPUT_COLUMNS)].isna().any().any():
        raise RuntimeError(
            "Continuous strategy dates do not match market-factor dates."
        )

    expected_dates = calendar[
        (calendar >= strategy["date"].min()) & (calendar <= strategy["date"].max())
    ]

    if (
        not strategy["date"]
        .reset_index(drop=True)
        .equals(pd.Series(expected_dates, name="date"))
    ):
        raise RuntimeError("Continuous strategy dates contain a gap or overlap.")

    return strategy[list(_DAILY_OUTPUT_COLUMNS)]


def _build_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple gross performance statistics."""
    series = {
        "all_stocks": "all_stocks_return",
        "exclude_most_fragile_decile": "screened_return",
        "crsp_total_market": "crsp_market_return",
    }
    rows = []

    for strategy, column in series.items():
        returns = daily[column]
        excess_returns = returns - daily["risk_free_rate"]
        total_growth = float((1 + returns).prod())
        rows.append(
            {
                "strategy": strategy,
                "trading_days": len(returns),
                "cumulative_return": total_growth - 1,
                "annualized_return": total_growth ** (252 / len(returns)) - 1,
                "annualized_volatility": returns.std(ddof=1) * np.sqrt(252),
                "sharpe_ratio": (
                    np.sqrt(252) * excess_returns.mean() / returns.std(ddof=1)
                ),
                "maximum_drawdown": maximum_drawdown(returns),
                "downside_volatility": downside_volatility(returns),
            }
        )

    return pd.DataFrame(rows)


def _build_alpha_tests(daily: pd.DataFrame) -> pd.DataFrame:
    """Estimate CAPM and FF5-plus-momentum alpha with HAC inference."""
    dependent_series = {
        "all_stocks": daily["all_stocks_return"] - daily["risk_free_rate"],
        "exclude_most_fragile_decile": (
            daily["screened_return"] - daily["risk_free_rate"]
        ),
        "screened_minus_all_stocks": daily["active_return"],
    }
    rows = []

    for portfolio, dependent in dependent_series.items():
        for model_name, factors in _FACTOR_MODELS.items():
            regressors = sm.add_constant(daily[factors], has_constant="add")
            model = sm.OLS(dependent, regressors).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": _HAC_LAGS},
            )
            daily_alpha = float(model.params["const"])
            rows.append(
                {
                    "portfolio": portfolio,
                    "model": model_name,
                    "trading_days": len(daily),
                    "daily_alpha": daily_alpha,
                    "annualized_alpha": 252 * daily_alpha,
                    "alpha_standard_error": float(model.bse["const"]),
                    "alpha_t_statistic": float(model.tvalues["const"]),
                    "alpha_p_value": float(model.pvalues["const"]),
                    "adjusted_r_squared": float(model.rsquared_adj),
                    "market_beta": float(model.params["market_excess_return"]),
                }
            )

    return pd.DataFrame(rows)


def _validate_outputs(paths: tuple[Path, Path, Path]) -> int:
    """Validate the completed investment-evaluation outputs."""
    daily = pd.read_parquet(paths[0])
    performance = pd.read_csv(paths[1])
    alpha = pd.read_csv(paths[2])

    if (
        tuple(daily.columns) != _DAILY_OUTPUT_COLUMNS
        or daily.empty
        or daily["date"].duplicated().any()
        or daily.isna().any().any()
    ):
        raise RuntimeError("Invalid continuous strategy-return output.")

    if (
        set(performance["strategy"])
        != {
            "all_stocks",
            "exclude_most_fragile_decile",
            "crsp_total_market",
        }
        or performance.isna().any().any()
    ):
        raise RuntimeError("Invalid strategy-performance output.")

    if (
        len(alpha) != 6
        or set(alpha["model"]) != set(_FACTOR_MODELS)
        or alpha.isna().any().any()
    ):
        raise RuntimeError("Invalid strategy-alpha output.")

    if (daily[["all_stocks_return", "screened_return"]] < -1).any().any():
        raise RuntimeError("A strategy daily return is below minus one.")

    return len(daily)


def build_investment_evaluation(
    signals_path: Path,
    daily_stock_path: Path,
    market_index_path: Path,
    factors_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> InvestmentEvaluationResult:
    """Build or validate the supplementary strategy and alpha evaluation."""
    signals_path = Path(signals_path)
    daily_stock_path = Path(daily_stock_path)
    market_index_path = Path(market_index_path)
    factors_path = Path(factors_path)
    output_directory = Path(output_directory)
    _require_columns(signals_path, _SIGNAL_COLUMNS, "Fragility Score signals")
    _require_columns(daily_stock_path, _STOCK_COLUMNS, "CRSP daily stock data")
    _require_columns(market_index_path, _MARKET_COLUMNS, "CRSP market index")
    _require_columns(factors_path, _FACTOR_COLUMNS, "daily factor data")
    output_paths = _output_paths(output_directory)
    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError(
                "Investment-evaluation outputs are incomplete. "
                "Rebuild with overwrite=True."
            )

        newest_input = max(
            path.stat().st_mtime
            for path in (
                signals_path,
                daily_stock_path,
                market_index_path,
                factors_path,
                Path(__file__),
                Path(__file__).with_name("metrics.py"),
            )
        )

        if newest_input > min(path.stat().st_mtime for path in output_paths):
            raise RuntimeError(
                "Investment-evaluation inputs or code are newer than the "
                "outputs. Rebuild with overwrite=True."
            )

        trading_days = _validate_outputs(output_paths)
        return InvestmentEvaluationResult(False, *output_paths, trading_days)

    _LOGGER.info("Building continuous Fragility Score strategy returns.")
    daily = _build_daily_returns(
        signals_path,
        daily_stock_path,
        market_index_path,
        factors_path,
    )
    performance = _build_performance(daily)
    _LOGGER.info("Estimating CAPM and FF5-plus-momentum alpha.")
    alpha = _build_alpha_tests(daily)
    output_directory.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_paths[0], index=False, compression="zstd")
    performance.to_csv(output_paths[1], index=False)
    alpha.to_csv(output_paths[2], index=False)
    trading_days = _validate_outputs(output_paths)

    return InvestmentEvaluationResult(True, *output_paths, trading_days)


__all__ = ["InvestmentEvaluationResult", "build_investment_evaluation"]
