"""Build the investment robustness checks."""

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
    "permno",
    "fragility_decile",
}
_SAMPLE_COLUMNS = {
    "report_period",
    "permno",
    "sample_split",
    "market_cap_usd",
}
_STOCK_COLUMNS = {"permno", "dlycaldt", "dlyret"}
_PRIMARY_DAILY_COLUMNS = {
    "date",
    "report_period",
    "all_stocks_return",
    "screened_return",
    "active_return",
    "market_excess_return",
    "smb",
    "hml",
    "rmw",
    "cma",
    "momentum",
    "risk_free_rate",
}
_VARIANTS = {
    "equal_all": ("equal", 10),
    "equal_exclude_d10": ("equal", 9),
    "equal_exclude_top20": ("equal", 8),
    "value_all": ("value", 10),
    "value_exclude_d10": ("value", 9),
}
_COST_BPS = (10, 25, 50)
_HAC_LAGS = (5, 21, 63)
_FACTORS = ["market_excess_return", "smb", "hml", "rmw", "cma", "momentum"]
_DEFENSIVE_FACTOR_COLUMNS = {
    "least_fragile_return",
    "most_fragile_return",
    "defensive_fragility_factor_return",
    "least_fragile_20_return",
    "most_fragile_20_return",
    "defensive_fragility_factor_20_return",
    "half_exposure_fragility_factor_return",
    "volatility_managed_fragility_factor_return",
    "fragility_factor_exposure",
    "volatility_managed_one_way_turnover",
}
_FACTOR_VARIANTS = {
    "original": "defensive_fragility_factor_return",
    "dfrag_20": "defensive_fragility_factor_20_return",
    "half_exposure": "half_exposure_fragility_factor_return",
    "volatility_managed": "volatility_managed_fragility_factor_return",
}
_FACTOR_PORTFOLIOS = {
    "dfrag_10": (
        1,
        1,
        10,
        10,
        "least_fragile_return",
        "most_fragile_return",
        "defensive_fragility_factor_return",
    ),
    "dfrag_20": (
        1,
        2,
        9,
        10,
        "least_fragile_20_return",
        "most_fragile_20_return",
        "defensive_fragility_factor_20_return",
    ),
}
_COST_STRATEGIES = {
    "equal_all": "equal_all_return",
    "equal_exclude_d10": "equal_exclude_d10_return",
    "dfrag_10": "defensive_fragility_factor_return",
    "dfrag_20": "defensive_fragility_factor_20_return",
    "volatility_managed": "volatility_managed_fragility_factor_return",
}
_HALF_EXPOSURE = 0.5
_VOLATILITY_TARGET = 0.15
_VOLATILITY_WINDOW = 63
_FACTOR_HAC_LAG = 21
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InvestmentRobustnessResult:
    """Paths and dimensions of the investment-robustness outputs."""

    created: bool
    daily_returns_path: Path
    performance_path: Path
    turnover_path: Path
    costs_path: Path
    alpha_path: Path
    factor_performance_path: Path
    factor_alpha_path: Path
    trading_days: int


def _require_columns(path: Path, columns: set[str], description: str) -> None:
    """Require a readable Parquet file with the expected columns."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")

    try:
        available = set(pq.ParquetFile(path).schema_arrow.names)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid {description}: {path}") from error

    missing = columns - available

    if missing:
        raise RuntimeError(f"Missing columns in {description}: {sorted(missing)}")


def _load_inputs(
    signals_path: Path,
    sample_path: Path,
    daily_stock_path: Path,
    primary_daily_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the test signals, calendar, and required stock returns."""
    signals = pd.read_parquet(signals_path, columns=list(_SIGNAL_COLUMNS))
    sample = pd.read_parquet(sample_path, columns=list(_SAMPLE_COLUMNS))
    sample = sample.loc[sample["sample_split"] == "test"].drop(
        columns="sample_split"
    )
    primary_daily = pd.read_parquet(
        primary_daily_path,
        columns=list(_PRIMARY_DAILY_COLUMNS),
    ).sort_values("date").reset_index(drop=True)
    signals = signals.merge(
        sample,
        on=["report_period", "permno"],
        how="inner",
        validate="one_to_one",
    )

    if signals.duplicated(["report_period", "permno"]).any():
        raise RuntimeError("Robustness signals contain duplicate stock-quarters.")

    if len(signals) != len(sample) or primary_daily["date"].duplicated().any():
        raise RuntimeError("Robustness inputs do not match the frozen test sample.")

    if signals["market_cap_usd"].isna().any() or (
        signals["market_cap_usd"] <= 0
    ).any():
        raise RuntimeError("Value-weighted portfolios require positive market caps.")

    members = signals[["permno"]].drop_duplicates()
    connection = duckdb.connect()
    connection.register("_members", members)

    try:
        daily_stock = connection.execute(
            """
            SELECT d.permno, d.dlycaldt, d.dlyret
            FROM read_parquet(?) d
            JOIN _members m USING (permno)
            WHERE d.dlycaldt BETWEEN ? AND ?
            ORDER BY d.dlycaldt, d.permno
            """,
            [
                str(daily_stock_path),
                primary_daily["date"].min(),
                primary_daily["date"].max(),
            ],
        ).fetchdf()
    finally:
        connection.close()

    if daily_stock.duplicated(["dlycaldt", "permno"]).any():
        raise RuntimeError("CRSP stock data contain duplicate security-date rows.")

    return signals, primary_daily, daily_stock


def _target_weights(
    quarter: pd.DataFrame,
    weighting: str,
    maximum_decile: int,
    minimum_decile: int = 1,
) -> pd.Series:
    """Return one portfolio's formation weights."""
    selected = quarter.loc[
        quarter["fragility_decile"].between(minimum_decile, maximum_decile)
    ]

    if selected.empty:
        raise RuntimeError("A robustness portfolio has no constituent stocks.")

    if weighting == "equal":
        weights = pd.Series(1.0, index=selected["permno"])
    else:
        weights = selected.set_index("permno")["market_cap_usd"].astype(float)

    return weights / weights.sum()


def _one_way_turnover(
    previous_weights: pd.Series | None,
    target_weights: pd.Series,
) -> float:
    """Calculate one-way turnover at a portfolio formation date."""
    if previous_weights is None:
        return 1.0

    securities = previous_weights.index.union(target_weights.index)
    return float(
        0.5
        * (
            target_weights.reindex(securities, fill_value=0)
            - previous_weights.reindex(securities, fill_value=0)
        )
        .abs()
        .sum()
    )


def _portfolio_path(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[np.ndarray, pd.Series]:
    """Return buy-and-hold daily returns and weights at period end."""
    stock_wealth = (1 + returns[weights.index]).cumprod().mul(weights, axis="columns")
    portfolio_wealth = stock_wealth.sum(axis="columns")
    previous_wealth = np.concatenate(([1.0], portfolio_wealth.to_numpy()[:-1]))
    portfolio_returns = portfolio_wealth.to_numpy() / previous_wealth - 1
    ending_weights = stock_wealth.iloc[-1] / portfolio_wealth.iloc[-1]

    return portfolio_returns, ending_weights


def _build_portfolios(
    signals: pd.DataFrame,
    primary_daily: pd.DataFrame,
    daily_stock: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all gross robustness portfolios and their turnover."""
    daily_rows = []
    turnover_rows = []
    previous_weights: dict[str, pd.Series | None] = {
        variant: None for variant in _VARIANTS
    }
    previous_factor_weights: dict[str, pd.Series | None] = {
        f"{factor}_{leg}": None
        for factor in _FACTOR_PORTFOLIOS
        for leg in ("long", "short")
    }

    for report_period, calendar in primary_daily.groupby("report_period", sort=True):
        quarter = signals.loc[signals["report_period"] == report_period]
        dates = pd.DatetimeIndex(calendar["date"])
        members = pd.Index(quarter["permno"])
        window = daily_stock.loc[
            daily_stock["dlycaldt"].isin(dates)
            & daily_stock["permno"].isin(members)
        ]
        returns = window.pivot(
            index="dlycaldt",
            columns="permno",
            values="dlyret",
        ).reindex(index=dates, columns=members).fillna(0)
        period = pd.DataFrame({"date": dates, "report_period": report_period})

        for variant, (weighting, maximum_decile) in _VARIANTS.items():
            target_weights = _target_weights(
                quarter,
                weighting,
                maximum_decile,
            )
            turnover = _one_way_turnover(previous_weights[variant], target_weights)
            portfolio_returns, ending_weights = _portfolio_path(
                returns,
                target_weights,
            )
            period[f"{variant}_return"] = portfolio_returns
            previous_weights[variant] = ending_weights
            turnover_rows.append(
                {
                    "report_period": report_period,
                    "strategy": variant,
                    "stocks": len(target_weights),
                    "one_way_turnover": turnover,
                }
            )

        for factor, specification in _FACTOR_PORTFOLIOS.items():
            (
                long_minimum,
                long_maximum,
                short_minimum,
                short_maximum,
                long_column,
                short_column,
                factor_column,
            ) = specification
            long_weights = _target_weights(
                quarter,
                "equal",
                long_maximum,
                minimum_decile=long_minimum,
            )
            short_weights = _target_weights(
                quarter,
                "equal",
                short_maximum,
                minimum_decile=short_minimum,
            )
            long_return, long_ending_weights = _portfolio_path(
                returns,
                long_weights,
            )
            short_return, short_ending_weights = _portfolio_path(
                returns,
                short_weights,
            )
            turnover = _one_way_turnover(
                previous_factor_weights[f"{factor}_long"],
                long_weights,
            ) + _one_way_turnover(
                previous_factor_weights[f"{factor}_short"],
                short_weights,
            )
            previous_factor_weights[f"{factor}_long"] = long_ending_weights
            previous_factor_weights[f"{factor}_short"] = short_ending_weights
            period[long_column] = long_return
            period[short_column] = short_return
            period[factor_column] = long_return - short_return
            turnover_rows.append(
                {
                    "report_period": report_period,
                    "strategy": factor,
                    "stocks": len(long_weights) + len(short_weights),
                    "one_way_turnover": turnover,
                }
            )

        daily_rows.append(period)

    return (
        pd.concat(daily_rows, ignore_index=True),
        pd.DataFrame(turnover_rows),
    )


def _performance_row(
    strategy: str,
    returns: pd.Series,
    risk_free_rate: pd.Series,
) -> dict[str, object]:
    """Return standard performance statistics for one daily series."""
    growth = float((1 + returns).prod())
    volatility = float(returns.std(ddof=1))
    return {
        "strategy": strategy,
        "trading_days": len(returns),
        "cumulative_return": growth - 1,
        "annualized_return": growth ** (252 / len(returns)) - 1,
        "annualized_volatility": volatility * np.sqrt(252),
        "sharpe_ratio": np.sqrt(252)
        * float((returns - risk_free_rate).mean())
        / volatility,
        "maximum_drawdown": maximum_drawdown(returns),
        "downside_volatility": downside_volatility(returns),
    }


def _build_performance(
    daily: pd.DataFrame,
    risk_free_rate: pd.Series,
) -> pd.DataFrame:
    """Calculate gross performance for all portfolio variants."""
    return pd.DataFrame(
        [
            _performance_row(
                variant,
                daily[f"{variant}_return"],
                risk_free_rate,
            )
            for variant in _VARIANTS
        ]
    )


def _add_factor_risk_management(
    daily: pd.DataFrame,
    turnover: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add fixed and lagged-volatility exposure variants of DFRAG."""
    factor_return = daily["defensive_fragility_factor_return"]
    trailing_volatility = (
        factor_return.rolling(_VOLATILITY_WINDOW).std(ddof=1).shift(1)
        * np.sqrt(252)
    )
    exposure = (_VOLATILITY_TARGET / trailing_volatility).clip(upper=1.0)
    exposure = exposure.fillna(1.0)

    daily["half_exposure_fragility_factor_return"] = (
        _HALF_EXPOSURE * factor_return
    )
    daily["fragility_factor_exposure"] = exposure
    daily["volatility_managed_fragility_factor_return"] = (
        exposure * factor_return
    )

    base_rows = turnover.loc[turnover["strategy"].eq("dfrag_10")].set_index(
        "report_period"
    )
    base_turnover = base_rows["one_way_turnover"]
    formation_day = ~daily["report_period"].duplicated()
    formation_turnover = daily["report_period"].map(base_turnover).where(
        formation_day,
        0.0,
    )
    exposure_turnover = 2 * exposure.diff().abs().fillna(0.0)
    daily["volatility_managed_one_way_turnover"] = (
        exposure * formation_turnover + exposure_turnover
    )
    managed_turnover = (
        daily.groupby("report_period", as_index=False)[
            "volatility_managed_one_way_turnover"
        ]
        .sum()
        .rename(
            columns={
                "volatility_managed_one_way_turnover": "one_way_turnover"
            }
        )
    )
    managed_turnover["strategy"] = "volatility_managed"
    managed_turnover["stocks"] = managed_turnover["report_period"].map(
        base_rows["stocks"]
    )
    turnover = pd.concat(
        [
            turnover,
            managed_turnover[
                ["report_period", "strategy", "stocks", "one_way_turnover"]
            ],
        ],
        ignore_index=True,
    )

    return daily, turnover


def _build_factor_performance(daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate performance for the factor specifications and variants."""
    zero_rate = pd.Series(0.0, index=daily.index)

    return pd.DataFrame(
        [
            _performance_row(strategy, daily[column], zero_rate)
            for strategy, column in _FACTOR_VARIANTS.items()
        ]
    )


def _build_factor_alpha(
    daily: pd.DataFrame,
    primary_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate FF5-plus-momentum alpha for each factor specification."""
    regressors = sm.add_constant(primary_daily[_FACTORS], has_constant="add")
    rows = []

    for strategy, column in _FACTOR_VARIANTS.items():
        model = sm.OLS(daily[column], regressors).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": _FACTOR_HAC_LAG},
        )
        rows.append(
            {
                "strategy": strategy,
                "annualized_alpha": 252 * float(model.params["const"]),
                "alpha_standard_error": float(model.bse["const"]),
                "alpha_t_statistic": float(model.tvalues["const"]),
                "alpha_p_value": float(model.pvalues["const"]),
                "adjusted_r_squared": float(model.rsquared_adj),
            }
        )

    return pd.DataFrame(rows)


def _build_cost_results(
    daily: pd.DataFrame,
    turnover: pd.DataFrame,
    risk_free_rate: pd.Series,
) -> pd.DataFrame:
    """Apply fixed transaction-cost scenarios to portfolios and factors."""
    rows = []

    for cost_bps in _COST_BPS:
        cost_rate = cost_bps / 10_000

        for strategy, return_column in _COST_STRATEGIES.items():
            if strategy == "volatility_managed":
                net_return = (
                    daily[return_column]
                    - cost_rate * daily["volatility_managed_one_way_turnover"]
                )
                row = _performance_row(
                    strategy,
                    net_return,
                    pd.Series(0.0, index=net_return.index),
                )
                row["cost_bps"] = cost_bps
                rows.append(row)
                continue

            costs = turnover.loc[
                turnover["strategy"] == strategy,
                ["report_period", "one_way_turnover"],
            ].copy()
            costs["formation_cost"] = cost_rate * costs["one_way_turnover"]
            series = daily[["date", "report_period", return_column]].merge(
                costs[["report_period", "formation_cost"]],
                on="report_period",
                how="left",
                validate="many_to_one",
            )
            first_day = ~series["report_period"].duplicated()
            applied_cost = series["formation_cost"].where(first_day, 0)
            if strategy.startswith("dfrag"):
                net_return = series[return_column] - applied_cost
                benchmark_rate = pd.Series(0.0, index=net_return.index)
            else:
                net_return = (1 + series[return_column]) * (1 - applied_cost) - 1
                benchmark_rate = risk_free_rate
            row = _performance_row(strategy, net_return, benchmark_rate)
            row["cost_bps"] = cost_bps
            rows.append(row)

    return pd.DataFrame(rows)[
        [
            "cost_bps",
            "strategy",
            "trading_days",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "maximum_drawdown",
            "downside_volatility",
        ]
    ]


def _build_alpha_robustness(primary_daily: pd.DataFrame) -> pd.DataFrame:
    """Re-estimate active FF5-plus-momentum alpha at fixed HAC lags."""
    regressors = sm.add_constant(primary_daily[_FACTORS], has_constant="add")
    rows = []

    for hac_lag in _HAC_LAGS:
        model = sm.OLS(primary_daily["active_return"], regressors).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lag},
        )
        daily_alpha = float(model.params["const"])
        rows.append(
            {
                "model": "ff5_momentum",
                "hac_lag": hac_lag,
                "trading_days": len(primary_daily),
                "daily_alpha": daily_alpha,
                "annualized_alpha": 252 * daily_alpha,
                "alpha_standard_error": float(model.bse["const"]),
                "alpha_t_statistic": float(model.tvalues["const"]),
                "alpha_p_value": float(model.pvalues["const"]),
                "adjusted_r_squared": float(model.rsquared_adj),
            }
        )

    return pd.DataFrame(rows)


def _output_paths(
    output_directory: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Return the investment-robustness output paths."""
    return (
        output_directory / "investment_robustness_daily_returns.parquet",
        output_directory / "investment_robustness_performance.csv",
        output_directory / "investment_robustness_turnover.csv",
        output_directory / "investment_robustness_costs.csv",
        output_directory / "investment_robustness_alpha_hac.csv",
        output_directory / "fragility_factor_risk_management_performance.csv",
        output_directory / "fragility_factor_risk_management_alpha.csv",
    )


def _validate_outputs(
    paths: tuple[Path, Path, Path, Path, Path, Path, Path],
) -> int:
    """Validate completed investment-robustness outputs."""
    daily = pd.read_parquet(paths[0])
    performance = pd.read_csv(paths[1])
    turnover = pd.read_csv(paths[2])
    costs = pd.read_csv(paths[3])
    alpha = pd.read_csv(paths[4])
    factor_performance = pd.read_csv(paths[5])
    factor_alpha = pd.read_csv(paths[6])
    report_periods = daily["report_period"].nunique()

    if daily.empty or daily["date"].duplicated().any() or daily.isna().any().any():
        raise RuntimeError("Invalid investment-robustness daily returns.")

    if not _DEFENSIVE_FACTOR_COLUMNS.issubset(daily.columns) or not np.allclose(
        daily["defensive_fragility_factor_return"],
        daily["least_fragile_return"] - daily["most_fragile_return"],
    ):
        raise RuntimeError("Invalid fragility-factor returns.")

    if not np.allclose(
        daily["defensive_fragility_factor_20_return"],
        daily["least_fragile_20_return"] - daily["most_fragile_20_return"],
    ):
        raise RuntimeError("Invalid 20-percent fragility-factor returns.")

    factor_return = daily["defensive_fragility_factor_return"]
    exposure = daily["fragility_factor_exposure"]

    if (
        not exposure.between(0, 1).all()
        or not exposure.iloc[:_VOLATILITY_WINDOW].eq(1).all()
        or not np.allclose(
            daily["half_exposure_fragility_factor_return"],
            _HALF_EXPOSURE * factor_return,
        )
        or not np.allclose(
            daily["volatility_managed_fragility_factor_return"],
            exposure * factor_return,
        )
    ):
        raise RuntimeError("Invalid fragility-factor risk management.")

    if (
        set(performance["strategy"]) != set(_VARIANTS)
        or len(performance) != len(_VARIANTS)
    ):
        raise RuntimeError("Invalid investment-robustness performance output.")

    portfolio_turnover = turnover.loc[turnover["strategy"].isin(_VARIANTS)]
    factor_turnover = turnover.loc[turnover["strategy"].isin(_FACTOR_PORTFOLIOS)]
    managed_turnover = turnover.loc[
        turnover["strategy"].eq("volatility_managed")
    ]

    if (
        len(turnover)
        != report_periods * (len(_VARIANTS) + len(_FACTOR_PORTFOLIOS) + 1)
        or set(turnover["strategy"])
        != set(_VARIANTS) | set(_FACTOR_PORTFOLIOS) | {"volatility_managed"}
        or not portfolio_turnover["one_way_turnover"].between(0, 1).all()
        or not factor_turnover["one_way_turnover"].between(0, 2).all()
        or not managed_turnover["one_way_turnover"].ge(0).all()
    ):
        raise RuntimeError("Invalid investment-robustness turnover output.")

    if (
        len(costs) != len(_COST_STRATEGIES) * len(_COST_BPS)
        or set(costs["strategy"]) != set(_COST_STRATEGIES)
        or set(costs["cost_bps"]) != set(_COST_BPS)
    ):
        raise RuntimeError("Invalid transaction-cost robustness output.")

    if len(alpha) != len(_HAC_LAGS) or set(alpha["hac_lag"]) != set(_HAC_LAGS):
        raise RuntimeError("Invalid HAC-lag robustness output.")

    expected_factor_strategies = set(_FACTOR_VARIANTS)

    if (
        len(factor_performance) != len(_FACTOR_VARIANTS)
        or set(factor_performance["strategy"]) != expected_factor_strategies
        or len(factor_alpha) != len(_FACTOR_VARIANTS)
        or set(factor_alpha["strategy"]) != expected_factor_strategies
    ):
        raise RuntimeError("Invalid fragility-factor risk-management outputs.")

    numeric_outputs = (
        performance,
        turnover,
        costs,
        alpha,
        factor_performance,
        factor_alpha,
    )

    if any(frame.isna().any().any() for frame in numeric_outputs):
        raise RuntimeError("An investment-robustness output contains missing values.")

    return len(daily)


def build_investment_robustness(
    signals_path: Path,
    sample_path: Path,
    daily_stock_path: Path,
    primary_daily_path: Path,
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> InvestmentRobustnessResult:
    """Build or validate the investment robustness checks."""
    signals_path = Path(signals_path)
    sample_path = Path(sample_path)
    daily_stock_path = Path(daily_stock_path)
    primary_daily_path = Path(primary_daily_path)
    output_directory = Path(output_directory)
    _require_columns(signals_path, _SIGNAL_COLUMNS, "Fragility Score signals")
    _require_columns(sample_path, _SAMPLE_COLUMNS, "modeling sample")
    _require_columns(daily_stock_path, _STOCK_COLUMNS, "CRSP daily stock data")
    _require_columns(
        primary_daily_path,
        _PRIMARY_DAILY_COLUMNS,
        "primary strategy returns",
    )
    output_paths = _output_paths(output_directory)
    existing = [path for path in output_paths if path.exists()]

    if existing and not overwrite:
        if len(existing) != len(output_paths):
            raise RuntimeError(
                "Investment-robustness outputs are incomplete. "
                "Rebuild with overwrite=True."
            )

        newest_input = max(
            path.stat().st_mtime
            for path in (
                signals_path,
                sample_path,
                daily_stock_path,
                primary_daily_path,
                Path(__file__),
                Path(__file__).with_name("metrics.py"),
            )
        )

        if newest_input > min(path.stat().st_mtime for path in output_paths):
            raise RuntimeError(
                "Investment-robustness inputs or code are newer than the "
                "outputs. Rebuild with overwrite=True."
            )

        trading_days = _validate_outputs(output_paths)
        return InvestmentRobustnessResult(False, *output_paths, trading_days)

    _LOGGER.info("Building alternative weighting and screening portfolios.")
    signals, primary_daily, daily_stock = _load_inputs(
        signals_path,
        sample_path,
        daily_stock_path,
        primary_daily_path,
    )
    daily, turnover = _build_portfolios(signals, primary_daily, daily_stock)
    daily, turnover = _add_factor_risk_management(daily, turnover)

    if not daily["date"].equals(primary_daily["date"].reset_index(drop=True)):
        raise RuntimeError("Robustness returns do not match the primary calendar.")

    risk_free_rate = primary_daily["risk_free_rate"].reset_index(drop=True)
    performance = _build_performance(daily, risk_free_rate)
    costs = _build_cost_results(daily, turnover, risk_free_rate)
    _LOGGER.info("Re-estimating active alpha with alternative HAC lags.")
    alpha = _build_alpha_robustness(primary_daily)
    factor_performance = _build_factor_performance(daily)
    factor_alpha = _build_factor_alpha(daily, primary_daily)
    output_directory.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_paths[0], index=False, compression="zstd")
    performance.to_csv(output_paths[1], index=False)
    turnover.to_csv(output_paths[2], index=False)
    costs.to_csv(output_paths[3], index=False)
    alpha.to_csv(output_paths[4], index=False)
    factor_performance.to_csv(output_paths[5], index=False)
    factor_alpha.to_csv(output_paths[6], index=False)
    trading_days = _validate_outputs(output_paths)

    return InvestmentRobustnessResult(
        True,
        *output_paths,
        trading_days,
    )


__all__ = ["InvestmentRobustnessResult", "build_investment_robustness"]
