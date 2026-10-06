"""Build the investment robustness checks."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.evaluation.robustness import build_investment_robustness  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional robustness rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build investment robustness results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing investment-robustness outputs.",
    )
    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build and summarize the investment robustness checks."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    paths = get_paths_config()
    result = build_investment_robustness(
        signals_path=paths.tables / "evaluation" / "fragility_stock_quarters.parquet",
        sample_path=paths.processed / "modeling" / "modeling_sample.parquet",
        daily_stock_path=paths.raw / "crsp" / "daily_stock.parquet",
        primary_daily_path=(
            paths.tables / "evaluation" / "fragility_strategy_daily_returns.parquet"
        ),
        output_directory=paths.tables / "robustness",
        overwrite=overwrite,
    )
    performance = pd.read_csv(result.performance_path)
    costs = pd.read_csv(result.costs_path)
    alpha = pd.read_csv(result.alpha_path)
    factor_performance = pd.read_csv(result.factor_performance_path)
    factor_alpha = pd.read_csv(result.factor_alpha_path)
    daily = pd.read_parquet(
        result.daily_returns_path,
        columns=[
            "defensive_fragility_factor_return",
            "defensive_fragility_factor_20_return",
            "fragility_factor_exposure",
        ],
    )

    print()
    print(
        f"Investment robustness: "
        f"{'created' if result.created else 'already present'}"
    )
    print(f"Trading days: {result.trading_days:,}")
    print("Gross strategy performance")
    print(
        performance[
            ["strategy", "annualized_return", "sharpe_ratio", "maximum_drawdown"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print("Transaction-cost sensitivity")
    print(
        costs[
            ["cost_bps", "strategy", "annualized_return", "maximum_drawdown"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print("Active FF5+momentum alpha HAC sensitivity")
    print(
        alpha[
            ["hac_lag", "annualized_alpha", "alpha_t_statistic", "alpha_p_value"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print("Fragility-factor breadth")
    for label, column in {
        "DFRAG-10 (D1 minus D10)": "defensive_fragility_factor_return",
        "DFRAG-20 (D1-D2 minus D9-D10)": (
            "defensive_fragility_factor_20_return"
        ),
    }.items():
        returns = daily[column]
        print(
            f"{label}: annualized mean {252 * returns.mean():.4f}, "
            f"annualized volatility {252**0.5 * returns.std():.4f}"
        )
    print()
    print("Fragility-factor specifications and risk management")
    print(
        factor_performance[
            [
                "strategy",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
                "maximum_drawdown",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print("Fragility-factor FF5+momentum alpha")
    print(
        factor_alpha[
            [
                "strategy",
                "annualized_alpha",
                "alpha_t_statistic",
                "alpha_p_value",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(
        "Volatility-managed exposure: "
        f"mean {daily['fragility_factor_exposure'].mean():.3f}, "
        f"minimum {daily['fragility_factor_exposure'].min():.3f}"
    )
    print("Investment-robustness outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
