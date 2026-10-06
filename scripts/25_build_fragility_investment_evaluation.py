"""Build the supplementary Fragility Score investment evaluation."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from src.evaluation.investment import build_investment_evaluation  # noqa: E402
from src.utils.configuration import get_paths_config  # noqa: E402


def _parse_arguments() -> argparse.Namespace:
    """Parse the optional investment-evaluation rebuild flag."""
    parser = argparse.ArgumentParser(
        description="Build the Fragility Score strategy and alpha evaluation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild and replace existing investment-evaluation outputs.",
    )

    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Build and summarize the supplementary investment evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    paths = get_paths_config()
    result = build_investment_evaluation(
        signals_path=(paths.tables / "evaluation" / "fragility_stock_quarters.parquet"),
        daily_stock_path=paths.raw / "crsp" / "daily_stock.parquet",
        market_index_path=paths.raw / "crsp" / "daily_market_index.parquet",
        factors_path=paths.interim / "fama_french" / "daily_factors.parquet",
        output_directory=paths.tables / "evaluation",
        overwrite=overwrite,
    )
    status = "created" if result.created else "already present"
    performance = pd.read_csv(result.performance_path)
    alpha = pd.read_csv(result.alpha_path)
    active_alpha = alpha.loc[
        (alpha["portfolio"] == "screened_minus_all_stocks")
        & (alpha["model"] == "ff5_momentum")
    ].iloc[0]

    print()
    print(f"Investment evaluation: {status}")
    print(f"Trading days: {result.trading_days:,}")
    print("Performance")
    print(
        performance[
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
    print(
        "Active FF5+momentum alpha: "
        f"{active_alpha['annualized_alpha']:.4f} "
        f"(p={active_alpha['alpha_p_value']:.4f})"
    )
    print("Investment-evaluation outputs: PASS")


if __name__ == "__main__":
    arguments = _parse_arguments()
    main(overwrite=arguments.overwrite)
