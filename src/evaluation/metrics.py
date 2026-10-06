"""Performance metrics shared by the economic evaluations."""

import numpy as np
import pandas as pd


def maximum_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Return maximum drawdown as a positive magnitude."""
    values = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1 + values)
    running_peak = np.maximum.accumulate(
        np.concatenate(([1.0], wealth))
    )[1:]
    return float(np.max(1 - wealth / running_peak))


def downside_volatility(
    returns: pd.Series | np.ndarray,
    *,
    annualization: int = 252,
) -> float:
    """Return annualized downside volatility around a zero threshold."""
    values = np.asarray(returns, dtype=float)
    return float(
        np.sqrt(annualization * np.mean(np.minimum(values, 0) ** 2))
    )


__all__ = ["downside_volatility", "maximum_drawdown"]
