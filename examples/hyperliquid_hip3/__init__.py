"""Example: run the toraniko factor model on Hyperliquid HIP-3 (trader.xyz) equity perps.

See ``README.md`` for the design, the no-look-ahead timing rules, and the data caveats.
"""

from .backtest import (
    BacktestConfig,
    BacktestResult,
    FactorInputs,
    FactorModel,
    build_base,
    estimate_factors,
    neutral_mv_weights,
    run_backtest,
    summary_stats,
    target_weights,
)
from .data import HyperliquidHIP3, UnderlyingData, YahooUnderlying, load_spy

__all__ = [
    "HyperliquidHIP3",
    "YahooUnderlying",
    "UnderlyingData",
    "load_spy",
    "FactorInputs",
    "FactorModel",
    "BacktestConfig",
    "BacktestResult",
    "build_base",
    "estimate_factors",
    "run_backtest",
    "target_weights",
    "neutral_mv_weights",
    "summary_stats",
]
