"""Factor model construction and a no-look-ahead backtest on HIP-3 perps.

The pipeline is deliberately split so the timing rules are easy to verify:

1. :func:`build_base` turns the underlying-equity panels into the long Polars frames
   ``toraniko`` expects and computes the three style factors with the library's own
   ``factor_mom`` / ``factor_sze`` / ``factor_val``.
2. :func:`estimate_factors` runs ``estimate_factor_returns`` and reconstructs residuals
   as ``epsilon = winsorize(r) - B @ f`` (an identity that holds exactly for the model),
   which also sidesteps the library's residual-frame truncation when there are >100 dates.
3. :func:`run_backtest` walks forward one trading day at a time. At each date it uses only
   information available then (signals are lagged inside ``factor_mom``; covariance and
   specific risk use history up to *t*), forms a market/sector-neutral mean-variance book
   over names already listed on HIP-3, and realises the perp return from *t* to *t+1*.

Nothing from the warm-up source is realised as PnL — only the HIP-3 perp returns are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import polars as pl

from toraniko.math import winsorize
from toraniko.model import estimate_factor_returns
from toraniko.styles import factor_mom, factor_sze, factor_val

STYLE_COLS = ("mom_score", "sze_score", "val_score")


def _to_long(wide: pd.DataFrame, value_name: str) -> pl.DataFrame:
    """Convert a ``dates x tickers`` panel into a tidy ``date | symbol | <value>`` frame."""
    long = (
        wide.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="symbol", value_name=value_name)
        .dropna(subset=[value_name])
    )
    return pl.from_pandas(long).with_columns(pl.col("date").cast(pl.Date))


def _finite(col: str) -> pl.Expr:
    return pl.col(col).is_finite() & pl.col(col).is_not_null() & pl.col(col).is_not_nan()


@dataclass
class FactorInputs:
    """Assembled, cleaned inputs ready for ``estimate_factor_returns``."""

    base: pl.DataFrame
    sector_names: list[str]


def build_base(
    close: pd.DataFrame,
    market_cap: pd.DataFrame,
    book_price: pd.DataFrame,
    sales_price: pd.DataFrame,
    cf_price: pd.DataFrame,
    sectors: dict[str, str],
    *,
    mom_trailing_days: int = 120,
    mom_lag: int = 20,
    mom_winsor: float = 0.01,
    min_names_per_day: int = 12,
) -> FactorInputs:
    """Build the per-(date, symbol) modelling frame from underlying-equity panels.

    Style scores are produced by the ``toraniko`` library functions unchanged; the only
    additions are dropping non-finite rows (the library assumes pre-cleaned data) and
    discarding cross-sections too small to support a stable regression.
    """
    universe = [t for t in close.columns if t in sectors]
    close, market_cap = close[universe], market_cap[universe]

    returns = _to_long(close.pct_change(), "asset_returns").filter(_finite("asset_returns"))
    caps = _to_long(market_cap, "market_cap").filter(_finite("market_cap") & (pl.col("market_cap") > 0))
    # Outer-join the value components: they have different histories (TTM sales/cash flow start
    # much later than book value), and factor_val averages over whichever ratios are present.
    value = (
        _to_long(book_price[universe], "book_price")
        .join(_to_long(sales_price[universe], "sales_price"), on=["date", "symbol"], how="full", coalesce=True)
        .join(_to_long(cf_price[universe], "cf_price"), on=["date", "symbol"], how="full", coalesce=True)
    )

    mom = factor_mom(returns, trailing_days=mom_trailing_days, lag=mom_lag, winsor_factor=mom_winsor).collect()
    sze = factor_sze(caps).collect()
    val = factor_val(value).collect()
    style = (
        mom.filter(_finite("mom_score"))
        .select("date", "symbol", "mom_score")
        .join(sze.filter(_finite("sze_score")).select("date", "symbol", "sze_score"), on=["date", "symbol"])
        .join(val.filter(_finite("val_score")).select("date", "symbol", "val_score"), on=["date", "symbol"])
    )

    sector_names = sorted(set(sectors.values()))
    onehot = pl.from_pandas(
        pd.get_dummies(pd.Series(sectors, name="sector")).astype("int64").rename_axis("symbol").reset_index()
    )
    sector_frame = returns.select("date", "symbol").unique().join(onehot, on="symbol")

    base = (
        returns.join(caps, on=["date", "symbol"])
        .join(style, on=["date", "symbol"])
        .join(sector_frame, on=["date", "symbol"])
        .drop_nulls()
    )
    keep = base.group_by("date").len().filter(pl.col("len") >= min_names_per_day).get_column("date")
    base = base.filter(pl.col("date").is_in(keep.implode()))
    return FactorInputs(base, sector_names)


@dataclass
class FactorModel:
    """Estimated factor returns and reconstructed residuals, all sorted by date."""

    factor_returns: pd.DataFrame  # date-indexed: market + sectors + styles
    residuals: pd.DataFrame  # dates x symbols (idiosyncratic returns)
    base: pd.DataFrame  # the modelling frame as pandas, sorted by date
    factor_cols: list[str] = field(default_factory=list)


def estimate_factors(
    inputs: FactorInputs, *, winsor_factor: float = 0.05, residualize_styles: bool = True
) -> FactorModel:
    """Run the library's factor-return estimation and reconstruct clean residuals.

    ``residualize_styles=True`` (the library default) orthogonalises the style factors to the
    market and sector factors, so style returns are net of market/sector exposure.

    Residuals are rebuilt as ``winsorize(r, winsor_factor) - B @ f`` rather than read from
    the returned frame; the two are identical for this model, and the reconstruction is
    immune to the library's >100-date residual truncation and covers every symbol.
    """
    b = inputs.base
    fac, _ = estimate_factor_returns(
        b.select("date", "symbol", "asset_returns"),
        b.select("date", "symbol", "market_cap"),
        b.select(["date", "symbol"] + inputs.sector_names),
        b.select("date", "symbol", *STYLE_COLS),
        winsor_factor=winsor_factor,
        residualize_styles=residualize_styles,
    )
    fac = fac.sort("date").to_pandas().set_index("date")  # guard the library's unsorted output
    fac.index = pd.to_datetime(fac.index)
    factor_cols = list(fac.columns)

    base_pd = b.to_pandas()
    base_pd["date"] = pd.to_datetime(base_pd["date"])
    base_pd = base_pd.sort_values("date")

    resid_rows = []
    for date, g in base_pd.groupby("date"):
        if date not in fac.index:
            continue
        exposures = _exposure_matrix(g, factor_cols, inputs.sector_names)
        fitted = exposures @ fac.loc[date].to_numpy()
        eps = winsorize(g["asset_returns"].to_numpy(), winsor_factor) - fitted
        resid_rows.append(pd.DataFrame({"date": date, "symbol": g["symbol"].to_numpy(), "resid": eps}))
    residuals = pd.concat(resid_rows).pivot(index="date", columns="symbol", values="resid").sort_index()
    return FactorModel(fac, residuals, base_pd, factor_cols)


def _exposure_matrix(g: pd.DataFrame, factor_cols: list[str], sector_names: list[str]) -> np.ndarray:
    """Asset exposure matrix B with columns in the model's factor order.

    Market exposure is 1 for every asset; sector exposures are the one-hot dummies; style
    exposures are the standardized scores. This reproduces ``r = B @ f + eps`` exactly.
    """
    columns = []
    for c in factor_cols:
        if c == "market":
            columns.append(np.ones(len(g)))
        else:  # sector dummy or style score, both present as columns of g
            columns.append(g[c].to_numpy(dtype="float64"))
    return np.column_stack(columns)


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def neutral_mv_weights(
    alpha: np.ndarray, exposures: np.ndarray, cov: np.ndarray, n_neutral: int, gross: float = 2.0
) -> np.ndarray:
    """Mean-variance optimal active weights, neutralised to the first ``n_neutral`` factors.

    Solves ``max alpha'w - (lambda/2) w'Σw`` s.t. ``Cᵀw = 0`` where ``C`` are the market and
    sector exposure columns; the closed form is ``w ∝ Σ⁻¹α`` projected onto ``Cᵀw = 0``.
    Weights are scaled to a fixed gross book (``gross`` = $1 long + $1 short by default).
    """
    c = exposures[:, :n_neutral]
    cov_inv = np.linalg.pinv(cov)
    raw = cov_inv @ alpha
    projected = raw - cov_inv @ c @ np.linalg.pinv(c.T @ cov_inv @ c) @ (c.T @ raw)
    total = np.abs(projected).sum()
    return projected / total * gross if total > 1e-12 else projected


@dataclass
class BacktestConfig:
    cov_window: int = 252
    min_specific_obs: int = 10
    min_tradable: int = 8
    gross: float = 2.0
    ridge: float = 1e-8


@dataclass
class BacktestResult:
    dates: pd.DatetimeIndex
    mvo: np.ndarray  # full risk-model PnL
    momentum: np.ndarray  # naive tercile long-short PnL
    turnover: np.ndarray  # daily one-sided turnover of the MVO book


def target_weights(
    model: FactorModel,
    sector_names: list[str],
    date: pd.Timestamp,
    tradable: list[str],
    config: BacktestConfig = BacktestConfig(),
) -> pd.Series:
    """Market/sector-neutral mean-variance target weights for ``tradable`` names on ``date``.

    This is the single source of truth for book construction, shared by the backtest and the
    daily report. Weights are scaled to a fixed gross book (``config.gross``); an empty Series
    is returned when fewer than ``config.min_tradable`` names are available.
    """
    cross = model.base[model.base["date"] == date]
    cross = cross[cross["symbol"].isin(tradable)].sort_values("symbol")
    if len(cross) < config.min_tradable:
        return pd.Series(dtype="float64")
    syms = cross["symbol"].to_numpy()
    exposures = _exposure_matrix(cross, model.factor_cols, sector_names)
    alpha = sum(_zscore(cross[c].to_numpy()) for c in STYLE_COLS)
    cov = _asset_cov(exposures, model.factor_returns, model.factor_cols, model.residuals, syms, date, config)
    weights = neutral_mv_weights(alpha, exposures, cov, 1 + len(sector_names), config.gross)
    return pd.Series(weights, index=syms)


def run_backtest(
    model: FactorModel,
    sector_names: list[str],
    perp_returns: pd.DataFrame,
    listing_dates: dict[str, pd.Timestamp],
    config: BacktestConfig = BacktestConfig(),
) -> BacktestResult:
    """Walk-forward backtest. PnL is realised only on HIP-3 perps, only after listing.

    Parameters
    ----------
    model: estimated factors/residuals/base from :func:`estimate_factors`
    sector_names: sector factor names (for the neutrality constraint width)
    perp_returns: ``dates x symbols`` HIP-3 perp returns aligned to the trading calendar
    listing_dates: first HIP-3 listing date per symbol (untradeable before it)
    """
    fac, base = model.factor_returns, model.base
    trade_dates = [d for d in sorted(base["date"].unique()) if d in fac.index]

    mvo, mom, turnover, realised = [], [], [], []
    prev = pd.Series(dtype="float64")
    for i in range(len(trade_dates) - 1):
        today, nxt = trade_dates[i], trade_dates[i + 1]
        cross = base[base["date"] == today]
        tradable = [
            s
            for s in cross["symbol"]
            if s in listing_dates
            and pd.notna(listing_dates[s])
            and listing_dates[s] <= today
            and s in perp_returns.columns
            and pd.notna(perp_returns.at[today, s])  # priced at t (no peek at t+1)
        ]
        if len(tradable) < config.min_tradable or len(fac.loc[:today]) < config.cov_window // 5:
            prev = pd.Series(dtype="float64")
            continue
        weights = target_weights(model, sector_names, today, tradable, config)
        syms = weights.index.to_numpy()
        cross = cross[cross["symbol"].isin(syms)].sort_values("symbol")

        fwd = perp_returns.loc[nxt]  # realised t -> t+1; missing next-day price counts as 0 return
        mvo.append(float((weights * fwd.reindex(syms).fillna(0)).sum()))
        mom.append(_naive_momentum_pnl(cross.set_index("symbol")["mom_score"], fwd))
        union = prev.index.union(weights.index)
        turnover.append(float((weights.reindex(union).fillna(0) - prev.reindex(union).fillna(0)).abs().sum()))
        prev = weights
        realised.append(nxt)

    return BacktestResult(pd.DatetimeIndex(realised), np.array(mvo), np.array(mom), np.array(turnover))


def _asset_cov(
    exposures: np.ndarray,
    fac: pd.DataFrame,
    factor_cols: list[str],
    residuals: pd.DataFrame,
    syms: np.ndarray,
    today: pd.Timestamp,
    config: BacktestConfig,
) -> np.ndarray:
    """Risk-model asset covariance ``Σ = B Σ_f Bᵀ + diag(specific var)`` using history up to ``today``."""
    fac_hist = fac.loc[:today].iloc[-config.cov_window :]
    factor_cov = np.cov(fac_hist[factor_cols].to_numpy().T)
    specific = []
    for s in syms:
        hist = residuals[s].loc[:today].dropna() if s in residuals.columns else pd.Series(dtype="float64")
        specific.append(hist.iloc[-config.cov_window :].var() if len(hist) >= config.min_specific_obs else np.nan)
    specific = np.array(specific)
    finite = np.isfinite(specific)
    specific[~finite] = np.median(specific[finite]) if finite.any() else 1e-4
    return exposures @ factor_cov @ exposures.T + np.diag(specific) + config.ridge * np.eye(len(syms))


def _naive_momentum_pnl(scores: pd.Series, fwd: pd.Series) -> float:
    """Equal-weight, dollar-neutral top/bottom-tercile momentum book PnL for one day."""
    k = max(1, len(scores) // 3)
    longs = scores.nlargest(k).index
    shorts = scores.drop(longs).nsmallest(k).index  # exclude longs so legs are always disjoint
    weights = pd.Series(0.0, index=scores.index)
    weights[longs] = 1.0 / len(longs)
    weights[shorts] = -1.0 / len(shorts)
    return float((weights * fwd.reindex(scores.index).fillna(0)).sum())


def summary_stats(returns: np.ndarray) -> dict[str, float]:
    """Annualised performance statistics for a daily return series (252 trading days)."""
    n = len(returns)
    sd = returns.std(ddof=1)
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "total_pct": (wealth[-1] - 1.0) * 100,
        "ann_pct": (wealth[-1] ** (252 / n) - 1.0) * 100,
        "vol_pct": sd * np.sqrt(252) * 100,
        "sharpe": returns.mean() / sd * np.sqrt(252),
        "t_stat": returns.mean() / (sd / np.sqrt(n)),
        "max_drawdown_pct": drawdown.min() * 100,
        "hit_rate_pct": (returns > 0).mean() * 100,
    }
