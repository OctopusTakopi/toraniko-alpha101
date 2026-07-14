"""PnL and information-coefficient analysis for Alpha101 scores."""

from __future__ import annotations

import numpy as np
import polars as pl

from toraniko.alpha101 import _rank_1d


def analyze_alpha101(
    scores_df: pl.DataFrame | pl.LazyFrame,
    returns_df: pl.DataFrame | pl.LazyFrame,
    quantile: float = 0.2,
    annualization: int = 252,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Backtest each alpha as an equal-weight next-day long-short portfolio.

    Scores at date ``t`` are matched to each symbol's return at ``t + 1``.
    The top and bottom ``quantile`` of the cross-section are held long and
    short respectively.  The function returns an alpha summary and daily
    diagnostics (PnL, IC, rank IC, coverage and one-sided turnover).
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("`quantile` must be in (0, 0.5]")
    scores = scores_df.collect() if isinstance(scores_df, pl.LazyFrame) else scores_df
    returns = returns_df.collect() if isinstance(returns_df, pl.LazyFrame) else returns_df
    if not isinstance(scores, pl.DataFrame) or not isinstance(returns, pl.DataFrame):
        raise TypeError("scores and returns must be Polars DataFrames or LazyFrames")
    alpha_cols = sorted(name for name in scores.columns if name.startswith("alpha") and name[5:].isdigit())
    if not alpha_cols:
        raise ValueError("`scores_df` must contain alphaNNN columns")
    if not {"date", "symbol", "asset_returns"} <= set(returns.columns):
        raise ValueError("`returns_df` must contain date, symbol and asset_returns")

    forward = (
        returns.sort("symbol", "date")
        .with_columns(pl.col("asset_returns").shift(-1).over("symbol").alias("forward_return"))
        .select("date", "symbol", "forward_return")
    )
    joined = scores.select("date", "symbol", *alpha_cols).join(forward, on=["date", "symbol"]).sort("date", "symbol")
    date_groups = joined.partition_by("date", maintain_order=True)
    symbols = joined["symbol"].unique().sort().to_list()
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    daily_rows: list[dict[str, object]] = []

    for alpha in alpha_cols:
        previous = np.zeros(len(symbols), dtype=float)
        for date_group in date_groups:
            date = date_group["date"][0]
            cross = date_group.select("symbol", alpha, "forward_return")
            signal = cross[alpha].to_numpy()
            future = cross["forward_return"].to_numpy()
            valid = np.isfinite(signal) & np.isfinite(future)
            if valid.sum() < 5:
                continue
            s, r = signal[valid], future[valid]
            percentile = _rank_1d(s)
            tail = max(1, int(np.floor(len(s) * quantile)))
            order = np.argsort(s, kind="mergesort")
            short_idx, long_idx = order[:tail], order[-tail:]
            weights = np.zeros(len(symbols), dtype=float)
            valid_symbols = np.asarray(cross["symbol"].to_list(), dtype=object)[valid]
            for index in long_idx:
                weights[symbol_index[valid_symbols[index]]] = 0.5 / tail
            for index in short_idx:
                weights[symbol_index[valid_symbols[index]]] = -0.5 / tail
            turnover = 0.5 * np.abs(weights - previous).sum()
            previous = weights
            pnl = 0.5 * (r[long_idx].mean() - r[short_idx].mean())
            ic = _corr(s, r)
            rank_ic = _corr(percentile, _rank_1d(r))
            daily_rows.append(
                {
                    "date": date,
                    "alpha": alpha,
                    "pnl": pnl,
                    "ic": ic,
                    "rank_ic": rank_ic,
                    "turnover": turnover,
                    "coverage": float(valid.mean()),
                }
            )

    daily = pl.DataFrame(daily_rows)
    summaries = [_summarize(group, annualization) for group in daily.partition_by("alpha", maintain_order=True)]
    summary = pl.DataFrame(summaries).sort("alpha")
    return summary, daily


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    x_scale, y_scale = np.max(np.abs(x)), np.max(np.abs(y))
    if x_scale > 0:
        x = x / x_scale
    if y_scale > 0:
        y = y / y_scale
    denominator = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def _summarize(group: pl.DataFrame, annualization: int) -> dict[str, object]:
    pnl = group["pnl"].to_numpy()
    ic = group["ic"].to_numpy()
    rank_ic = group["rank_ic"].to_numpy()
    pnl_std = np.nanstd(pnl, ddof=1)
    wealth = np.cumprod(1 + pnl)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    return {
        "alpha": group["alpha"][0],
        "days": len(group),
        "annual_return": float(np.nanmean(pnl) * annualization),
        "annual_volatility": float(pnl_std * np.sqrt(annualization)),
        "sharpe": float(np.nanmean(pnl) / pnl_std * np.sqrt(annualization)) if pnl_std > 0 else np.nan,
        "max_drawdown": float(np.nanmin(drawdown)),
        "mean_ic": float(np.nanmean(ic)),
        "ic_ir": _information_ratio(ic, annualization),
        "mean_rank_ic": float(np.nanmean(rank_ic)),
        "rank_ic_ir": _information_ratio(rank_ic, annualization),
        "mean_turnover": float(group["turnover"].mean()),
        "mean_coverage": float(group["coverage"].mean()),
    }


def _information_ratio(values: np.ndarray, annualization: int) -> float:
    std = np.nanstd(values, ddof=1)
    return float(np.nanmean(values) / std * np.sqrt(annualization)) if std > 0 else np.nan


def render_alpha101_report(
    summary: pl.DataFrame,
    *,
    title: str = "WorldQuant Alpha101 PnL and Alpha Analysis",
    dataset_note: str = "",
    rows: int = 20,
) -> str:
    """Render a compact Markdown report from :func:`analyze_alpha101`."""
    ranked = summary.sort("sharpe", descending=True, nulls_last=True)
    lines = [f"# {title}", "", dataset_note, ""] if dataset_note else [f"# {title}", ""]
    lines.extend(
        [
            "Method: signals at *t*, next-session returns at *t+1*, equal-weight top/bottom quintiles, "
            "50% long and 50% short. Returns exclude costs.",
            "",
            f"Alphas analyzed: {summary.height}",
            "",
            "## Highest Sharpe alphas",
            "",
            _markdown_table(ranked.head(rows)),
            "",
            "## Lowest Sharpe alphas",
            "",
            _markdown_table(ranked.tail(rows).sort("sharpe")),
            "",
            "## Cross-alpha diagnostics",
            "",
            f"- Median annual return: {summary['annual_return'].median():.2%}",
            f"- Median Sharpe: {summary['sharpe'].median():.3f}",
            f"- Median rank IC: {summary['mean_rank_ic'].median():.4f}",
            f"- Median one-sided turnover: {summary['mean_turnover'].median():.2%}",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pl.DataFrame) -> str:
    columns = ["alpha", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "mean_rank_ic", "mean_turnover"]
    header = "| Alpha | Ann. return | Ann. vol | Sharpe | Max DD | Rank IC | Turnover |"
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for values in frame.select(columns).iter_rows():
        alpha, annual_return, annual_volatility, sharpe, max_drawdown, rank_ic, turnover = values
        rows.append(
            f"| {alpha} | {annual_return:.2%} | {annual_volatility:.2%} | {sharpe:.3f} | "
            f"{max_drawdown:.2%} | {rank_ic:.4f} | {turnover:.2%} |"
        )
    return "\n".join(rows)
