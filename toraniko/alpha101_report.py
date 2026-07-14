"""PnL and information-coefficient analysis for Alpha101 scores."""

from __future__ import annotations

from pathlib import Path

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


def latest_alpha101_weights(
    scores_df: pl.DataFrame | pl.LazyFrame,
    quantile: float = 0.2,
) -> pl.DataFrame:
    """Return the latest available top/bottom portfolio weights for every alpha."""
    if not 0 < quantile <= 0.5:
        raise ValueError("`quantile` must be in (0, 0.5]")
    scores = scores_df.collect() if isinstance(scores_df, pl.LazyFrame) else scores_df
    if not isinstance(scores, pl.DataFrame):
        raise TypeError("`scores_df` must be a Polars DataFrame or LazyFrame")
    alpha_cols = sorted(name for name in scores.columns if name.startswith("alpha") and name[5:].isdigit())
    if not alpha_cols:
        raise ValueError("`scores_df` must contain alphaNNN columns")

    symbols = scores["symbol"].unique().sort().to_list()
    groups = scores.sort("date", "symbol").partition_by("date", maintain_order=True)
    rows: list[dict[str, object]] = []
    for alpha in alpha_cols:
        for cross in reversed(groups):
            signal = cross[alpha].to_numpy()
            valid = np.isfinite(signal)
            if valid.sum() < 5:
                continue
            valid_symbols = np.asarray(cross["symbol"].to_list(), dtype=object)[valid]
            values = signal[valid]
            tail = max(1, int(np.floor(len(values) * quantile)))
            order = np.argsort(values, kind="mergesort")
            weights = {symbol: 0.0 for symbol in symbols}
            for index in order[-tail:]:
                weights[valid_symbols[index]] = 0.5 / tail
            for index in order[:tail]:
                weights[valid_symbols[index]] = -0.5 / tail
            date = cross["date"][0]
            rows.extend(
                {"date": date, "alpha": alpha, "symbol": symbol, "weight": weight}
                for symbol, weight in weights.items()
            )
            break
    return pl.DataFrame(rows)


def plot_alpha101_pnl(
    summary: pl.DataFrame,
    daily: pl.DataFrame,
    output: str | Path,
    *,
    highlight: int = 8,
) -> None:
    """Plot cumulative PnL and cross-alpha diagnostics for all 101 portfolios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    pnl = daily.pivot(on="alpha", index="date", values="pnl").sort("date")
    alpha_cols = sorted(name for name in pnl.columns if name.startswith("alpha"))
    values = pnl.select(alpha_cols).to_numpy()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    wealth = np.cumprod(1 + filled, axis=0) - 1
    active = finite.sum(axis=1)
    composite_returns = np.divide(
        np.where(finite, values, 0.0).sum(axis=1),
        active,
        out=np.zeros(len(values), dtype=float),
        where=active > 0,
    )
    composite = np.cumprod(1 + composite_returns) - 1
    composite_wealth = composite + 1
    drawdown = composite_wealth / np.maximum.accumulate(composite_wealth) - 1
    dates = pnl["date"].to_list()
    highlighted = summary.sort("sharpe", descending=True).head(highlight)["alpha"].to_list()

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], hspace=0.32, wspace=0.25)
    ax = fig.add_subplot(grid[0, :])
    for column in range(len(alpha_cols)):
        ax.plot(dates, wealth[:, column], color="0.72", alpha=0.22, linewidth=0.55)
    colors = plt.colormaps["tab10"]
    for index, alpha in enumerate(highlighted):
        column = alpha_cols.index(alpha)
        ax.plot(dates, wealth[:, column], color=colors(index), linewidth=1.25, label=alpha)
    ax.plot(dates, composite, color="black", linewidth=2.2, label="equal-weight Alpha101")
    ax.axhline(0, color="0.3", linewidth=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Alpha101 next-day long-short cumulative PnL (101 gray paths; top Sharpe highlighted)")
    ax.set_ylabel("cumulative return")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.legend(ncol=3, fontsize=8, loc="upper left")

    ax = fig.add_subplot(grid[1, 0])
    ranked = summary.sort("sharpe")
    selected = pl.concat([ranked.head(10), ranked.tail(10)]).sort("sharpe")
    bar_colors = np.where(selected["sharpe"].to_numpy() >= 0, "#2ca02c", "#d62728")
    ax.barh(selected["alpha"].to_list(), selected["sharpe"].to_numpy(), color=bar_colors)
    ax.axvline(0, color="0.3", linewidth=0.6)
    ax.set_title("Bottom and top decile by Sharpe")
    ax.set_xlabel("annualized Sharpe (before costs)")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)

    ax = fig.add_subplot(grid[1, 1])
    ax.fill_between(dates, drawdown, 0, color="#d62728", alpha=0.55)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Equal-weight Alpha101 composite drawdown")
    ax.set_ylabel("drawdown")
    ax.grid(True, color="0.9", linewidth=0.5)

    fig.suptitle("Toraniko WorldQuant Alpha101 PnL analysis", fontsize=14, fontweight="bold")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_alpha101_weights(weights: pl.DataFrame, output: str | Path) -> None:
    """Plot the latest symbol weights for all Alpha101 long-short portfolios."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    latest = weights.sort("alpha", "symbol")
    matrix = latest.pivot(on="symbol", index="alpha", values="weight").sort("alpha")
    symbols = [column for column in matrix.columns if column != "alpha"]
    values = matrix.select(symbols).to_numpy()
    limit = np.max(np.abs(values)) if values.size else 1.0
    composite = values.mean(axis=0)

    fig = plt.figure(figsize=(18, 15))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.3], wspace=0.2)
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(np.arange(len(symbols)), labels=symbols, rotation=60, ha="right", fontsize=8)
    rows = np.arange(0, matrix.height, 5)
    ax.set_yticks(rows, labels=[matrix["alpha"][int(row)] for row in rows], fontsize=7)
    ax.set_xlabel("symbol")
    ax.set_ylabel("alpha (every fifth label shown)")
    ax.set_title("Latest Alpha101 portfolio weights: short red, long green")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax = fig.add_subplot(grid[0, 1])
    order = np.argsort(composite)
    colors = np.where(composite[order] >= 0, "#2ca02c", "#d62728")
    ax.barh(np.asarray(symbols)[order], composite[order], color=colors)
    ax.axvline(0, color="0.3", linewidth=0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_title("Mean weight across 101 alphas")
    ax.set_xlabel("portfolio weight")
    ax.grid(True, axis="x", color="0.9", linewidth=0.5)

    dates = latest["date"].unique().sort().to_list()
    latest_date = max(dates) if dates else None
    date_label = latest_date.strftime("%Y-%m-%d") if hasattr(latest_date, "strftime") else str(latest_date or "")
    date_text = f"latest signals through {date_label}" if date_label else ""
    fig.suptitle(f"Toraniko WorldQuant Alpha101 weights — {date_text}", fontsize=14, fontweight="bold")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
