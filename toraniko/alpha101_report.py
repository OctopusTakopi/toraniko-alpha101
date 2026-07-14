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
    prices_df: pl.DataFrame | pl.LazyFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Backtest each alpha as an equal-weight next-day long-short portfolio.

    Scores at date ``t`` are matched to each symbol's return at ``t + 1``.
    The top and bottom ``quantile`` of the cross-section are held long and
    short respectively.  The function returns an alpha summary and daily
    diagnostics (PnL, IC, rank IC, coverage and one-sided turnover).  When
    ``prices_df`` supplies execution close prices, the analysis also reconstructs
    gross dollar volume and shares traded for the paper's turnover and
    cents-per-share definitions, assuming one dollar of gross investment.
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
    has_prices = prices_df is not None
    if prices_df is not None:
        prices = prices_df.collect() if isinstance(prices_df, pl.LazyFrame) else prices_df
        if not isinstance(prices, pl.DataFrame):
            raise TypeError("prices_df must be a Polars DataFrame or LazyFrame")
        if not {"date", "symbol", "close"} <= set(prices.columns):
            raise ValueError("prices_df must contain date, symbol and close")
        joined = joined.join(
            prices.select("date", "symbol", pl.col("close").cast(pl.Float64)),
            on=["date", "symbol"],
            how="left",
        )
    date_groups = joined.partition_by("date", maintain_order=True)
    symbols = joined["symbol"].unique().sort().to_list()
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    daily_rows: list[dict[str, object]] = []

    for alpha in alpha_cols:
        previous = np.zeros(len(symbols), dtype=float)
        previous_shares = np.zeros(len(symbols), dtype=float)
        for date_group in date_groups:
            date = date_group["date"][0]
            columns = ["symbol", alpha, "forward_return"]
            if has_prices:
                columns.append("close")
            cross = date_group.select(columns)
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
            row: dict[str, object] = {
                "date": date,
                "alpha": alpha,
                "pnl": pnl,
                "ic": ic,
                "rank_ic": rank_ic,
                "turnover": turnover,
                "coverage": float(valid.mean()),
            }
            if has_prices:
                price_vector = np.full(len(symbols), np.nan, dtype=float)
                for symbol, price in zip(cross["symbol"].to_list(), cross["close"].to_numpy()):
                    price_vector[symbol_index[symbol]] = price
                tradable = np.isfinite(price_vector) & (price_vector > 0)
                required = (weights != 0) | (previous_shares != 0)
                if np.all(tradable[required]):
                    target_shares = np.divide(
                        weights,
                        price_vector,
                        out=np.zeros_like(weights),
                        where=tradable,
                    )
                    shares_traded = np.abs(target_shares - previous_shares)
                    row["shares_traded"] = float(shares_traded.sum())
                    row["dollar_volume"] = float(np.dot(shares_traded, np.where(tradable, price_vector, 0.0)))
                    previous_shares = target_shares
                else:
                    row["shares_traded"] = np.nan
                    row["dollar_volume"] = np.nan
            daily_rows.append(row)

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


def analyze_alpha101_paper(
    summary: pl.DataFrame,
    daily: pl.DataFrame,
    annualization: int = 252,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the alpha-level, pairwise and regression data used by the paper's figures."""
    required = {
        "alpha",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "paper_turnover",
        "cents_per_share",
    }
    if not required <= set(summary.columns):
        missing = ", ".join(sorted(required - set(summary.columns)))
        raise ValueError(f"summary is missing paper metrics: {missing}")
    if not {"date", "alpha", "pnl"} <= set(daily.columns):
        raise ValueError("daily must contain date, alpha and pnl")

    metrics = (
        summary.select(
            "alpha",
            "sharpe",
            pl.col("paper_turnover").alias("turnover"),
            "cents_per_share",
            (pl.col("annual_volatility") / np.sqrt(annualization)).alias("daily_volatility"),
            (pl.col("annual_return") / annualization).alias("daily_return"),
            "annual_return",
        )
        .sort("alpha")
        .with_columns(pl.col("turnover").log().alias("log_turnover"))
        .with_columns(
            (pl.col("log_turnover") - pl.col("log_turnover").mean()).alias("centered_log_turnover")
        )
    )
    alpha_names = metrics["alpha"].to_list()
    turnover_loading = dict(zip(alpha_names, metrics["centered_log_turnover"].to_list()))
    pnl = daily.pivot(on="alpha", index="date", values="pnl").sort("date")
    pair_rows: list[dict[str, object]] = []
    for i, alpha_i in enumerate(alpha_names):
        left = pnl[alpha_i].to_numpy()
        for alpha_j in alpha_names[:i]:
            right = pnl[alpha_j].to_numpy()
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.sum() < 3:
                continue
            correlation = _corr(left[finite], right[finite])
            if not np.isfinite(correlation):
                continue
            loading_i, loading_j = turnover_loading[alpha_i], turnover_loading[alpha_j]
            pair_rows.append(
                {
                    "alpha_i": alpha_i,
                    "alpha_j": alpha_j,
                    "observations": int(finite.sum()),
                    "correlation": correlation,
                    "turnover_sum": loading_i + loading_j,
                    "turnover_product": loading_i * loading_j,
                }
            )
    pairs = pl.DataFrame(pair_rows)
    correlations = pairs["correlation"].to_numpy()
    pair_fit = _fit_ols(
        correlations,
        {
            "turnover_sum": pairs["turnover_sum"].to_numpy(),
            "turnover_product": pairs["turnover_product"].to_numpy(),
        },
    )
    pair_contribution = (
        pair_fit["coefficients"][1] * pairs["turnover_sum"].to_numpy()
        + pair_fit["coefficients"][2] * pairs["turnover_product"].to_numpy()
    )
    pairs = pairs.with_columns(
        pl.Series("demeaned_correlation", correlations - np.mean(correlations)),
        pl.Series("turnover_model_contribution", pair_contribution),
    )

    daily_return = metrics["daily_return"].to_numpy()
    daily_volatility = metrics["daily_volatility"].to_numpy()
    turnover = metrics["turnover"].to_numpy()
    positive_return = (daily_return > 0) & (daily_volatility > 0) & (turnover > 0)
    positive_volatility = (daily_volatility > 0) & (turnover > 0)
    regressions: list[dict[str, object]] = []
    regressions.extend(
        _regression_rows(
            "log_return_on_log_volatility",
            np.log(daily_return[positive_return]),
            {"log_volatility": np.log(daily_volatility[positive_return])},
        )
    )
    regressions.extend(
        _regression_rows(
            "log_return_on_log_volatility_and_turnover",
            np.log(daily_return[positive_return]),
            {
                "log_volatility": np.log(daily_volatility[positive_return]),
                "log_turnover": np.log(turnover[positive_return]),
            },
        )
    )
    regressions.extend(
        _regression_rows(
            "correlation_on_turnover_tensors",
            correlations,
            {
                "turnover_sum": pairs["turnover_sum"].to_numpy(),
                "turnover_product": pairs["turnover_product"].to_numpy(),
            },
        )
    )
    regressions.extend(
        _regression_rows(
            "log_volatility_on_log_turnover",
            np.log(daily_volatility[positive_volatility]),
            {"log_turnover": np.log(turnover[positive_volatility])},
        )
    )
    return metrics, pairs, pl.DataFrame(regressions)


def plot_alpha101_paper_figures(
    metrics: pl.DataFrame,
    pairs: pl.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Recreate Figures 1-4 of *101 Formulaic Alphas* with the supplied results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "figure1": output_dir / "paper_figure1_distributions.png",
        "figure2": output_dir / "paper_figure2_return_vs_volatility.png",
        "figure3": output_dir / "paper_figure3_turnover_vs_correlation.png",
        "figure4": output_dir / "paper_figure4_volatility_vs_turnover.png",
    }

    panels = [
        (metrics["sharpe"].to_numpy(), "Sharpe Ratio", "#3366cc", False),
        (metrics["turnover"].to_numpy(), "log(Turnover)", "#2ca02c", True),
        (metrics["cents_per_share"].to_numpy(), "log(Cents-per-Share)", "#c49a6c", True),
        (metrics["daily_volatility"].to_numpy(), "log(Volatility)", "#ff9900", True),
        (metrics["annual_return"].to_numpy(), "log(Return)", "#ff9da7", True),
        (pairs["correlation"].to_numpy(), "Correlation", "#a64dff", False),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for ax, (raw_values, label, color, take_log) in zip(axes.flat, panels):
        finite = raw_values[np.isfinite(raw_values)]
        excluded = 0
        if take_log:
            excluded = int((finite <= 0).sum())
            finite = np.log(finite[finite > 0])
        x, density = _gaussian_kde(finite)
        ax.plot(x, density, color=color, linewidth=2)
        ax.fill_between(x, 0, density, color=color, alpha=0.12)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        note = f"n={len(finite)}"
        if excluded:
            note += f"; {excluded} nonpositive excluded"
        ax.text(0.98, 0.94, note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="0.35")
        ax.grid(True, color="0.92", linewidth=0.6)
    fig.suptitle("Paper Figure 1 remade — Alpha101 empirical distributions", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(paths["figure1"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    daily_return = metrics["daily_return"].to_numpy()
    volatility = metrics["daily_volatility"].to_numpy()
    turnover = metrics["turnover"].to_numpy()
    positive_return = (daily_return > 0) & (volatility > 0) & np.isfinite(daily_return) & np.isfinite(volatility)
    x = np.log(volatility[positive_return])
    y = np.log(daily_return[positive_return])
    fit = _fit_ols(y, {"log_volatility": x})
    _plot_regression(
        x,
        y,
        fit,
        paths["figure2"],
        "Paper Figure 2 remade — Return vs. volatility",
        "log(Volatility)",
        "log(Return)",
        "#18b64b",
    )

    x = pairs["turnover_model_contribution"].to_numpy()
    y = pairs["demeaned_correlation"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=11, color="#ff9dad", alpha=0.3, edgecolors="none")
    ax.axhline(0, color="0.35", linewidth=0.8)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel("Turnover-factor contribution")
    ax.set_ylabel("Demeaned Correlation")
    ax.set_title("Paper Figure 3 remade — Does turnover explain alpha correlations?")
    ax.text(
        0.02,
        0.98,
        f"pairs={len(x):,}; corr(x, y)={_corr(x, y):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(paths["figure3"], dpi=180, bbox_inches="tight")
    plt.close(fig)

    positive_volatility = (volatility > 0) & (turnover > 0) & np.isfinite(volatility) & np.isfinite(turnover)
    x = np.log(turnover[positive_volatility])
    y = np.log(volatility[positive_volatility])
    fit = _fit_ols(y, {"log_turnover": x})
    _plot_regression(
        x,
        y,
        fit,
        paths["figure4"],
        "Paper Figure 4 remade — Volatility vs. turnover",
        "log(Turnover)",
        "log(Volatility)",
        "#ff9900",
    )
    return paths


def render_alpha101_paper_report(
    metrics: pl.DataFrame,
    pairs: pl.DataFrame,
    regressions: pl.DataFrame,
) -> str:
    """Render the remade paper-figure statistics as Markdown."""
    positive_returns = int((metrics["annual_return"] > 0).sum())
    positive_cps = int((metrics["cents_per_share"] > 0).sum())
    lines = [
        "# Alpha101 paper figures remade with current data",
        "",
        "Definitions follow [*101 Formulaic Alphas*](https://arxiv.org/pdf/1601.00991), Section 3. ",
        "Turnover is gross daily dollars traded per dollar of gross investment; cents-per-share is ",
        "100 times mean daily PnL divided by mean daily shares traded (buys plus sells).",
        "",
        f"- Alphas: {metrics.height}",
        f"- Pairwise return correlations: {pairs.height:,}",
        f"- Mean / median correlation: {pairs['correlation'].mean():.4f} / {pairs['correlation'].median():.4f}",
        f"- Positive-return alphas used in log-return panels: {positive_returns}/{metrics.height}",
        f"- Positive-CPS alphas used in the log-CPS panel: {positive_cps}/{metrics.height}",
        "",
        "Nonpositive return and CPS observations are excluded only from panels where the paper takes a natural log; ",
        "signals are not flipped and absolute values are not substituted.",
        "",
        "## Regressions",
        "",
        "| Model | Term | Estimate | Std. error | t-stat | n | R² | Adjusted R² | F-stat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regressions.iter_rows(named=True):
        lines.append(
            f"| {row['model']} | {row['term']} | {row['estimate']:.6f} | {row['standard_error']:.6f} | "
            f"{row['t_statistic']:.3f} | {row['observations']} | {row['r_squared']:.4f} | "
            f"{row['adjusted_r_squared']:.4f} | {row['f_statistic']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Figure 1: empirical distributions](paper_figure1_distributions.png)",
            "",
            "![Figure 2: return vs volatility](paper_figure2_return_vs_volatility.png)",
            "",
            "![Figure 3: turnover vs demeaned correlation](paper_figure3_turnover_vs_correlation.png)",
            "",
            "![Figure 4: volatility vs turnover](paper_figure4_volatility_vs_turnover.png)",
            "",
        ]
    )
    return "\n".join(lines)


def _fit_ols(y: np.ndarray, features: dict[str, np.ndarray]) -> dict[str, object]:
    names = list(features)
    columns = [np.ones(len(y)), *(np.asarray(features[name], dtype=float) for name in names)]
    design = np.column_stack(columns)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    design, y = design[finite], y[finite]
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    observations, parameters = design.shape
    degrees_of_freedom = observations - parameters
    residual_sum_squares = float(np.dot(residuals, residuals))
    total_sum_squares = float(np.dot(y - y.mean(), y - y.mean()))
    variance = residual_sum_squares / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
    covariance = variance * np.linalg.pinv(design.T @ design)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    r_squared = 1 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else np.nan
    adjusted_r_squared = (
        1 - (1 - r_squared) * (observations - 1) / degrees_of_freedom if degrees_of_freedom > 0 else np.nan
    )
    predictors = parameters - 1
    f_statistic = (
        ((total_sum_squares - residual_sum_squares) / predictors) / (residual_sum_squares / degrees_of_freedom)
        if predictors > 0 and degrees_of_freedom > 0 and residual_sum_squares > 0
        else np.nan
    )
    return {
        "terms": ["intercept", *names],
        "coefficients": coefficients,
        "standard_errors": standard_errors,
        "t_statistics": t_statistics,
        "observations": observations,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "f_statistic": f_statistic,
    }


def _regression_rows(model: str, y: np.ndarray, features: dict[str, np.ndarray]) -> list[dict[str, object]]:
    fit = _fit_ols(y, features)
    return [
        {
            "model": model,
            "term": term,
            "estimate": float(fit["coefficients"][index]),
            "standard_error": float(fit["standard_errors"][index]),
            "t_statistic": float(fit["t_statistics"][index]),
            "observations": int(fit["observations"]),
            "r_squared": float(fit["r_squared"]),
            "adjusted_r_squared": float(fit["adjusted_r_squared"]),
            "f_statistic": float(fit["f_statistic"]),
        }
        for index, term in enumerate(fit["terms"])
    ]


def _gaussian_kde(values: np.ndarray, points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE using R's ``bw.nrd0`` bandwidth rule."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([0.0]), np.array([0.0])
    if len(values) == 1 or np.ptp(values) == 0:
        center = float(values[0])
        return np.array([center - 1e-6, center, center + 1e-6]), np.array([0.0, 1.0, 0.0])
    standard_deviation = np.std(values, ddof=1)
    quartiles = np.percentile(values, [25, 75])
    scale = min(standard_deviation, (quartiles[1] - quartiles[0]) / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        scale = standard_deviation
    bandwidth = 0.9 * scale * len(values) ** (-0.2)
    grid = np.linspace(values.min() - 3 * bandwidth, values.max() + 3 * bandwidth, points)
    standardized = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * standardized**2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def _plot_regression(
    x: np.ndarray,
    y: np.ndarray,
    fit: dict[str, object],
    output: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, s=25, color=color, alpha=0.8, edgecolors="white", linewidths=0.3)
    line_x = np.linspace(x.min(), x.max(), 200)
    coefficients = fit["coefficients"]
    ax.plot(line_x, coefficients[0] + coefficients[1] * line_x, color="#2455d6", linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"y = {coefficients[0]:.3f} + {coefficients[1]:.3f}x\n"
        f"n={fit['observations']}; R²={fit['r_squared']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    ax.grid(True, color="0.92", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
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
    result = {
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
    if {"dollar_volume", "shares_traded"} <= set(group.columns):
        mean_shares = float(group["shares_traded"].mean())
        result["paper_turnover"] = float(group["dollar_volume"].mean())
        result["cents_per_share"] = float(100 * np.nanmean(pnl) / mean_shares) if mean_shares > 0 else np.nan
    return result


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
