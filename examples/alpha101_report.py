"""Generate an Alpha101 PnL/IC report from cached Yahoo-format data.

This example deliberately keeps data acquisition outside the factor library.  It expects a
wide OHLCV parquet produced by ``yfinance.download`` and the point-in-time share caches used by
the HIP-3 example.  Daily VWAP is approximated by typical price because Yahoo has no daily VWAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import polars as pl

from toraniko.alpha101 import factor_alpha101
from toraniko.alpha101_report import (
    analyze_alpha101,
    analyze_alpha101_paper,
    latest_alpha101_weights,
    plot_alpha101_paper_figures,
    plot_alpha101_pnl,
    plot_alpha101_weights,
    render_alpha101_paper_report,
    render_alpha101_report,
)

CLASSIFICATIONS = {
    "AAPL": ("Technology", "Hardware", "Computing Hardware"),
    "AMD": ("Technology", "Semiconductors", "Chip Designers"),
    "AMZN": ("Consumer Cyclical", "E-Commerce", "Online Marketplaces"),
    "ASML": ("Technology", "Semiconductors", "Chip Manufacturing"),
    "AVGO": ("Technology", "Semiconductors", "Chip Designers"),
    "BABA": ("Consumer Cyclical", "E-Commerce", "Online Marketplaces"),
    "BX": ("Financial Services", "Capital Markets", "Trading and Asset Management"),
    "COIN": ("Financial Services", "Capital Markets", "Trading and Asset Management"),
    "COST": ("Consumer Defensive", "Retail", "General Retail"),
    "DELL": ("Technology", "Hardware", "Computing Hardware"),
    "EBAY": ("Consumer Cyclical", "E-Commerce", "Online Marketplaces"),
    "GOOGL": ("Communication Services", "Interactive Media", "Digital Media"),
    "IBM": ("Technology", "Hardware", "Computing Hardware"),
    "INTC": ("Technology", "Semiconductors", "Chip Manufacturing"),
    "LLY": ("Healthcare", "Pharmaceuticals", "Large-Cap Pharma"),
    "META": ("Communication Services", "Interactive Media", "Digital Media"),
    "MSFT": ("Technology", "Software", "Enterprise Software"),
    "NFLX": ("Communication Services", "Interactive Media", "Digital Media"),
    "NVDA": ("Technology", "Semiconductors", "Chip Designers"),
    "ORCL": ("Technology", "Software", "Enterprise Software"),
    "TSLA": ("Consumer Cyclical", "Automotive", "Electric Vehicles"),
    "TSM": ("Technology", "Semiconductors", "Chip Manufacturing"),
}


def build_market_data(ohlcv_path: Path, shares_cache: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pd.read_parquet(ohlcv_path)
    frames = []
    for ticker in CLASSIFICATIONS:
        bars = raw.xs(ticker, axis=1, level="Ticker").copy()
        adjustment = bars["Adj Close"] / bars["Close"]
        shares = pl.read_parquet(shares_cache / f"yh_{ticker}.parquet").to_pandas().set_index("Date")
        bars["market_cap"] = shares["close_raw"] * shares["shares"]
        bars["execution_price"] = bars["Close"]
        bars["symbol"] = ticker
        bars["date"] = bars.index
        bars["returns"] = bars["Adj Close"].pct_change(fill_method=None)
        for column in ("Open", "High", "Low", "Close"):
            bars[column] *= adjustment
        bars["vwap"] = (bars["High"] + bars["Low"] + bars["Close"]) / 3
        frames.append(
            bars.rename(columns=str.lower)[
                [
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "vwap",
                    "returns",
                    "market_cap",
                    "execution_price",
                ]
            ]
        )
    market = pl.from_pandas(pd.concat(frames, ignore_index=True)).sort("date", "symbol")
    classifications = pl.DataFrame(
        [
            {"symbol": symbol, "sector": levels[0], "industry": levels[1], "subindustry": levels[2]}
            for symbol, levels in CLASSIFICATIONS.items()
        ]
    )
    return market, classifications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv", type=Path, required=True)
    parser.add_argument("--shares-cache", type=Path, default=Path("examples/hyperliquid_hip3/.cache"))
    parser.add_argument("--output", type=Path, default=Path("reports/alpha101_analysis.md"))
    parser.add_argument("--pnl-output", type=Path, default=Path("reports/alpha101_pnl.png"))
    parser.add_argument("--weights-output", type=Path, default=Path("reports/alpha101_weights.png"))
    parser.add_argument("--paper-output", type=Path, default=Path("reports/alpha101_paper_figures.md"))
    args = parser.parse_args()

    market, classifications = build_market_data(args.ohlcv, args.shares_cache)
    scores = factor_alpha101(market, classifications).collect()
    summary, daily = analyze_alpha101(
        scores,
        market.select("date", "symbol", pl.col("returns").alias("asset_returns")),
        prices_df=market.select("date", "symbol", pl.col("execution_price").alias("close")),
    )
    paper_metrics, pairwise_correlations, paper_regressions = analyze_alpha101_paper(summary, daily)
    weights = latest_alpha101_weights(scores)
    note = (
        f"Dataset: {len(CLASSIFICATIONS)} liquid equities, {market['date'].min().date()} to "
        f"{market['date'].max().date()}. Yahoo adjusted OHLCV; typical price proxies daily VWAP; "
        "point-in-time shares are used for market cap. Classification labels are a fixed research taxonomy."
    )
    report = render_alpha101_report(summary, dataset_note=note)
    report += (
        "\n## Charts\n\n![Alpha101 cumulative PnL](alpha101_pnl.png)\n\n"
        "![Alpha101 weights](alpha101_weights.png)\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    summary.write_csv(args.output.with_suffix(".csv"))
    plot_alpha101_pnl(summary, daily, args.pnl_output)
    plot_alpha101_weights(weights, args.weights_output)
    args.paper_output.parent.mkdir(parents=True, exist_ok=True)
    args.paper_output.write_text(
        render_alpha101_paper_report(paper_metrics, pairwise_correlations, paper_regressions)
    )
    paper_metrics.write_csv(args.paper_output.with_name("alpha101_paper_metrics.csv"))
    pairwise_correlations.write_csv(args.paper_output.with_name("alpha101_pairwise_correlations.csv"))
    paper_regressions.write_csv(args.paper_output.with_name("alpha101_paper_regressions.csv"))
    plot_alpha101_paper_figures(paper_metrics, pairwise_correlations, args.paper_output.parent)


if __name__ == "__main__":
    main()
