"""Daily target-weight report for manual trading of the HIP-3 book.

Running this prints, and plots, the book you should be holding *today*:

1. Recent factor (indicator) performance — which factors are working lately.
2. Day-over-day changes — names newly entered, increased, decreased or exited.
3. The full target-weight list — the book to maintain, as % of gross (GMV).

Usage
-----
    python -m examples.hyperliquid_hip3.report --output /tmp/hip3_today.png

Weights are the same market/sector-neutral mean-variance book the backtest trades
(``backtest.target_weights``); this tool simply reports the most recent day instead of
walking history. It is decision support, not an order router — see ``README.md`` caveats.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, build_base, estimate_factors, target_weights
from .run import build_perp_panel, load_all, resolve_universe

FACTOR_LABELS = {"market": "Market", "mom_score": "Momentum", "sze_score": "Size", "val_score": "Value"}
_EPS = 1e-9


def _tradable_on(model, listing, perp_returns, date) -> list[str]:
    """Names listed on HIP-3 by ``date`` that have a finite perp return on ``date``."""
    cross = model.base[model.base["date"] == date]
    return [
        s
        for s in cross["symbol"]
        if s in listing
        and pd.notna(listing[s])
        and listing[s] <= date
        and s in perp_returns.columns
        and pd.notna(perp_returns.at[date, s])
    ]


def _classify(today_w: float, prev_w: float) -> str:
    """Label the day-over-day change for one name."""
    flat_today, flat_prev = abs(today_w) < _EPS, abs(prev_w) < _EPS
    if flat_today and flat_prev:
        return "HOLD"
    if flat_prev:
        return "NEW"
    if flat_today:
        return "EXIT"
    if (today_w > 0) != (prev_w > 0):  # crossed from long to short or vice versa
        return "FLIP"
    if abs(today_w) > abs(prev_w) + _EPS:
        return "INCREASE"
    if abs(today_w) < abs(prev_w) - _EPS:
        return "DECREASE"
    return "HOLD"


def build_positions(today_w: pd.Series, prev_w: pd.Series, gross: float) -> pd.DataFrame:
    """Assemble a per-name table of today/previous weights (% of GMV), deltas and actions."""
    names = sorted(set(today_w.index) | set(prev_w.index))
    today = today_w.reindex(names).fillna(0.0)
    prev = prev_w.reindex(names).fillna(0.0)
    scale = 100.0 / gross  # express as % of gross book (GMV)
    table = pd.DataFrame(
        {
            "today_pct": today * scale,
            "prev_pct": prev * scale,
            "delta_pct": (today - prev) * scale,
            "side": np.where(today > _EPS, "LONG", np.where(today < -_EPS, "SHORT", "FLAT")),
            "action": [_classify(today[n], prev[n]) for n in names],
        },
        index=names,
    )
    return table.sort_values("today_pct", ascending=False)


def _print_table(positions: pd.DataFrame, today: pd.Timestamp) -> None:
    print(f"\nTarget book for {today.date()}  (weights as % of gross / GMV)\n")
    print(f"{'ticker':8}{'side':6}{'today%':>9}{'prev%':>9}{'delta%':>9}  action")
    for name, r in positions.iterrows():
        print(f"{name:8}{r['side']:6}{r['today_pct']:>9.2f}{r['prev_pct']:>9.2f}{r['delta_pct']:>+9.2f}  {r['action']}")
    longs = positions["today_pct"].clip(lower=0).sum()
    shorts = positions["today_pct"].clip(upper=0).sum()
    print(f"\nlong {longs:.0f}%  short {shorts:.0f}%  gross {longs - shorts:.0f}%  net {longs + shorts:+.1f}% (of GMV)")


def _plot(factor_returns: pd.DataFrame, positions: pd.DataFrame, today: pd.Timestamp, window: int, output: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.7], hspace=0.32, wspace=0.28)
    fig.suptitle(f"HIP-3 target book — {today.date()}", fontsize=13, fontweight="bold")

    # (1) recent factor performance
    ax = fig.add_subplot(grid[0, :])
    recent = factor_returns.tail(window)
    for col, label in FACTOR_LABELS.items():
        if col in recent.columns:
            ax.plot(recent.index, np.cumprod(1.0 + recent[col].to_numpy()) - 1.0, lw=1.3, label=label)
    ax.axhline(0, color="grey", lw=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(f"Indicator (factor) performance — last {window} trading days", fontsize=10)
    ax.legend(loc="best", ncol=4, fontsize=8)
    ax.grid(True, color="0.9", lw=0.5)

    # (2) day-over-day weight changes (exclude unchanged holds)
    ax = fig.add_subplot(grid[1, 0])
    changed = positions[positions["action"] != "HOLD"].sort_values("delta_pct")
    if len(changed):
        colors = ["#2ca02c" if d > 0 else "#d62728" for d in changed["delta_pct"]]
        ax.barh(changed.index, changed["delta_pct"], color=colors)
        for name, r in changed.iterrows():
            offset = 0.1 if r["delta_pct"] >= 0 else -0.1
            ax.text(
                r["delta_pct"] + offset,
                name,
                r["action"],
                va="center",
                ha="left" if r["delta_pct"] >= 0 else "right",
                fontsize=7,
            )
    else:
        ax.text(0.5, 0.5, "no changes vs. previous day", ha="center", va="center", transform=ax.transAxes)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_title("Changes vs. previous day (Δ weight, % of GMV)", fontsize=10)
    ax.set_xlabel("Δ weight (%)")
    ax.grid(True, axis="x", color="0.9", lw=0.5)

    # (3) full target weights
    ax = fig.add_subplot(grid[1, 1])
    book = positions[positions["side"] != "FLAT"]
    colors = ["#2ca02c" if w > 0 else "#d62728" for w in book["today_pct"]]
    ax.barh(book.index[::-1], book["today_pct"][::-1], color=colors[::-1])
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_title("Target weights to maintain (% of GMV)", fontsize=10)
    ax.set_xlabel("weight (%)  —  long > 0, short < 0")
    ax.grid(True, axis="x", color="0.9", lw=0.5)

    fig.savefig(output, dpi=130, bbox_inches="tight")
    print(f"\nsaved report -> {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None, help="explicit universe; skips auto-discovery")
    parser.add_argument(
        "--auto-discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan trader.xyz for listed equities (default: on)",
    )
    parser.add_argument("--scan-ttl-days", type=int, default=7, help="re-scan if the cached scan is older than this")
    parser.add_argument("--start", default="2025-10-01", help="HIP-3 / trading start (YYYY-MM-DD)")
    parser.add_argument("--end", default=str(dt.date.today()), help="end date (YYYY-MM-DD)")
    parser.add_argument("--warmup-start", default="2023-01-01", help="underlying warm-up start (YYYY-MM-DD)")
    parser.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), ".cache"))
    parser.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    parser.add_argument(
        "--window", type=int, default=60, help="lookback (trading days) for the factor-performance panel"
    )
    parser.add_argument("--output", default="/tmp/hip3_today.png", help="report image output path")
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    warmup = dt.date.fromisoformat(args.warmup_start)

    tickers = args.tickers or resolve_universe(
        args.cache_dir, auto_discover=args.auto_discover, refresh=args.refresh, ttl_days=args.scan_ttl_days
    )
    under, hip, _ = load_all(tickers, start, end, warmup, args.cache_dir, args.refresh)
    inputs = build_base(
        under.close, under.market_cap, under.book_price, under.sales_price, under.cf_price, under.sectors
    )
    model = estimate_factors(inputs)
    perp_returns, listing = build_perp_panel(hip, under.close.index)

    config = BacktestConfig()
    trade_dates = [d for d in sorted(model.base["date"].unique()) if d in model.factor_returns.index]
    if len(trade_dates) < 2:
        raise SystemExit("not enough history to form a book")
    today, prev = trade_dates[-1], trade_dates[-2]

    today_w = target_weights(
        model, inputs.sector_names, today, _tradable_on(model, listing, perp_returns, today), config
    )
    prev_w = target_weights(model, inputs.sector_names, prev, _tradable_on(model, listing, perp_returns, prev), config)
    if today_w.empty:
        raise SystemExit(f"no tradable book on {today.date()}")

    positions = build_positions(today_w, prev_w, config.gross)
    _print_table(positions, today)
    _plot(model.factor_returns, positions, today, args.window, args.output)


if __name__ == "__main__":
    main()
