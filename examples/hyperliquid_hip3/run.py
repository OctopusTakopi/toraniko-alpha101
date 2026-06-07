"""End-to-end runner: warm-start the toraniko model on underlying history and trade HIP-3 perps.

Usage
-----
    python -m examples.hyperliquid_hip3.run --output /tmp/hip3_backtest.png

The first run downloads data (HIP-3 candles + Yahoo prices/shares/fundamentals) and caches
it under ``--cache-dir``; subsequent runs are offline unless ``--refresh`` is passed.

This is an *example*, not investment advice. Read ``README.md`` for the data caveats — the
HIP-3 history is only a few months long, so results here are illustrative, not conclusive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from .backtest import build_base, estimate_factors, run_backtest, summary_stats
from .data import HyperliquidHIP3, YahooUnderlying, load_spy

# Curated fallback universe of HIP-3 (trader.xyz) single-name equities, used when
# auto-discovery is disabled or unavailable (offline). The default path scans live.
DEFAULT_TICKERS = [
    "AAPL",
    "AMD",
    "AMZN",
    "ARM",
    "AVGO",
    "BABA",
    "BB",
    "BX",
    "COIN",
    "COST",
    "CRCL",
    "DELL",
    "DKNG",
    "EBAY",
    "GME",
    "GOOGL",
    "HIMS",
    "HOOD",
    "IBM",
    "INTC",
    "LLY",
    "LITE",
    "META",
    "MRVL",
    "MSFT",
    "MSTR",
    "MU",
    "NBIS",
    "NFLX",
    "NOW",
    "NVDA",
    "ORCL",
    "PLTR",
    "RIVN",
    "RKLB",
    "TSLA",
    "TSM",
    "ZM",
]


def _read_scan_cache(path: str, ttl_days: int) -> list[str] | None:
    """Return the cached discovered universe if present and fresher than ``ttl_days``."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        scanned = dt.date.fromisoformat(blob["scanned"])
    except (ValueError, KeyError):
        return None
    if (dt.date.today() - scanned).days > ttl_days:
        return None
    return blob.get("tickers") or None


def _write_scan_cache(path: str, tickers: list[str]) -> None:
    with open(path, "w") as fh:
        json.dump({"scanned": str(dt.date.today()), "tickers": tickers}, fh)


def resolve_universe(cache_dir: str, *, auto_discover: bool, refresh: bool, ttl_days: int) -> list[str]:
    """Resolve the trading universe: cached scan -> live trader.xyz scan -> curated fallback."""
    if not auto_discover:
        return list(DEFAULT_TICKERS)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "discovered_tickers.json")
    if not refresh and (cached := _read_scan_cache(path, ttl_days)) is not None:
        print(f"universe: {len(cached)} equities (cached trader.xyz scan)")
        return cached
    print("scanning trader.xyz for listed equities ...")
    try:
        tickers = HyperliquidHIP3().discover_equities()
    except Exception as exc:  # offline / API error -> stay usable
        print(f"  scan failed ({exc}); using curated fallback list")
        return list(DEFAULT_TICKERS)
    if not tickers:
        print("  scan returned no equities; using curated fallback list")
        return list(DEFAULT_TICKERS)
    _write_scan_cache(path, tickers)
    new = sorted(set(tickers) - set(DEFAULT_TICKERS))
    print(f"  found {len(tickers)} equities" + (f"; new vs curated: {', '.join(new)}" if new else ""))
    return tickers


def _fresh_today(path: str) -> bool:
    """True if ``path`` exists and was written today (market caches refresh once per day)."""
    return os.path.exists(path) and dt.date.fromtimestamp(os.path.getmtime(path)) == dt.date.today()


def _cached(path: str, refresh: bool, build):
    """Daily-fresh parquet cache: reuse only if written today, else rebuild via ``build``."""
    if _fresh_today(path) and not refresh:
        return pd.read_parquet(path)
    obj = build()
    obj.to_frame().to_parquet(path) if isinstance(obj, pd.Series) else obj.to_parquet(path)
    return obj


def _load_hip(cache_dir: str, tickers: list[str], start, end, refresh: bool) -> "pd.DataFrame":
    """Daily-fresh, universe-incremental HIP-3 close matrix: fetch only missing names same-day."""
    path = os.path.join(cache_dir, "hip3_close.parquet")
    if _fresh_today(path) and not refresh:
        cached = pd.read_parquet(path)
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            return cached
        extra = HyperliquidHIP3().close_matrix(missing, start, end)
        merged = cached.join(extra, how="outer") if len(extra.columns) else cached
    else:
        merged = HyperliquidHIP3().close_matrix(tickers, start, end)
    merged.sort_index().to_parquet(path)
    return merged


def load_all(tickers, start, end, warmup_start, cache_dir, refresh):
    """Return (underlying_data, hip3_close_matrix, spy_series).

    Caches refresh once per calendar day; the underlying cache is per-ticker (see
    ``YahooUnderlying``), so a new listing only fetches that name rather than the whole universe.
    """
    os.makedirs(cache_dir, exist_ok=True)
    under = YahooUnderlying(cache_dir=cache_dir, refresh=refresh).load(tickers, warmup_start, end)
    hip = _load_hip(cache_dir, tickers, start, end, refresh)
    spy = _cached(os.path.join(cache_dir, "spy.parquet"), refresh, lambda: load_spy(start, end))
    spy = spy["SPY"] if isinstance(spy, pd.DataFrame) else spy
    return under, hip, spy


def build_perp_panel(hip_close: pd.DataFrame, calendar: pd.DatetimeIndex):
    """Align HIP-3 closes to the trading calendar; return (returns, first-listing dates)."""
    hip_close.index = pd.to_datetime(hip_close.index)
    aligned = hip_close.reindex(calendar)
    listing = {t: hip_close[t].first_valid_index() for t in hip_close.columns}
    return aligned.pct_change(), listing


def _print_table(rows: dict[str, np.ndarray]) -> None:
    header = f"{'strategy':24}{'total%':>8}{'ann%':>8}{'vol%':>7}{'Sharpe':>8}{'t':>7}{'maxDD%':>8}{'hit%':>6}"
    print(header)
    for name, series in rows.items():
        s = summary_stats(series)
        print(
            f"{name:24}{s['total_pct']:>8.1f}{s['ann_pct']:>8.1f}{s['vol_pct']:>7.1f}"
            f"{s['sharpe']:>8.2f}{s['t_stat']:>7.2f}{s['max_drawdown_pct']:>8.1f}{s['hit_rate_pct']:>6.0f}"
        )


def _plot(result, rows, output) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(12, 6))
    styles = {
        "Full model MVO": dict(lw=1.4, color="#1f77b4"),
        "  net @10bps": dict(lw=1.0, color="#d62728"),
        "Naive momentum": dict(lw=1.0, color="0.5", ls="--"),
        "S&P 500 (SPY)": dict(lw=1.3, color="#2ca02c"),
    }
    for name, series in rows.items():
        ax.plot(result.dates, np.cumprod(1.0 + series) - 1.0, label=name, **styles.get(name, {}))
    ax.axhline(0, color="grey", lw=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(
        "toraniko full model on Hyperliquid HIP-3 perps (trader.xyz)\n"
        "warm-started on underlying history; market/sector-neutral; PnL realised on perps, post-listing",
        fontsize=10,
    )
    ax.grid(True, color="0.85", lw=0.6)
    ax.legend(loc="best")
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    print(f"\nsaved chart -> {output}")


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
    parser.add_argument("--cost-bps", type=float, default=10.0, help="per-turnover trading cost in bps")
    parser.add_argument("--output", default="/tmp/hip3_backtest.png", help="chart output path")
    args = parser.parse_args(argv)

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    warmup = dt.date.fromisoformat(args.warmup_start)

    tickers = args.tickers or resolve_universe(
        args.cache_dir, auto_discover=args.auto_discover, refresh=args.refresh, ttl_days=args.scan_ttl_days
    )
    under, hip, spy = load_all(tickers, start, end, warmup, args.cache_dir, args.refresh)
    span = f"{under.close.index.min().date()} -> {under.close.index.max().date()}"
    print(f"underlying warm-up: {under.close.shape[1]} names, {span}")

    inputs = build_base(
        under.close, under.market_cap, under.book_price, under.sales_price, under.cf_price, under.sectors
    )
    model = estimate_factors(inputs)
    perp_returns, listing = build_perp_panel(hip, under.close.index)
    result = run_backtest(model, inputs.sector_names, perp_returns, listing)

    spy_returns = spy.pct_change().reindex(result.dates).fillna(0).to_numpy()
    net = result.mvo - (args.cost_bps / 1e4) * result.turnover
    rows = {
        "Full model MVO": result.mvo,
        "  net @10bps": net,
        "Naive momentum": result.momentum,
        "S&P 500 (SPY)": spy_returns,
    }
    print(
        f"\nbacktest: {len(result.mvo)} days "
        f"({result.dates[0].date()} -> {result.dates[-1].date()}), "
        f"turnover {result.turnover.mean() * 100:.0f}%/day\n"
    )
    _print_table(rows)
    _plot(result, rows, args.output)


if __name__ == "__main__":
    main()
