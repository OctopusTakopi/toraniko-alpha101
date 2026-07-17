"""Run all 101 formulaic alphas on a broad S&P Composite 1500 universe.

The constituent snapshot combines the current S&P 500, MidCap 400 and SmallCap 600 lists.
Adjusted Yahoo daily OHLCV drives the formulas and returns; unadjusted closes and historical
shares outstanding reconstruct market capitalization and paper-style cents-per-share.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Allow running as a plain script (``python examples/alpha101_full_market/run.py``)
# from any working directory by putting the repository root on the import path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from toraniko.alpha101 import ALPHA101_FORMULA_VERSION, factor_alpha101  # noqa: E402
from toraniko.alpha101_report import (  # noqa: E402
    analyze_alpha101,
    analyze_alpha101_paper,
    latest_alpha101_weights,
    plot_alpha101_paper_figures,
    plot_alpha101_pnl,
    plot_alpha101_weights,
    render_alpha101_paper_report,
    render_alpha101_report,
)

CONSTITUENT_PAGES = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P MidCap 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P SmallCap 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
GICS_PAGE = "https://en.wikipedia.org/wiki/Global_Industry_Classification_Standard"
USER_AGENT = "toraniko-alpha101-research/1.0"
LOGGER = logging.getLogger(__name__)
SUBINDUSTRY_ALIASES = {"Specialty Stores": "Other Specialty Retail"}


def _score_cache_version(path: Path) -> str | None:
    try:
        metadata = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    version = metadata.get("formula_version") if isinstance(metadata, dict) else None
    return version if isinstance(version, str) else None


def fetch_universe(snapshot_path: Path, *, refresh: bool = False) -> pd.DataFrame:
    """Return a cached current-constituent S&P Composite 1500 snapshot with GICS levels."""
    if snapshot_path.exists() and not refresh:
        return pd.read_csv(snapshot_path)

    import requests

    headers = {"User-Agent": USER_AGENT}
    frames = []
    for index_name, url in CONSTITUENT_PAGES.items():
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        frames.append(
            table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].assign(index=index_name)
        )
    response = requests.get(GICS_PAGE, headers=headers, timeout=30)
    response.raise_for_status()
    hierarchy = pd.read_html(StringIO(response.text))[0]
    hierarchy = hierarchy[["Sector.1", "Industry.1", "Sub-Industry.1"]].rename(
        columns={"Sector.1": "sector", "Industry.1": "industry", "Sub-Industry.1": "subindustry"}
    )

    universe = pd.concat(frames, ignore_index=True).rename(
        columns={
            "Symbol": "source_symbol",
            "Security": "security",
            "GICS Sector": "source_sector",
            "GICS Sub-Industry": "subindustry",
        }
    )
    universe["subindustry"] = universe["subindustry"].replace(SUBINDUSTRY_ALIASES)
    universe = universe.merge(hierarchy, on="subindustry", how="left")
    universe["symbol"] = universe["source_symbol"].map(normalize_yahoo_symbol)
    universe["snapshot_date"] = dt.date.today().isoformat()
    columns = [
        "symbol",
        "source_symbol",
        "security",
        "index",
        "sector",
        "industry",
        "subindustry",
        "snapshot_date",
    ]
    universe = universe[columns].sort_values("symbol").drop_duplicates("symbol", keep="first")
    if universe[["sector", "industry", "subindustry"]].isna().any().any():
        raise ValueError("GICS hierarchy mapping is incomplete")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(snapshot_path, index=False)
    return universe


def normalize_yahoo_symbol(symbol: str) -> str:
    """Translate S&P class-share notation to Yahoo's ticker notation."""
    return str(symbol).strip().upper().replace(".", "-")


def load_prices(
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    cache_dir: Path,
    *,
    refresh: bool = False,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Download and cache adjusted OHLCV for the requested universe."""
    cache_path = cache_dir / "prices.parquet"
    metadata_path = cache_dir / "prices.json"
    digest = hashlib.sha256("\n".join(sorted(symbols)).encode()).hexdigest()
    expected = {"start": start.isoformat(), "end": end.isoformat(), "universe_sha256": digest}
    if cache_path.exists() and metadata_path.exists() and not refresh:
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            return pd.read_parquet(cache_path)

    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    collected: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=False,
            repair=True,
            progress=False,
            threads=True,
            timeout=30,
        )
        collected.update(_split_yahoo_download(data, batch))
        LOGGER.info("prices: %d/%d requested", min(offset + len(batch), len(symbols)), len(symbols))

    missing = [symbol for symbol in symbols if symbol not in collected]
    for symbol in missing:
        try:
            data = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=False,
                repair=True,
                progress=False,
                threads=False,
                timeout=30,
            )
            collected.update(_split_yahoo_download(data, [symbol]))
        except Exception:
            LOGGER.debug("single-symbol price fallback failed for %s", symbol, exc_info=True)
            continue
    if not collected:
        raise RuntimeError("Yahoo returned no price histories")
    prices = pd.concat(collected.values(), ignore_index=True).sort_values(["date", "symbol"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path, index=False)
    metadata_path.write_text(json.dumps({**expected, "symbols_downloaded": len(collected)}, indent=2))
    LOGGER.info("prices: retained %d/%d symbols", len(collected), len(symbols))
    return prices


def _split_yahoo_download(data: pd.DataFrame, requested: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    if data is None or data.empty:
        return frames
    ticker_level = "Ticker" if isinstance(data.columns, pd.MultiIndex) and "Ticker" in data.columns.names else None
    for symbol in requested:
        try:
            bars = data.xs(symbol, axis=1, level=ticker_level).copy() if ticker_level else data.copy()
        except KeyError:
            continue
        required = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not required <= set(bars.columns) or bars["Adj Close"].notna().sum() < 30:
            continue
        bars = bars.sort_index()
        adjustment = bars["Adj Close"] / bars["Close"]
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(bars.index).tz_localize(None).normalize(),
                "symbol": symbol,
                "open": bars["Open"] * adjustment,
                "high": bars["High"] * adjustment,
                "low": bars["Low"] * adjustment,
                "close": bars["Adj Close"],
                "execution_price": bars["Close"],
                "volume": bars["Volume"],
            }
        ).dropna(subset=["close"])
        frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3
        frames[symbol] = frame
    return frames


def load_shares(
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    cache_dir: Path,
    *,
    refresh: bool = False,
    workers: int = 4,
) -> dict[str, pd.Series]:
    """Download and cache point-in-time shares outstanding for market capitalization."""
    shares_dir = cache_dir / "shares"
    shares_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.Series] = {}
    pending = []
    for symbol in symbols:
        path = shares_dir / f"{symbol}.parquet"
        if path.exists() and not refresh:
            frame = pd.read_parquet(path)
            result[symbol] = pd.Series(frame["shares"].to_numpy(), index=pd.to_datetime(frame["date"]))
        else:
            pending.append(symbol)

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_fetch_shares, symbol, start - dt.timedelta(days=370), end): symbol
                for symbol in pending
            }
            for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    series = future.result()
                except Exception:
                    series = pd.Series(dtype=float)
                if len(series):
                    result[symbol] = series
                    pd.DataFrame({"date": series.index, "shares": series.to_numpy()}).to_parquet(
                        shares_dir / f"{symbol}.parquet", index=False
                    )
                if count % 50 == 0 or count == len(pending):
                    LOGGER.info("shares: %d/%d fetched; %d usable", count, len(pending), len(result))
    return result


def _fetch_shares(symbol: str, start: dt.date, end: dt.date) -> pd.Series:
    import yfinance as yf

    for attempt in range(3):
        try:
            series = yf.Ticker(symbol).get_shares_full(start=start, end=end)
            if series is not None and len(series):
                series = pd.to_numeric(series, errors="coerce").dropna()
                index = pd.to_datetime(series.index)
                series.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
                return series[series > 0][~series.index.duplicated(keep="last")].sort_index()
        except Exception:
            if attempt == 2:
                break
            time.sleep(1 + attempt)
    return pd.Series(dtype=float)


def build_market_data(prices: pd.DataFrame, shares: dict[str, pd.Series]) -> pd.DataFrame:
    """Attach point-in-time capitalization and returns to adjusted OHLCV."""
    frames = []
    for symbol, bars in prices.groupby("symbol", sort=False):
        bars = bars.sort_values("date").copy()
        series = shares.get(symbol)
        if series is None or not len(series):
            aligned_shares = pd.Series(np.nan, index=bars.index)
        else:
            aligned_shares = series.reindex(pd.DatetimeIndex(bars["date"]), method="ffill")
            aligned_shares.index = bars.index
        bars["market_cap"] = bars["execution_price"] * aligned_shares
        bars["returns"] = bars["close"].pct_change(fill_method=None)
        frames.append(bars)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])


def run_report(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = args.output_dir / "universe.csv"
    universe = fetch_universe(universe_path, refresh=args.refresh_universe)
    symbols = universe["symbol"].tolist()
    prices = load_prices(
        symbols,
        args.warmup_start,
        args.end,
        args.cache_dir,
        refresh=args.refresh_prices,
        batch_size=args.batch_size,
    )
    downloaded = sorted(prices["symbol"].unique())
    shares = load_shares(
        downloaded,
        args.warmup_start,
        args.end,
        args.cache_dir,
        refresh=args.refresh_shares,
        workers=args.workers,
    )
    market_pd = build_market_data(prices, shares)
    market = pl.from_pandas(market_pd).sort("date", "symbol")
    classifications = pl.from_pandas(
        universe.loc[universe["symbol"].isin(downloaded), ["symbol", "sector", "industry", "subindustry"]]
    )

    scores_path = args.cache_dir / "scores.parquet"
    scores_metadata_path = args.cache_dir / "scores.json"
    refresh_scores = args.refresh_scores or args.refresh_prices or args.refresh_shares or args.refresh_universe
    use_cached_scores = scores_path.exists() and not refresh_scores
    if use_cached_scores and _score_cache_version(scores_metadata_path) != ALPHA101_FORMULA_VERSION:
        LOGGER.info("factors: cached score formula version is stale; recomputing")
        use_cached_scores = False
    if use_cached_scores:
        LOGGER.info("factors: loading cached Alpha101 scores")
        scores = pl.read_parquet(scores_path)
        score_bounds = scores.select(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.col("symbol").n_unique().alias("symbols"),
        ).row(0, named=True)
        market_last = market.select(pl.col("date").max()).item()
        cache_is_current = (
            pd.Timestamp(score_bounds["first_date"]).date() == args.start
            and pd.Timestamp(score_bounds["last_date"]).date() == pd.Timestamp(market_last).date()
            and score_bounds["symbols"] == len(downloaded)
        )
        if not cache_is_current:
            LOGGER.info("factors: cached scores are stale; recomputing")
            use_cached_scores = False
    if not use_cached_scores:
        LOGGER.info("factors: computing 101 alphas over %d symbols", len(downloaded))
        scores = factor_alpha101(market, classifications).collect().filter(pl.col("date") >= args.start)
        scores.write_parquet(scores_path)
        scores_metadata_path.write_text(json.dumps({"formula_version": ALPHA101_FORMULA_VERSION}, indent=2))
    summary, daily = analyze_alpha101(
        scores,
        market.select("date", "symbol", pl.col("returns").alias("asset_returns")),
        prices_df=market.select("date", "symbol", pl.col("execution_price").alias("close")),
    )
    weights = latest_alpha101_weights(scores)
    paper_metrics, pairwise_correlations, paper_regressions = analyze_alpha101_paper(summary, daily)

    market_cap_symbols = sum(series is not None and len(series) > 0 for series in shares.values())
    analysis_first = daily["date"].min()
    analysis_last = daily["date"].max()
    if hasattr(analysis_first, "date"):
        analysis_first = analysis_first.date()
    if hasattr(analysis_last, "date"):
        analysis_last = analysis_last.date()
    note = (
        f"Dataset: current-constituent S&P Composite 1500 snapshot ({len(universe)} securities; "
        f"{len(downloaded)} with Yahoo OHLCV; {market_cap_symbols} with historical shares), "
        f"analysis {analysis_first} to {analysis_last}, warm-up from {args.warmup_start}. "
        "Yahoo adjusted daily OHLCV; typical price proxies VWAP; historical shares and raw closes form market cap. "
        "Current GICS classifications and membership introduce survivorship bias."
    )
    report = render_alpha101_report(summary, title="Full-Market WorldQuant Alpha101 Analysis", dataset_note=note)
    report += (
        "\n## Charts\n\n![Alpha101 cumulative PnL](alpha101_pnl.png)\n\n"
        "![Alpha101 weights](alpha101_weights.png)\n"
    )
    (args.output_dir / "alpha101_analysis.md").write_text(report)
    summary.write_csv(args.output_dir / "alpha101_analysis.csv")
    plot_alpha101_pnl(summary, daily, args.output_dir / "alpha101_pnl.png")
    plot_alpha101_weights(weights, args.output_dir / "alpha101_weights.png", classifications)
    (args.output_dir / "alpha101_paper_figures.md").write_text(
        render_alpha101_paper_report(paper_metrics, pairwise_correlations, paper_regressions)
    )
    paper_metrics.write_csv(args.output_dir / "alpha101_paper_metrics.csv")
    pairwise_correlations.write_csv(args.output_dir / "alpha101_pairwise_correlations.csv")
    paper_regressions.write_csv(args.output_dir / "alpha101_paper_regressions.csv")
    plot_alpha101_paper_figures(paper_metrics, pairwise_correlations, args.output_dir)
    metadata = {
        "universe_snapshot": str(universe["snapshot_date"].iloc[0]),
        "constituents": len(universe),
        "price_histories": len(downloaded),
        "share_histories": market_cap_symbols,
        "warmup_start": args.warmup_start.isoformat(),
        "analysis_start": args.start.isoformat(),
        "data_end_exclusive": args.end.isoformat(),
        "last_observation": str(market_pd["date"].max().date()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    today = dt.date.today()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2023, 1, 3))
    parser.add_argument("--warmup-start", type=dt.date.fromisoformat, default=dt.date(2021, 12, 30))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=today + dt.timedelta(days=1))
    parser.add_argument("--cache-dir", type=Path, default=Path("examples/alpha101_full_market/.cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/full_market"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh-universe", action="store_true")
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--refresh-shares", action="store_true")
    parser.add_argument("--refresh-scores", action="store_true")
    args = parser.parse_args()
    if not args.warmup_start < args.start < args.end:
        parser.error("expected warmup-start < start < end")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_report(parse_args())
