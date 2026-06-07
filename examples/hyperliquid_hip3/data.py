"""Data adapters for the Hyperliquid HIP-3 example.

Two sources are used, each for what it is good at:

* :class:`HyperliquidHIP3` — the on-chain perpetual prices that you can actually
  *trade*. A name is untradeable here until it is listed on HIP-3, so the listing
  date is read directly from the first available candle.
* :class:`YahooUnderlying` — the *underlying* equity history, used only to warm up
  the model (signals, factor covariance, point-in-time market cap and fundamentals).
  Nothing from Yahoo is ever realised as PnL; it only feeds quantities that are known
  strictly in the past at each decision date.

All time series returned here are point-in-time: market cap uses *raw* (unadjusted) price
times historical shares outstanding, returns use adjusted close, and fundamentals are lagged
by a filing delay before they become visible. Names without a historical share series are
dropped rather than back-filled with today's count (no look-ahead).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass

import pandas as pd
import requests

HIP3_INFO_URL = "https://api.hyperliquid.xyz/info"
# Conservative gap between a fiscal period end and the date the figures are public.
DEFAULT_FILING_LAG = pd.Timedelta(days=60)

# trader.xyz lists more than single-name equities. These index / commodity / FX / ETF symbols
# are excluded from auto-discovery up front because several collide with real equity tickers on
# Yahoo (e.g. GOLD=Barrick, CL=Colgate), which would otherwise pass the equity validation below.
NON_EQUITY_SYMBOLS = frozenset(
    {
        # broad indices / vol
        "XYZ100",
        "SP500",
        "KR200",
        "JP225",
        "NIFTY",
        "IBOV",
        "VIX",
        "H100",
        "DXY",
        # commodities
        "GOLD",
        "SILVER",
        "CL",
        "BRENTOIL",
        "COPPER",
        "NATGAS",
        "URANIUM",
        "ALUMINIUM",
        "PLATINUM",
        "PALLADIUM",
        "CORN",
        "WHEAT",
        "TTF",
        # FX
        "JPY",
        "EUR",
        "GBP",
        "KRW",
        # ETFs
        "EWY",
        "EWJ",
        "EWZ",
        "EWT",
        "XLE",
        "URNM",
    }
)


class HyperliquidHIP3:
    """Read-only client for HIP-3 builder-deployed perpetual market data.

    Parameters
    ----------
    dex: builder-deployed perp dex name (trader.xyz lists equities under ``"xyz"``)
    session: optional pre-configured ``requests.Session``
    timeout: per-request timeout in seconds
    """

    def __init__(self, dex: str = "xyz", session: requests.Session | None = None, timeout: int = 30):
        self.dex = dex
        self.timeout = timeout
        self._session = session or requests.Session()

    def _post(self, payload: dict) -> object:
        resp = self._session.post(HIP3_INFO_URL, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def universe(self) -> list[str]:
        """Return the bare coin names listed on ``self.dex`` (without the dex prefix)."""
        meta = self._post({"type": "meta", "dex": self.dex})
        return [u["name"].split(":")[1] for u in meta["universe"]]

    def discover_equities(self, validate: bool = True) -> list[str]:
        """Auto-discover the single-name equities currently listed on ``self.dex``.

        Starts from the live listing, drops the :data:`NON_EQUITY_SYMBOLS` blocklist, then (if
        ``validate``) keeps only names Yahoo confirms are equities with a sector, which also
        filters out foreign/unmappable listings. New equity listings are picked up automatically.
        """
        candidates = [c for c in self.universe() if c not in NON_EQUITY_SYMBOLS]
        if not validate:
            return sorted(candidates)
        import logging

        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # quiet 404s for non-US symbols
        equities = []
        for coin in candidates:
            try:
                info = yf.Ticker(coin).info
            except Exception:
                continue
            if info.get("quoteType") == "EQUITY" and info.get("sector"):
                equities.append(coin)
        return sorted(equities)

    def daily_closes(self, ticker: str, start: dt.date, end: dt.date) -> pd.Series:
        """Return a daily close-price series for ``ticker`` (index = UTC date).

        The HIP-3 asset is addressed as ``"{dex}:{ticker}"``. An empty series is
        returned when the asset has no candles in the window (e.g. not yet listed).
        """
        coin = f"{self.dex}:{ticker}"
        utc = dt.timezone.utc  # epoch from explicit UTC midnight, independent of machine timezone
        req = {
            "coin": coin,
            "interval": "1d",
            "startTime": int(dt.datetime.combine(start, dt.time(), utc).timestamp() * 1000),
            "endTime": int(dt.datetime.combine(end, dt.time(), utc).timestamp() * 1000),
        }
        candles = self._post({"type": "candleSnapshot", "req": req})
        if not candles:
            return pd.Series(dtype="float64", name=ticker)
        index = pd.to_datetime([dt.datetime.fromtimestamp(c["t"] / 1000, dt.timezone.utc).date() for c in candles])
        return pd.Series([float(c["c"]) for c in candles], index=index, name=ticker)

    def close_matrix(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return a ``dates x tickers`` close-price matrix; missing names are dropped."""
        series = {t: s for t in tickers if len(s := self.daily_closes(t, start, end))}
        return pd.DataFrame(series).sort_index()


@dataclass
class UnderlyingData:
    """Point-in-time underlying-equity data, all aligned to the trading-day calendar.

    Attributes
    ----------
    close: ``dates x tickers`` adjusted close prices (for returns)
    market_cap: ``dates x tickers`` market cap = raw_price(t) * shares_outstanding(t)
    book_price / sales_price / cf_price: ``dates x tickers`` value ratios, filing-lagged
    sectors: mapping of ticker -> GICS-style sector
    """

    close: pd.DataFrame
    market_cap: pd.DataFrame
    book_price: pd.DataFrame
    sales_price: pd.DataFrame
    cf_price: pd.DataFrame
    sectors: dict[str, str]


class YahooUnderlying:
    """Loads point-in-time underlying-equity data from Yahoo Finance via ``yfinance``.

    When ``cache_dir`` is set, each ticker is cached individually (price, shares, fundamentals
    and sector) and refreshed at most once per calendar day. This makes repeated same-day runs
    offline, and a universe change (e.g. a newly listed name) only fetches the new ticker(s).

    Parameters
    ----------
    filing_lag: delay applied to fundamentals before they are treated as known, to
        avoid using figures before they were publicly filed
    cache_dir: directory for the per-ticker cache; if ``None``, nothing is cached
    refresh: re-fetch every ticker even if a fresh cache entry exists
    """

    # close = adjusted (for total returns); close_raw = unadjusted (for point-in-time market cap)
    _COLS = ("close", "close_raw", "shares", "equity", "ttm_rev", "ttm_ocf")
    _CACHE_VERSION = "v3-rawcap"

    def __init__(
        self, filing_lag: pd.Timedelta = DEFAULT_FILING_LAG, cache_dir: str | None = None, refresh: bool = False
    ):
        self.filing_lag = filing_lag
        self.cache_dir = cache_dir
        self.refresh = refresh

    def load(self, tickers: list[str], start: dt.date, end: dt.date) -> UnderlyingData:
        """Return point-in-time panels for ``tickers``, fetching only what is missing/stale."""
        if self.cache_dir is None:
            frames, sectors = self._fetch_frames(tickers, start, end)
            return self._assemble(frames, sectors)

        os.makedirs(self.cache_dir, exist_ok=True)
        meta = self._read_meta()
        if meta.get("__version__") != self._CACHE_VERSION:  # cache format changed -> refetch all
            meta = {"__version__": self._CACHE_VERSION}
        today = str(dt.date.today())
        stale = [t for t in tickers if self.refresh or meta.get(t, {}).get("fetched") != today]
        if stale:
            frames, sectors = self._fetch_frames(stale, start, end)
            for t in stale:  # stamp all attempted names (incl. empties) so we don't refetch today
                meta[t] = {"sector": sectors.get(t), "fetched": today, "has_data": t in frames}
                if t in frames:
                    frames[t].to_parquet(self._path(t))
            self._write_meta(meta)

        frames, sectors = {}, {}
        for t in tickers:  # read only cache entries written by this version (has_data flag)
            if meta.get(t, {}).get("has_data") and os.path.exists(self._path(t)):
                frames[t] = pd.read_parquet(self._path(t))
                if sector := meta.get(t, {}).get("sector"):
                    sectors[t] = sector
        return self._assemble(frames, sectors)

    def _fetch_frames(self, tickers: list[str], start: dt.date, end: dt.date) -> tuple[dict, dict]:
        """Fetch raw Yahoo data and assemble one tidy per-ticker frame (``_COLS``) each."""
        import yfinance as yf

        # auto_adjust=False so we keep BOTH adjusted close (total returns) and raw close (cap).
        data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)
        adj, raw = data["Adj Close"], data["Close"]
        if isinstance(adj, pd.Series):
            adj, raw = adj.to_frame(tickers[0]), raw.to_frame(tickers[0])
        adj = adj.dropna(how="all").sort_index()
        adj.index = pd.to_datetime(adj.index)
        raw = raw.reindex(adj.index)
        kept = [t for t in tickers if t in adj.columns and adj[t].notna().sum() > 0]
        if not kept:
            return {}, {}
        adj = adj[kept]
        shares = self._shares_panel(yf, kept, adj, start)
        equity, revenue_ttm, ocf_ttm = self._fundamentals_panels(yf, kept, adj)
        frames, sectors = {}, {}
        for t in kept:
            frames[t] = pd.DataFrame(
                {
                    "close": adj[t],
                    "close_raw": raw[t],
                    "shares": shares[t],
                    "equity": equity[t],
                    "ttm_rev": revenue_ttm[t],
                    "ttm_ocf": ocf_ttm[t],
                }
            )
            if sector := self._sector(yf, t):
                sectors[t] = sector
        return frames, sectors

    @classmethod
    def _assemble(cls, frames: dict, sectors: dict) -> UnderlyingData:
        """Combine per-ticker frames into aligned ``dates x tickers`` panels."""
        if not frames:
            empty = pd.DataFrame()
            return UnderlyingData(empty, empty, empty, empty, empty, {})
        panels = {col: pd.concat({t: frames[t][col] for t in frames}, axis=1) for col in cls._COLS}
        close = panels["close"].dropna(how="all").sort_index()  # adjusted, for returns
        market_cap = panels["close_raw"].reindex(close.index) * panels["shares"].reindex(
            close.index
        )  # raw price * shares
        book_price = panels["equity"].reindex(close.index) / market_cap
        sales_price = panels["ttm_rev"].reindex(close.index) / market_cap
        cf_price = panels["ttm_ocf"].reindex(close.index) / market_cap
        return UnderlyingData(close, market_cap, book_price, sales_price, cf_price, sectors)

    def _path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"yh_{ticker}.parquet")

    def _read_meta(self) -> dict:
        path = os.path.join(self.cache_dir, "yh_meta.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        return {}

    def _write_meta(self, meta: dict) -> None:
        with open(os.path.join(self.cache_dir, "yh_meta.json"), "w") as fh:
            json.dump(meta, fh)

    def _shares_panel(self, yf, tickers: list[str], close: pd.DataFrame, start: dt.date) -> pd.DataFrame:
        """Point-in-time shares outstanding, forward-filled onto the price calendar.

        No fallback to current shares: if a ticker has no historical share series, its column is
        left NaN so its market cap is undefined and the name is dropped — avoiding any look-ahead
        from applying today's share count to past dates.
        """
        cols = {}
        for t in tickers:
            try:
                s = yf.Ticker(t).get_shares_full(start=start)
            except Exception:
                s = None
            if s is not None and len(s):
                s = s[~s.index.duplicated(keep="last")].sort_index()
                idx = pd.to_datetime(s.index)
                s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                cols[t] = s.reindex(close.index, method="ffill")
            else:
                cols[t] = pd.Series(index=close.index, dtype="float64")  # NaN -> name dropped
        return pd.DataFrame(cols).reindex_like(close)

    def _fundamentals_panels(self, yf, tickers: list[str], close: pd.DataFrame):
        """Filing-lagged book equity, TTM revenue and TTM operating cash flow panels."""
        eq, rev, ocf = {}, {}, {}
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                eq[t] = self._stepped(self._row(tk.quarterly_balance_sheet, ("Stockholders Equity",)), close.index)
                rev[t] = self._stepped(self._ttm(tk.quarterly_financials, "Total Revenue"), close.index)
                ocf[t] = self._stepped(self._ttm(tk.quarterly_cashflow, "Operating Cash Flow"), close.index)
            except Exception:  # yfinance scraper/network error on one ticker must not halt the rest
                empty = pd.Series(index=close.index, dtype="float64")
                eq[t] = rev[t] = ocf[t] = empty
        return (
            pd.DataFrame(eq).reindex_like(close),
            pd.DataFrame(rev).reindex_like(close),
            pd.DataFrame(ocf).reindex_like(close),
        )

    @staticmethod
    def _row(stmt: pd.DataFrame | None, names: tuple[str, ...]) -> pd.Series:
        if stmt is None or stmt.empty:
            return pd.Series(dtype="float64")
        for n in names:
            if n in stmt.index:
                return stmt.loc[n].dropna()
        return pd.Series(dtype="float64")

    @classmethod
    def _ttm(cls, stmt: pd.DataFrame | None, name: str) -> pd.Series:
        """Trailing-twelve-month sum of a flow statement line, indexed by period end.

        Returns empty when fewer than four quarters are available, so a young company's single
        quarter is never mixed cross-sectionally with mature companies' TTM (which would make it
        look ~4x more expensive); such names simply lack this value component until they mature.
        """
        row = cls._row(stmt, (name,)).sort_index()
        if len(row) < 4:
            return pd.Series(dtype="float64")
        return row.rolling(4).sum().dropna()

    def _stepped(self, by_period_end: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
        """Make a fundamental visible only ``filing_lag`` after period end, then step it forward."""
        if by_period_end.empty:
            return pd.Series(index=calendar, dtype="float64")
        s = by_period_end.copy()
        s.index = pd.to_datetime(s.index) + self.filing_lag
        return s[~s.index.duplicated(keep="last")].sort_index().reindex(calendar, method="ffill")

    @staticmethod
    def _sector(yf, ticker: str) -> str | None:
        try:
            return yf.Ticker(ticker).info.get("sector")
        except Exception:
            return None


def load_spy(start: dt.date, end: dt.date) -> pd.Series:
    """Daily S&P 500 (SPY) adjusted close, for the buy-and-hold benchmark."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    spy.index = pd.to_datetime(spy.index)
    return spy.rename("SPY")
