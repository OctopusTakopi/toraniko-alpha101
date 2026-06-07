# toraniko on Hyperliquid HIP-3 equity perps

An end-to-end example of running the [`toraniko`](https://github.com/0xfdf/toraniko) factor
risk model on the single-name equity perpetuals that **trader.xyz** deploys on Hyperliquid via
[HIP-3](https://hyperliquid.gitbook.io/hyperliquid-docs/hyperliquid-improvement-proposals-hips/hip-3-builder-deployed-perpetuals).

It estimates a market + sector + style (momentum, size, value) factor model with `toraniko`,
builds a **market- and sector-neutral mean-variance** book from it, and backtests that book —
realising PnL only on the HIP-3 perps — against a naive momentum book and S&P 500 buy-and-hold.
A companion tool prints the **target book to hold today** for manual trading.

> Educational example, not investment advice or a validated strategy. The HIP-3 history is only
> a few months long, so every result here is **illustrative, not statistically conclusive** — see
> [Caveats](#caveats).

## Why two data sources

A perp DEX gives you tradeable prices but no market cap and no fundamentals, and HIP-3 is too
young to even warm up a momentum signal. So responsibilities are split:

| Quantity | Source | Used for |
| --- | --- | --- |
| Tradeable returns | Hyperliquid HIP-3 perps | **Realised PnL only**, and only after a name is listed |
| Returns for signals | Yahoo underlying (adjusted close) | Momentum, factor returns, covariance |
| Market cap | Yahoo **raw** price × point-in-time shares | Size factor and WLS weighting |
| Fundamentals | Yahoo quarterly statements, filing-lagged | Value factor (time-varying) |
| Sectors | Yahoo | Sector factors |

The listing rule only limits *where PnL is realised* — you trade the perp once it is live. It
does not limit the data feeding the signal and risk model, so warming those up on the
underlying's long history is legitimate and introduces no look-ahead (see below).

## Method

Each trading day *t*, using only information available at *t*:

1. **Style factors** (`toraniko` functions, unchanged): `factor_mom` (120-day, exp half-life,
   lagged), `factor_sze` (size / SMB from market cap), `factor_val` (book/sales/cashflow-to-price).
2. **Factor returns**: `estimate_factor_returns` runs the cross-sectional WLS regression
   (weights `√cap`, sector sum-to-zero constraint) for market + sectors + styles, with
   **`residualize_styles=True`** so style returns are orthogonalised to market and sector.
3. **Risk model**: asset covariance `Σ = B Σ_f Bᵀ + diag(specific var)`, where `B` are the
   factor exposures, `Σ_f` is the factor-return covariance over a trailing window, and specific
   risk comes from per-name residuals. Residuals are reconstructed as `winsorize(r) − B·f`
   (exact for this model).
4. **Alpha**: equal-weighted composite of the standardised style scores, `z(mom)+z(sze)+z(val)`.
5. **Portfolio**: market- and sector-neutral mean-variance weights,
   `w ∝ Σ⁻¹α` projected onto `Cᵀw = 0` (C = market + sector exposures), scaled to a fixed gross
   book ($1 long / $1 short). This is `backtest.target_weights`, shared by the backtest and the
   daily report so they never disagree.

## No look-ahead

- **Signals** — `factor_mom` lags returns internally; size uses market cap from *raw*
  (unadjusted) price × point-in-time shares (so split/dividend adjustment can't leak future
  corporate actions into the cap level); value uses filing-lagged fundamentals (60-day default
  gap after each fiscal period end). Names without a historical share series are dropped, not
  back-filled with today's count.
- **Risk model** — factor covariance and specific risk use only history up to *t*.
- **Tradability** — a name enters only after its first HIP-3 candle (its listing date), and the
  book at *t* is restricted to names already priced at *t* (no peeking at *t+1*).
- **Realisation** — the book formed at the close of *t* earns the perp return *t → t+1*.

Residual look-ahead it does **not** remove (data-limited): the traded universe is the *current*
trader.xyz listing (delisted names are missing — survivorship), sector labels are current, and
the 60-day filing lag is an assumption, not the actual filing date.

## Layout

| File | Responsibility |
| --- | --- |
| `data.py` | `HyperliquidHIP3` (candles + equity auto-discovery) and `YahooUnderlying` (point-in-time underlying data, per-ticker cache) |
| `backtest.py` | Build factor inputs, run `estimate_factor_returns`, risk model, `target_weights`, walk-forward backtest |
| `run.py` | CLI: load → estimate → backtest → stats table + chart (vs naive momentum and SPY) |
| `report.py` | CLI: today's target book — factor performance, day-over-day changes, full weight list |

## Usage

```bash
pip install -r examples/hyperliquid_hip3/requirements.txt

# Backtest: table + chart vs naive momentum and SPY
python -m examples.hyperliquid_hip3.run --output /tmp/hip3_backtest.png

# Today's target book for manual trading
python -m examples.hyperliquid_hip3.report --output /tmp/hip3_today.png
```

The report prints a per-name table (side, today vs previous weight as % of GMV, Δ, and an
action — NEW / INCREASE / DECREASE / FLIP / EXIT) and plots three panels: recent factor
performance, day-over-day weight changes, and the full target-weight list.

**Universe** is auto-discovered by default: the live trader.xyz listing minus a non-equity
blocklist (commodities/FX/indices/ETFs), validated as US equities via Yahoo. New equity
listings are picked up automatically.

**Caching** (under `.cache/`): the universe scan is cached for `--scan-ttl-days` (default 7);
Yahoo data is cached **per ticker** and the market caches (Yahoo, HIP-3, SPY) refresh **once per
calendar day**. So repeated same-day runs are offline, and a newly listed name only fetches that
one ticker. `--refresh` forces a full re-fetch.

Useful flags: `--no-auto-discover` (use the curated fallback list), `--tickers …` (pin an
explicit universe, skips discovery), `--scan-ttl-days`, `--start/--end`, `--warmup-start`,
`--cost-bps`.

Programmatic use:

```python
from examples.hyperliquid_hip3.data import YahooUnderlying, HyperliquidHIP3
from examples.hyperliquid_hip3.backtest import build_base, estimate_factors, run_backtest, target_weights

tickers = HyperliquidHIP3().discover_equities()
under = YahooUnderlying(cache_dir=".cache").load(tickers, start, end)
inputs = build_base(under.close, under.market_cap, under.book_price, under.sales_price, under.cf_price, under.sectors)
model = estimate_factors(inputs)                      # residualize_styles=True
weights = target_weights(model, inputs.sector_names, date, tradable)  # today's book
```

## Example output

*Illustrative snapshot (≈6-month window, 46 names) — not a validated result; see Caveats.*

Backtest — full-model MVO vs. naive momentum vs. S&P 500 buy-and-hold:

![backtest](images/backtest.png)

Daily target book — factor performance, day-over-day changes, and the weights to hold:

![today's book](images/today.png)

## Caveats

- **History depth.** HIP-3 launched October 2025 with staggered listings, so the tradeable
  window is only a few months. Factor *premia* are not statistically estimable over it (t-stats
  below 2), and strategy rankings flip with the window. The pipeline is correct; the *edge* is
  unverified.
- **Perp ≠ stock.** Realised PnL uses perp mark prices (funding, basis, 24/7 trading); the
  signal is computed on the underlying. **Funding costs are not modelled** — only a simple
  per-turnover trading cost.
- **Universe** relies on a hand-maintained non-equity blocklist (`data.NON_EQUITY_SYMBOLS`) for
  category symbols that collide with equity tickers (e.g. GOLD, CL); a curated list
  (`DEFAULT_TICKERS`) is the offline fallback.
- **No position limits or volatility targeting** — the book is scaled to a fixed gross only.
- This is **decision support, not an order router.**
