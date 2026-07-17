"""WorldQuant 101 formulaic alphas.

The implementation follows Kakushadze (2016).  Inputs and outputs use Toraniko's
long-form ``date``/``symbol`` convention while calculations use dense date by
symbol arrays so that time-series and cross-sectional operators retain their
published semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

Array = np.ndarray
ALPHA101_FORMULA_VERSION = "share-volume-v1"
_PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "vwap", "returns", "market_cap")
_CLASS_COLUMNS = ("sector", "industry", "subindustry")
_NEUTRALIZED_ALPHAS = {
    48: ("subindustry",),
    58: ("sector",),
    59: ("industry",),
    63: ("industry",),
    67: ("sector", "subindustry"),
    69: ("industry",),
    70: ("industry",),
    76: ("sector",),
    79: ("sector",),
    80: ("industry",),
    82: ("sector",),
    87: ("industry",),
    89: ("industry",),
    90: ("subindustry",),
    91: ("industry",),
    93: ("industry",),
    97: ("industry",),
    100: ("subindustry",),
}


def _window(value: int | float) -> int:
    """Convert fractional lookbacks with floor, as specified in paper A.1."""
    return max(1, int(np.floor(float(value))))


def _rank_1d(values: Array) -> Array:
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    x = values[valid]
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    # Average-rank ties without a Python loop: label equal-value runs, then assign
    # each run the mean of the ordinal ranks (1..n) it spans.
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    np.not_equal(sorted_x[1:], sorted_x[:-1], out=is_new[1:])
    group_id = np.cumsum(is_new) - 1
    ordinal = np.arange(1, n + 1, dtype=float)
    group_avg = np.bincount(group_id, weights=ordinal) / np.bincount(group_id)
    ranks = np.empty(n, dtype=float)
    ranks[order] = group_avg[group_id]
    result[valid] = ranks / n
    return result


def rank(x: Array) -> Array:
    return np.vstack([_rank_1d(row) for row in x])


def delay(x: Array, period: int | float) -> Array:
    d = _window(period)
    out = np.full_like(x, np.nan, dtype=float)
    out[d:] = x[:-d]
    return out


def delta(x: Array, period: int | float) -> Array:
    return x - delay(x, period)


def _rolling(x: Array, window: int | float, fn: Callable[[Array], Array]) -> Array:
    w = _window(window)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(w - 1, len(x)):
        block = x[i - w + 1 : i + 1]
        valid = np.isfinite(block).all(axis=0)
        if valid.any():
            out[i, valid] = fn(block[:, valid])
    return out


def ts_sum(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.sum(axis=0))


def sma(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.mean(axis=0))


def stddev(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.std(axis=0, ddof=1))


def ts_min(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.min(axis=0))


def ts_max(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.max(axis=0))


def product(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: z.prod(axis=0))


def correlation(x: Array, y: Array, window: int | float) -> Array:
    w = _window(window)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(w - 1, len(x)):
        a, b = x[i - w + 1 : i + 1], y[i - w + 1 : i + 1]
        valid = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        if not valid.any():
            continue
        aa, bb = a[:, valid], b[:, valid]
        aa, bb = aa - aa.mean(axis=0), bb - bb.mean(axis=0)
        denom = np.sqrt((aa * aa).sum(axis=0) * (bb * bb).sum(axis=0))
        # WorldQuant operator convention: a defined window with zero variance has zero
        # correlation.  Keeping NaN here makes nested correlation/decay formulas disappear.
        vals = np.divide((aa * bb).sum(axis=0), denom, out=np.zeros(denom.shape), where=denom != 0)
        out[i, valid] = vals
    return out


def covariance(x: Array, y: Array, window: int | float) -> Array:
    w = _window(window)
    return _rolling_cov(x, y, w)


def _rolling_cov(x: Array, y: Array, window: int) -> Array:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        a, b = x[i - window + 1 : i + 1], y[i - window + 1 : i + 1]
        valid = np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)
        if valid.any():
            aa, bb = a[:, valid], b[:, valid]
            out[i, valid] = ((aa - aa.mean(axis=0)) * (bb - bb.mean(axis=0))).sum(axis=0) / (window - 1)
    return out


def ts_rank(x: Array, window: int | float) -> Array:
    # Normalize like cross-sectional rank.  This is required by formulas such as
    # 35 (1 - Ts_Rank) and 68/86 (Ts_Rank compared directly with rank).
    return _rolling(x, window, lambda z: np.array([_rank_1d(z[:, j])[-1] for j in range(z.shape[1])]))


def ts_argmax(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: np.argmax(z, axis=0).astype(float) + 1.0)


def ts_argmin(x: Array, window: int | float) -> Array:
    return _rolling(x, window, lambda z: np.argmin(z, axis=0).astype(float) + 1.0)


def decay_linear(x: Array, window: int | float) -> Array:
    w = _window(window)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return _rolling(x, w, lambda z: weights @ z)


def scale(x: Array, k: float = 1.0) -> Array:
    denom = np.nansum(np.abs(x), axis=1, keepdims=True)
    return np.divide(k * x, denom, out=np.full_like(x, np.nan), where=denom != 0)


def signed_power(x: Array, exponent: Array | float) -> Array:
    with np.errstate(invalid="ignore", over="ignore"):
        return np.sign(x) * np.power(np.abs(x), exponent)


def _safe_div(numerator: Array, denominator: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = numerator / denominator
    out[~np.isfinite(out)] = np.nan
    return out


def _if_else(condition: Array, when_true: Array | float, when_false: Array | float, *condition_values: Array) -> Array:
    """Vectorized conditional that preserves undefined condition inputs as NaN."""
    out = np.asarray(np.where(condition, when_true, when_false), dtype=float)
    valid = np.ones(out.shape, dtype=bool)
    for value in condition_values:
        valid &= np.isfinite(value)
    out[~valid] = np.nan
    return out


def _less(left: Array, right: Array) -> Array:
    """Numeric less-than result with NaN propagation."""
    return _if_else(left < right, 1.0, 0.0, left, right)


def indneutralize(x: Array, groups: Array) -> Array:
    """Residualize each date against classification dummies (group demean)."""
    if groups.shape != x.shape:
        raise ValueError("classification matrix must align with market data")
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        valid = np.isfinite(x[i]) & (groups[i] != None)  # noqa: E711
        for group in np.unique(groups[i, valid]):
            members = valid & (groups[i] == group)
            out[i, members] = x[i, members] - x[i, members].mean()
    return out


class Alpha101:
    """Evaluate the 101 alphas on aligned date-by-symbol arrays."""

    def __init__(self, data: dict[str, Array], classifications: dict[str, Array]):
        for name in _PRICE_COLUMNS:
            setattr(self, "returns" if name == "returns" else name, data[name])
        self.classifications = classifications

    def _ind(self, x: Array, level: str) -> Array:
        try:
            groups = self.classifications[level]
        except KeyError as exc:
            raise ValueError(f"classification column '{level}' is required") from exc
        return indneutralize(x, groups)

    def _adv(self, window: int) -> Array:
        return sma(self.volume, window)

    def alpha(self, number: int) -> Array:
        if not 1 <= number <= 101:
            raise ValueError("alpha number must be between 1 and 101")
        return getattr(self, f"alpha{number:03d}")()

    def alpha001(self) -> Array:
        base = _if_else(self.returns < 0, stddev(self.returns, 20), self.close, self.returns)
        return rank(ts_argmax(signed_power(base, 2.0), 5)) - 0.5

    def alpha002(self) -> Array:
        return -correlation(rank(delta(np.log(self.volume), 2)), rank(_safe_div(self.close - self.open, self.open)), 6)

    def alpha003(self) -> Array:
        return -correlation(rank(self.open), rank(self.volume), 10)

    def alpha004(self) -> Array:
        return -ts_rank(rank(self.low), 9)

    def alpha005(self) -> Array:
        return rank(self.open - sma(self.vwap, 10)) * -np.abs(rank(self.close - self.vwap))

    def alpha006(self) -> Array:
        return -correlation(self.open, self.volume, 10)

    def alpha007(self) -> Array:
        change = delta(self.close, 7)
        adv20 = self._adv(20)
        return _if_else(adv20 < self.volume, -ts_rank(np.abs(change), 60) * np.sign(change), -1.0, adv20, self.volume)

    def alpha008(self) -> Array:
        value = ts_sum(self.open, 5) * ts_sum(self.returns, 5)
        return -rank(value - delay(value, 10))

    def alpha009(self) -> Array:
        change = delta(self.close, 1)
        minimum, maximum = ts_min(change, 5), ts_max(change, 5)
        fallback = _if_else(maximum < 0, change, -change, maximum)
        return _if_else(minimum > 0, change, fallback, minimum)

    def alpha010(self) -> Array:
        change = delta(self.close, 1)
        minimum, maximum = ts_min(change, 4), ts_max(change, 4)
        fallback = _if_else(maximum < 0, change, -change, maximum)
        value = _if_else(minimum > 0, change, fallback, minimum)
        return rank(value)

    def alpha011(self) -> Array:
        spread = self.vwap - self.close
        return (rank(ts_max(spread, 3)) + rank(ts_min(spread, 3))) * rank(delta(self.volume, 3))

    def alpha012(self) -> Array:
        return np.sign(delta(self.volume, 1)) * -delta(self.close, 1)

    def alpha013(self) -> Array:
        return -rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self) -> Array:
        return -rank(delta(self.returns, 3)) * correlation(self.open, self.volume, 10)

    def alpha015(self) -> Array:
        return -ts_sum(rank(correlation(rank(self.high), rank(self.volume), 3)), 3)

    def alpha016(self) -> Array:
        return -rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self) -> Array:
        return (
            -rank(ts_rank(self.close, 10))
            * rank(delta(delta(self.close, 1), 1))
            * rank(ts_rank(_safe_div(self.volume, self._adv(20)), 5))
        )

    def alpha018(self) -> Array:
        value = stddev(np.abs(self.close - self.open), 5) + self.close - self.open
        return -rank(value + correlation(self.close, self.open, 10))

    def alpha019(self) -> Array:
        direction = -np.sign((self.close - delay(self.close, 7)) + delta(self.close, 7))
        return direction * (1 + rank(1 + ts_sum(self.returns, 250)))

    def alpha020(self) -> Array:
        return (
            -rank(self.open - delay(self.high, 1))
            * rank(self.open - delay(self.close, 1))
            * rank(self.open - delay(self.low, 1))
        )

    def alpha021(self) -> Array:
        mean8, mean2, sd8 = sma(self.close, 8), sma(self.close, 2), stddev(self.close, 8)
        ratio = _safe_div(self.volume, self._adv(20))
        volume_case = _if_else(ratio >= 1, 1.0, -1.0, ratio)
        lower_case = _if_else(mean2 < mean8 - sd8, 1.0, volume_case, mean2, mean8, sd8)
        return _if_else(mean8 + sd8 < mean2, -1.0, lower_case, mean8, sd8, mean2)

    def alpha022(self) -> Array:
        return -delta(correlation(self.high, self.volume, 5), 5) * rank(stddev(self.close, 20))

    def alpha023(self) -> Array:
        mean_high = sma(self.high, 20)
        return _if_else(mean_high < self.high, -delta(self.high, 2), 0.0, mean_high, self.high)

    def alpha024(self) -> Array:
        trend = _safe_div(delta(sma(self.close, 100), 100), delay(self.close, 100))
        return _if_else(trend <= 0.05, -(self.close - ts_min(self.close, 100)), -delta(self.close, 3), trend)

    def alpha025(self) -> Array:
        return rank(-self.returns * self._adv(20) * self.vwap * (self.high - self.close))

    def alpha026(self) -> Array:
        return -ts_max(correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5), 3)

    def alpha027(self) -> Array:
        value = rank(sma(correlation(rank(self.volume), rank(self.vwap), 6), 2))
        return _if_else(value > 0.5, -1.0, 1.0, value)

    def alpha028(self) -> Array:
        return scale(correlation(self._adv(20), self.low, 5) + (self.high + self.low) / 2 - self.close)

    def alpha029(self) -> Array:
        inner = rank(rank(-rank(delta(self.close - 1, 5))))
        first = ts_min(product(rank(rank(scale(np.log(ts_sum(ts_min(inner, 2), 1))))), 1), 5)
        return first + ts_rank(delay(-self.returns, 6), 5)

    def alpha030(self) -> Array:
        change = delta(self.close, 1)
        direction = np.sign(change) + np.sign(delay(change, 1)) + np.sign(delay(change, 2))
        return _safe_div((1 - rank(direction)) * ts_sum(self.volume, 5), ts_sum(self.volume, 20))

    def alpha031(self) -> Array:
        first = rank(rank(rank(decay_linear(-rank(rank(delta(self.close, 10))), 10))))
        return first + rank(-delta(self.close, 3)) + np.sign(scale(correlation(self._adv(20), self.low, 12)))

    def alpha032(self) -> Array:
        return scale(sma(self.close, 7) - self.close) + 20 * scale(correlation(self.vwap, delay(self.close, 5), 230))

    def alpha033(self) -> Array:
        return rank(self.open / self.close - 1)

    def alpha034(self) -> Array:
        return rank(2 - rank(_safe_div(stddev(self.returns, 2), stddev(self.returns, 5))) - rank(delta(self.close, 1)))

    def alpha035(self) -> Array:
        return (
            ts_rank(self.volume, 32)
            * (1 - ts_rank(self.close + self.high - self.low, 16))
            * (1 - ts_rank(self.returns, 32))
        )

    def alpha036(self) -> Array:
        return (
            2.21 * rank(correlation(self.close - self.open, delay(self.volume, 1), 15))
            + 0.7 * rank(self.open - self.close)
            + 0.73 * rank(ts_rank(delay(-self.returns, 6), 5))
            + rank(np.abs(correlation(self.vwap, self._adv(20), 6)))
            + 0.6 * rank((sma(self.close, 200) - self.open) * (self.close - self.open))
        )

    def alpha037(self) -> Array:
        return rank(correlation(delay(self.open - self.close, 1), self.close, 200)) + rank(self.open - self.close)

    def alpha038(self) -> Array:
        return -rank(ts_rank(self.close, 10)) * rank(_safe_div(self.close, self.open))

    def alpha039(self) -> Array:
        value = delta(self.close, 7) * (1 - rank(decay_linear(_safe_div(self.volume, self._adv(20)), 9)))
        return -rank(value) * (1 + rank(ts_sum(self.returns, 250)))

    def alpha040(self) -> Array:
        return -rank(stddev(self.high, 10)) * correlation(self.high, self.volume, 10)

    def alpha041(self) -> Array:
        return np.sqrt(self.high * self.low) - self.vwap

    def alpha042(self) -> Array:
        return _safe_div(rank(self.vwap - self.close), rank(self.vwap + self.close))

    def alpha043(self) -> Array:
        return ts_rank(_safe_div(self.volume, self._adv(20)), 20) * ts_rank(-delta(self.close, 7), 8)

    def alpha044(self) -> Array:
        return -correlation(self.high, rank(self.volume), 5)

    def alpha045(self) -> Array:
        return (
            -rank(sma(delay(self.close, 5), 20))
            * correlation(self.close, self.volume, 2)
            * rank(correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2))
        )

    def alpha046(self) -> Array:
        trend = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        fallback = _if_else(trend < 0, 1.0, -delta(self.close, 1), trend)
        return _if_else(trend > 0.25, -1.0, fallback, trend)

    def alpha047(self) -> Array:
        first = _safe_div(rank(1 / self.close) * self.volume, self._adv(20))
        second = _safe_div(self.high * rank(self.high - self.close), sma(self.high, 5))
        return first * second - rank(self.vwap - delay(self.vwap, 5))

    def alpha048(self) -> Array:
        change = delta(self.close, 1)
        numerator = correlation(change, delta(delay(self.close, 1), 1), 250) * change / self.close
        denominator = ts_sum(signed_power(_safe_div(change, delay(self.close, 1)), 2), 250)
        return _safe_div(self._ind(numerator, "subindustry"), denominator)

    def alpha049(self) -> Array:
        trend = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        return _if_else(trend < -0.1, 1.0, -delta(self.close, 1), trend)

    def alpha050(self) -> Array:
        return -ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5)

    def alpha051(self) -> Array:
        trend = (delay(self.close, 20) - delay(self.close, 10)) / 10 - (delay(self.close, 10) - self.close) / 10
        return _if_else(trend < -0.05, 1.0, -delta(self.close, 1), trend)

    def alpha052(self) -> Array:
        return (
            -delta(ts_min(self.low, 5), 5)
            * rank((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
            * ts_rank(self.volume, 5)
        )

    def alpha053(self) -> Array:
        return -delta(_safe_div((self.close - self.low) - (self.high - self.close), self.close - self.low), 9)

    def alpha054(self) -> Array:
        return _safe_div(-(self.low - self.close) * self.open**5, (self.low - self.high) * self.close**5)

    def alpha055(self) -> Array:
        position = _safe_div(self.close - ts_min(self.low, 12), ts_max(self.high, 12) - ts_min(self.low, 12))
        return -correlation(rank(position), rank(self.volume), 6)

    def alpha056(self) -> Array:
        return -rank(_safe_div(ts_sum(self.returns, 10), ts_sum(ts_sum(self.returns, 2), 3))) * rank(
            self.returns * self.market_cap
        )

    def alpha057(self) -> Array:
        return -_safe_div(self.close - self.vwap, decay_linear(rank(ts_argmax(self.close, 30)), 2))

    def alpha058(self) -> Array:
        return -ts_rank(
            decay_linear(correlation(self._ind(self.vwap, "sector"), self.volume, 3.92795), 7.89291), 5.50322
        )

    def alpha059(self) -> Array:
        mixed = self.vwap * 0.728317 + self.vwap * (1 - 0.728317)
        return -ts_rank(decay_linear(correlation(self._ind(mixed, "industry"), self.volume, 4.25197), 16.2289), 8.19648)

    def alpha060(self) -> Array:
        position = _safe_div((self.close - self.low) - (self.high - self.close), self.high - self.low) * self.volume
        return -(2 * scale(rank(position)) - scale(rank(ts_argmax(self.close, 10))))

    def alpha061(self) -> Array:
        left = rank(self.vwap - ts_min(self.vwap, 16.1219))
        right = rank(correlation(self.vwap, self._adv(180), 17.9282))
        return _less(left, right)

    def alpha062(self) -> Array:
        left = rank(correlation(self.vwap, ts_sum(self._adv(20), 22.4101), 9.91009))
        comparison = _less(rank(self.open) + rank(self.open), rank((self.high + self.low) / 2) + rank(self.high))
        return -_less(left, rank(comparison))

    def alpha063(self) -> Array:
        first = rank(decay_linear(delta(self._ind(self.close, "industry"), 2.25164), 8.22237))
        mixed = self.vwap * 0.318108 + self.open * (1 - 0.318108)
        second = rank(decay_linear(correlation(mixed, ts_sum(self._adv(180), 37.2467), 13.557), 12.2883))
        return -(first - second)

    def alpha064(self) -> Array:
        mixed1 = self.open * 0.178404 + self.low * (1 - 0.178404)
        left = rank(correlation(ts_sum(mixed1, 12.7054), ts_sum(self._adv(120), 12.7054), 16.6208))
        mixed2 = ((self.high + self.low) / 2) * 0.178404 + self.vwap * (1 - 0.178404)
        return -_less(left, rank(delta(mixed2, 3.69741)))

    def alpha065(self) -> Array:
        mixed = self.open * 0.00817205 + self.vwap * (1 - 0.00817205)
        left = rank(correlation(mixed, ts_sum(self._adv(60), 8.6911), 6.40374))
        return -_less(left, rank(self.open - ts_min(self.open, 13.635)))

    def alpha066(self) -> Array:
        first = rank(decay_linear(delta(self.vwap, 3.51013), 7.23052))
        ratio = _safe_div(
            self.low * 0.96633 + self.low * (1 - 0.96633) - self.vwap,
            self.open - (self.high + self.low) / 2,
        )
        return -(first + ts_rank(decay_linear(ratio, 11.4157), 6.72611))

    def alpha067(self) -> Array:
        base = rank(self.high - ts_min(self.high, 2.14593))
        exponent = rank(correlation(self._ind(self.vwap, "sector"), self._ind(self._adv(20), "subindustry"), 6.02936))
        return -signed_power(base, exponent)

    def alpha068(self) -> Array:
        left = ts_rank(correlation(rank(self.high), rank(self._adv(15)), 8.91644), 13.9333)
        right = rank(delta(self.close * 0.518371 + self.low * (1 - 0.518371), 1.06157))
        return -_less(left, right)

    def alpha069(self) -> Array:
        base = rank(ts_max(delta(self._ind(self.vwap, "industry"), 2.72412), 4.79344))
        mixed = self.close * 0.490655 + self.vwap * (1 - 0.490655)
        exponent = ts_rank(correlation(mixed, self._adv(20), 4.92416), 9.0615)
        return -signed_power(base, exponent)

    def alpha070(self) -> Array:
        base = rank(delta(self.vwap, 1.29456))
        exponent = ts_rank(correlation(self._ind(self.close, "industry"), self._adv(50), 17.8256), 17.9171)
        return -signed_power(base, exponent)

    def alpha071(self) -> Array:
        first = ts_rank(
            decay_linear(correlation(ts_rank(self.close, 3.43976), ts_rank(self._adv(180), 12.0647), 18.0175), 4.20501),
            15.6948,
        )
        second = ts_rank(decay_linear(signed_power(rank(self.low + self.open - 2 * self.vwap), 2), 16.4662), 4.4388)
        return np.maximum(first, second)

    def alpha072(self) -> Array:
        first = rank(decay_linear(correlation((self.high + self.low) / 2, self._adv(40), 8.93345), 10.1519))
        second = rank(
            decay_linear(correlation(ts_rank(self.vwap, 3.72469), ts_rank(self.volume, 18.5188), 6.86671), 2.95011)
        )
        return _safe_div(first, second)

    def alpha073(self) -> Array:
        first = rank(decay_linear(delta(self.vwap, 4.72775), 2.91864))
        mixed = self.open * 0.147155 + self.low * (1 - 0.147155)
        second = ts_rank(decay_linear(-_safe_div(delta(mixed, 2.03608), mixed), 3.33829), 16.7411)
        return -np.maximum(first, second)

    def alpha074(self) -> Array:
        first = rank(correlation(self.close, ts_sum(self._adv(30), 37.4843), 15.1365))
        mixed = self.high * 0.0261661 + self.vwap * (1 - 0.0261661)
        second = rank(correlation(rank(mixed), rank(self.volume), 11.4791))
        return -_less(first, second)

    def alpha075(self) -> Array:
        left = rank(correlation(self.vwap, self.volume, 4.24304))
        right = rank(correlation(rank(self.low), rank(self._adv(50)), 12.4413))
        return _less(left, right)

    def alpha076(self) -> Array:
        first = rank(decay_linear(delta(self.vwap, 1.24383), 11.8259))
        corr = correlation(self._ind(self.low, "sector"), self._adv(81), 8.14941)
        second = ts_rank(decay_linear(ts_rank(corr, 19.569), 17.1543), 19.383)
        return -np.maximum(first, second)

    def alpha077(self) -> Array:
        first = rank(decay_linear((self.high + self.low) / 2 - self.vwap, 20.0451))
        second = rank(decay_linear(correlation((self.high + self.low) / 2, self._adv(40), 3.1614), 5.64125))
        return np.minimum(first, second)

    def alpha078(self) -> Array:
        mixed = self.low * 0.352233 + self.vwap * (1 - 0.352233)
        base = rank(correlation(ts_sum(mixed, 19.7428), ts_sum(self._adv(40), 19.7428), 6.83313))
        exponent = rank(correlation(rank(self.vwap), rank(self.volume), 5.77492))
        return signed_power(base, exponent)

    def alpha079(self) -> Array:
        mixed = self.close * 0.60733 + self.open * (1 - 0.60733)
        left = rank(delta(self._ind(mixed, "sector"), 1.23438))
        right = rank(correlation(ts_rank(self.vwap, 3.60973), ts_rank(self._adv(150), 9.18637), 14.6644))
        return _less(left, right)

    def alpha080(self) -> Array:
        mixed = self.open * 0.868128 + self.high * (1 - 0.868128)
        base = rank(np.sign(delta(self._ind(mixed, "industry"), 4.04545)))
        exponent = ts_rank(correlation(self.high, self._adv(10), 5.11456), 5.53756)
        return -signed_power(base, exponent)

    def alpha081(self) -> Array:
        corr = correlation(self.vwap, ts_sum(self._adv(10), 49.6054), 8.47743)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = rank(np.log(product(rank(signed_power(rank(corr), 4)), 14.9655)))
        right = rank(correlation(rank(self.vwap), rank(self.volume), 5.07914))
        return -_less(left, right)

    def alpha082(self) -> Array:
        first = rank(decay_linear(delta(self.open, 1.46063), 14.8717))
        mixed = self.open * 0.634196 + self.open * (1 - 0.634196)
        second = ts_rank(decay_linear(correlation(self._ind(self.volume, "sector"), mixed, 17.4842), 6.92131), 13.4283)
        return -np.minimum(first, second)

    def alpha083(self) -> Array:
        ratio = _safe_div(self.high - self.low, sma(self.close, 5))
        numerator = rank(delay(ratio, 2)) * rank(rank(self.volume))
        denominator = _safe_div(ratio, self.vwap - self.close)
        return _safe_div(numerator, denominator)

    def alpha084(self) -> Array:
        return signed_power(ts_rank(self.vwap - ts_max(self.vwap, 15.3217), 20.7127), delta(self.close, 4.96796))

    def alpha085(self) -> Array:
        mixed = self.high * 0.876703 + self.close * (1 - 0.876703)
        base = rank(correlation(mixed, self._adv(30), 9.61331))
        exponent = rank(
            correlation(ts_rank((self.high + self.low) / 2, 3.70596), ts_rank(self.volume, 10.1595), 7.11408)
        )
        return signed_power(base, exponent)

    def alpha086(self) -> Array:
        left = ts_rank(correlation(self.close, ts_sum(self._adv(20), 14.7444), 6.00049), 20.4195)
        right = rank(self.open + self.close - self.vwap - self.open)
        return -_less(left, right)

    def alpha087(self) -> Array:
        mixed = self.close * 0.369701 + self.vwap * (1 - 0.369701)
        first = rank(decay_linear(delta(mixed, 1.91233), 2.65461))
        corr = np.abs(correlation(self._ind(self._adv(81), "industry"), self.close, 13.4132))
        second = ts_rank(decay_linear(corr, 4.89768), 14.4535)
        return -np.maximum(first, second)

    def alpha088(self) -> Array:
        first = rank(decay_linear(rank(self.open) + rank(self.low) - rank(self.high) - rank(self.close), 8.06882))
        corr = correlation(ts_rank(self.close, 8.44728), ts_rank(self._adv(60), 20.6966), 8.01266)
        second = ts_rank(decay_linear(corr, 6.65053), 2.61957)
        return np.minimum(first, second)

    def alpha089(self) -> Array:
        first = ts_rank(decay_linear(correlation(self.low, self._adv(10), 6.94279), 5.51607), 3.79744)
        second = ts_rank(decay_linear(delta(self._ind(self.vwap, "industry"), 3.48158), 10.1466), 15.3012)
        return first - second

    def alpha090(self) -> Array:
        base = rank(self.close - ts_max(self.close, 4.66719))
        exponent = ts_rank(correlation(self._ind(self._adv(40), "subindustry"), self.low, 5.38375), 3.21856)
        return -signed_power(base, exponent)

    def alpha091(self) -> Array:
        corr1 = correlation(self._ind(self.close, "industry"), self.volume, 9.74928)
        first = ts_rank(decay_linear(decay_linear(corr1, 16.398), 3.83219), 4.8667)
        second = rank(decay_linear(correlation(self.vwap, self._adv(30), 4.01303), 2.6809))
        return -(first - second)

    def alpha092(self) -> Array:
        condition = _less((self.high + self.low) / 2 + self.close, self.low + self.open)
        first = ts_rank(decay_linear(condition, 14.7221), 18.8683)
        second = ts_rank(decay_linear(correlation(rank(self.low), rank(self._adv(30)), 7.58555), 6.94024), 6.80584)
        return np.minimum(first, second)

    def alpha093(self) -> Array:
        first = ts_rank(
            decay_linear(correlation(self._ind(self.vwap, "industry"), self._adv(81), 17.4193), 19.848), 7.54455
        )
        mixed = self.close * 0.524434 + self.vwap * (1 - 0.524434)
        second = rank(decay_linear(delta(mixed, 2.77377), 16.2664))
        return _safe_div(first, second)

    def alpha094(self) -> Array:
        base = rank(self.vwap - ts_min(self.vwap, 11.5783))
        exponent = ts_rank(correlation(ts_rank(self.vwap, 19.6462), ts_rank(self._adv(60), 4.02992), 18.0926), 2.70756)
        return -signed_power(base, exponent)

    def alpha095(self) -> Array:
        left = rank(self.open - ts_min(self.open, 12.4105))
        corr = correlation(ts_sum((self.high + self.low) / 2, 19.1351), ts_sum(self._adv(40), 19.1351), 12.8742)
        right = ts_rank(signed_power(rank(corr), 5), 11.7584)
        return _less(left, right)

    def alpha096(self) -> Array:
        first = ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume), 3.83878), 4.16783), 8.38151)
        corr = correlation(ts_rank(self.close, 7.45404), ts_rank(self._adv(60), 4.13242), 3.65459)
        second = ts_rank(decay_linear(ts_argmax(corr, 12.6556), 14.0365), 13.4143)
        return -np.maximum(first, second)

    def alpha097(self) -> Array:
        mixed = self.low * 0.721001 + self.vwap * (1 - 0.721001)
        first = rank(decay_linear(delta(self._ind(mixed, "industry"), 3.3705), 20.4523))
        corr = correlation(ts_rank(self.low, 7.87871), ts_rank(self._adv(60), 17.255), 4.97547)
        second = ts_rank(decay_linear(ts_rank(corr, 18.5925), 15.7152), 6.71659)
        return -(first - second)

    def alpha098(self) -> Array:
        first = rank(decay_linear(correlation(self.vwap, ts_sum(self._adv(5), 26.4719), 4.58418), 7.18088))
        corr = correlation(rank(self.open), rank(self._adv(15)), 20.8187)
        second = rank(decay_linear(ts_rank(ts_argmin(corr, 8.62571), 6.95668), 8.07206))
        return first - second

    def alpha099(self) -> Array:
        first = rank(correlation(ts_sum((self.high + self.low) / 2, 19.8975), ts_sum(self._adv(60), 19.8975), 8.8136))
        second = rank(correlation(self.low, self.volume, 6.28259))
        return -_less(first, second)

    def alpha100(self) -> Array:
        position = _safe_div((self.close - self.low) - (self.high - self.close), self.high - self.low) * self.volume
        first = 1.5 * scale(self._ind(self._ind(rank(position), "subindustry"), "subindustry"))
        second = scale(
            self._ind(correlation(self.close, rank(self._adv(20)), 5) - rank(ts_argmin(self.close, 30)), "subindustry")
        )
        return -(first - second) * _safe_div(self.volume, self._adv(20))

    def alpha101(self) -> Array:
        return _safe_div(self.close - self.open, self.high - self.low + 0.001)


def factor_alpha101(
    market_df: pl.DataFrame | pl.LazyFrame,
    classifications_df: pl.DataFrame | pl.LazyFrame,
    alphas: Iterable[int] | None = None,
) -> pl.LazyFrame:
    """Calculate WorldQuant alphas as Toraniko style-factor scores.

    ``market_df`` must contain ``date``, ``symbol``, OHLCV, ``vwap``,
    ``returns`` (or Toraniko's ``asset_returns``) and ``market_cap``.  ``classifications_df`` may be static by
    symbol or dated, and must provide the classification levels required by
    the selected alphas.  Computing all alphas therefore requires ``sector``,
    ``industry`` and ``subindustry``.

    The returned columns are ``date``, ``symbol`` and zero-padded
    ``alpha001`` ... ``alpha101`` scores, directly consumable as ``style_df``
    by :func:`toraniko.model.estimate_factor_returns`.
    """
    if not isinstance(market_df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("`market_df` must be a Polars DataFrame or LazyFrame")
    if not isinstance(classifications_df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError("`classifications_df` must be a Polars DataFrame or LazyFrame")

    market = market_df.collect() if isinstance(market_df, pl.LazyFrame) else market_df
    classifications = (
        classifications_df.collect() if isinstance(classifications_df, pl.LazyFrame) else classifications_df
    )
    return_column = "returns" if "returns" in market.columns else "asset_returns"
    required_market = {"date", "symbol", *(_PRICE_COLUMNS[:-2] + ("market_cap",)), return_column}
    missing_market = sorted(required_market - set(market.columns))
    if missing_market:
        raise ValueError(f"`market_df` is missing columns: {', '.join(missing_market)}")
    if market.select(pl.struct("date", "symbol").is_duplicated().any()).item():
        raise ValueError("`market_df` must contain at most one row per date and symbol")
    if "symbol" not in classifications.columns:
        raise ValueError("`classifications_df` must contain a 'symbol' column")

    selected = list(range(1, 102)) if alphas is None else list(dict.fromkeys(alphas))
    invalid = [number for number in selected if not isinstance(number, int) or not 1 <= number <= 101]
    if invalid:
        raise ValueError(f"alpha numbers must be integers between 1 and 101: {invalid}")
    required_classes = {level for number in selected for level in _NEUTRALIZED_ALPHAS.get(number, ())}
    missing_classes = sorted(required_classes - set(classifications.columns))
    if missing_classes:
        raise ValueError(f"`classifications_df` is missing columns: {', '.join(missing_classes)}")

    dates = market["date"].unique().sort().to_list()
    symbols = market["symbol"].unique().sort().to_list()
    date_index, symbol_index = {v: i for i, v in enumerate(dates)}, {v: i for i, v in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    data = {name: np.full(shape, np.nan, dtype=float) for name in _PRICE_COLUMNS}
    present = np.zeros(shape, dtype=bool)
    input_columns = tuple(return_column if name == "returns" else name for name in _PRICE_COLUMNS)
    for row in market.select("date", "symbol", *input_columns).iter_rows(named=True):
        i, j = date_index[row["date"]], symbol_index[row["symbol"]]
        present[i, j] = True
        for name in _PRICE_COLUMNS:
            value = row[return_column if name == "returns" else name]
            data[name][i, j] = np.nan if value is None else float(value)

    class_data = {name: np.full(shape, None, dtype=object) for name in required_classes}
    dated = "date" in classifications.columns
    available_symbols = set(symbols)
    for row in classifications.select(*(("date",) if dated else ()), "symbol", *sorted(required_classes)).iter_rows(
        named=True
    ):
        if row["symbol"] not in available_symbols:
            continue
        js = symbol_index[row["symbol"]]
        if dated:
            if row["date"] not in date_index:
                continue
            indices = (date_index[row["date"]],)
        else:
            indices = range(len(dates))
        for i in indices:
            for name in required_classes:
                class_data[name][i, js] = row[name]
    for name, groups in class_data.items():
        if any(groups[i, j] is None for i, j in zip(*np.where(present))):
            raise ValueError(f"classification '{name}' is missing for one or more market observations")

    calculator = Alpha101(data, class_data)
    values = {number: calculator.alpha(number) for number in selected}
    row_dates, row_symbols = np.where(present)
    output = {
        "date": [dates[index] for index in row_dates],
        "symbol": [symbols[index] for index in row_symbols],
        **{f"alpha{number:03d}": values[number][row_dates, row_symbols] for number in selected},
    }
    return pl.DataFrame(output).with_columns(
        pl.col("date").cast(market.schema["date"]),
        pl.col("symbol").cast(market.schema["symbol"]),
    ).lazy()


def alpha_neutralization_levels() -> dict[int, tuple[str, ...]]:
    """Return the paper-mandated industry neutralization levels by alpha."""
    return dict(_NEUTRALIZED_ALPHAS)
