"""Tests for the WorldQuant 101 implementation."""

import numpy as np
import polars as pl
import pytest

from toraniko.alpha101 import (
    _window,
    alpha_neutralization_levels,
    correlation,
    factor_alpha101,
    indneutralize,
    rank,
    ts_rank,
)


def _market_panel(days: int = 320, symbols: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(7)
    dates = np.repeat(np.arange(days), symbols)
    names = np.tile([f"S{i}" for i in range(symbols)], days)
    close = (100 + np.cumsum(rng.normal(0, 0.8, (days, symbols)), axis=0)).ravel()
    open_ = close * (1 + rng.normal(0, 0.002, close.size))
    spread = rng.uniform(0.1, 1.5, close.size)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(13, 0.25, close.size)
    returns = np.vstack(
        [
            np.full((1, symbols), np.nan),
            np.diff(close.reshape(days, symbols), axis=0) / close.reshape(days, symbols)[:-1],
        ]
    ).ravel()
    market = pl.DataFrame(
        {
            "date": dates,
            "symbol": names,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": (open_ + high + low + close) / 4,
            "returns": returns,
            "market_cap": close * rng.uniform(1e6, 2e6, close.size),
        }
    )
    classes = pl.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(symbols)],
            "sector": [f"sec{i // 3}" for i in range(symbols)],
            "industry": [f"ind{i // 2}" for i in range(symbols)],
            "subindustry": [f"sub{i // 2}" for i in range(symbols)],
        }
    )
    return market, classes


def test_fractional_windows_round_to_nearest_day():
    assert _window(3.49) == 3
    assert _window(3.50) == 4


def test_cross_sectional_and_time_series_rank_semantics():
    values = np.array([[3.0, 1.0, 2.0], [4.0, 1.0, 3.0], [5.0, 0.0, 2.0]])
    np.testing.assert_allclose(rank(values)[0], [1.0, 1 / 3, 2 / 3])
    np.testing.assert_allclose(ts_rank(values, 3)[-1], [1.0, 1 / 3, 0.5])


def test_correlation_uses_full_trailing_window():
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    result = correlation(x, x, 3)
    assert np.isnan(result[:2]).all()
    np.testing.assert_allclose(result[2:], 1.0)


def test_correlation_of_constant_defined_window_is_zero():
    constant = np.ones((4, 2))
    result = correlation(constant, constant, 3)
    np.testing.assert_allclose(result[2:], 0.0)


def test_indneutralize_is_group_demeaning():
    values = np.array([[1.0, 3.0, 10.0, 14.0]])
    groups = np.array([["a", "a", "b", "b"]], dtype=object)
    result = indneutralize(values, groups)
    np.testing.assert_allclose(result, [[-1.0, 1.0, -2.0, 2.0]])
    np.testing.assert_allclose(result[0, :2].sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(result[0, 2:].sum(), 0.0, atol=1e-12)


def test_all_101_alphas_integrate_with_toraniko_long_form():
    market, classes = _market_panel()
    result = factor_alpha101(market, classes).collect()
    assert result.shape == (market.height, 103)
    assert result.columns[2:] == [f"alpha{i:03d}" for i in range(1, 102)]
    expected_101 = (market["close"] - market["open"]) / (market["high"] - market["low"] + 0.001)
    np.testing.assert_allclose(result["alpha101"], expected_101)
    assert result.select(pl.exclude("date", "symbol").is_finite().any()).row(0).count(True) == 101


def test_neutralized_alpha_requires_its_paper_classification():
    market, classes = _market_panel(20)
    with pytest.raises(ValueError, match="subindustry"):
        factor_alpha101(market, classes.drop("subindustry"), alphas=[48])


def test_toraniko_asset_returns_alias_is_supported():
    market, classes = _market_panel(10)
    result = factor_alpha101(market.rename({"returns": "asset_returns"}), classes, alphas=[101]).collect()
    assert result.columns == ["date", "symbol", "alpha101"]


def test_neutralization_manifest_covers_all_paper_formulas():
    assert set(alpha_neutralization_levels()) == {
        48,
        58,
        59,
        63,
        67,
        69,
        70,
        76,
        79,
        80,
        82,
        87,
        89,
        90,
        91,
        93,
        97,
        100,
    }
