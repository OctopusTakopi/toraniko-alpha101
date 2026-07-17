import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).parents[2] / "examples" / "alpha101_full_market" / "run.py"
    spec = importlib.util.spec_from_file_location("alpha101_full_market", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yahoo_symbol_normalization():
    module = _load_module()
    assert module.normalize_yahoo_symbol(" brk.b ") == "BRK-B"


def test_score_cache_formula_version(tmp_path):
    module = _load_module()
    metadata = tmp_path / "scores.json"
    assert module._score_cache_version(metadata) is None

    metadata.write_text('{"formula_version": "old"}')
    assert module._score_cache_version(metadata) == "old"

    metadata.write_text("not json")
    assert module._score_cache_version(metadata) is None

    metadata.write_text("[]")
    assert module._score_cache_version(metadata) is None


def test_yahoo_download_conversion_and_point_in_time_market_cap():
    module = _load_module()
    dates = pd.date_range("2024-01-02", periods=30)
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["TEST"]],
        names=["Price", "Ticker"],
    )
    values = np.array(
        [[10.0 + day, 12.0 + day, 9.0 + day, 11.0 + day, 5.5 + 0.5 * day, 100.0 + day] for day in range(30)]
    )
    download = pd.DataFrame(values, index=dates, columns=columns)
    prices = module._split_yahoo_download(download, ["TEST"])["TEST"]
    np.testing.assert_allclose(prices["close"].iloc[:3], [5.5, 6.0, 6.5])
    np.testing.assert_allclose(prices["open"].iloc[:3], [5.0, 5.5, 6.0])
    np.testing.assert_allclose(prices["execution_price"].iloc[:3], [11.0, 12.0, 13.0])

    shares = {"TEST": pd.Series([100.0, 120.0], index=[dates[0], dates[2]])}
    market = module.build_market_data(prices, shares)
    np.testing.assert_allclose(market["market_cap"].iloc[:3], [1100.0, 1200.0, 1560.0])
    np.testing.assert_allclose(market["returns"].iloc[1:3], [6.0 / 5.5 - 1, 6.5 / 6.0 - 1])
