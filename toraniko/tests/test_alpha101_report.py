import numpy as np
import polars as pl

from toraniko.alpha101_report import _corr, analyze_alpha101, render_alpha101_report


def test_analysis_lags_returns_and_reports_expected_direction():
    rng = np.random.default_rng(11)
    symbols = [f"S{i}" for i in range(20)]
    score_rows, return_rows = [], []
    previous_signal = np.zeros(len(symbols))
    for date in range(40):
        signal = rng.normal(size=len(symbols))
        realized = 0.01 * previous_signal + rng.normal(scale=0.001, size=len(symbols))
        for i, symbol in enumerate(symbols):
            score_rows.append({"date": date, "symbol": symbol, "alpha001": signal[i]})
            return_rows.append({"date": date, "symbol": symbol, "asset_returns": realized[i]})
        previous_signal = signal
    summary, daily = analyze_alpha101(pl.DataFrame(score_rows), pl.DataFrame(return_rows))
    assert summary.height == 1
    assert summary["sharpe"][0] > 5
    assert summary["mean_rank_ic"][0] > 0.8
    assert daily["date"].min() == 0


def test_markdown_report_contains_performance_sections():
    summary = pl.DataFrame(
        {
            "alpha": ["alpha001"],
            "annual_return": [0.1],
            "annual_volatility": [0.2],
            "sharpe": [0.5],
            "max_drawdown": [-0.1],
            "mean_rank_ic": [0.02],
            "mean_turnover": [0.4],
        }
    )
    report = render_alpha101_report(summary)
    assert "Highest Sharpe" in report
    assert "alpha001" in report


def test_ic_correlation_is_stable_for_large_formula_values():
    x = np.array([1e250, 2e250, 3e250])
    np.testing.assert_allclose(_corr(x, x), 1.0)
