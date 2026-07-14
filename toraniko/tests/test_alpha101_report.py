import numpy as np
import polars as pl

from toraniko.alpha101_report import (
    _corr,
    analyze_alpha101,
    latest_alpha101_weights,
    plot_alpha101_pnl,
    plot_alpha101_weights,
    render_alpha101_report,
)


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


def test_latest_weights_are_dollar_neutral_and_plots_are_generated(tmp_path):
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{i}", "alpha001": float(i + date), "alpha002": float(9 - i + date)}
            for date in range(3)
            for i in range(10)
        ]
    )
    weights = latest_alpha101_weights(scores)
    totals = weights.group_by("alpha").agg(
        pl.col("weight").sum().alias("net"), pl.col("weight").abs().sum().alias("gross")
    )
    np.testing.assert_allclose(totals["net"], 0.0, atol=1e-12)
    np.testing.assert_allclose(totals["gross"], 1.0)

    summary = pl.DataFrame({"alpha": ["alpha001", "alpha002"], "sharpe": [1.0, -0.5]})
    daily = pl.DataFrame(
        [
            {"date": date, "alpha": alpha, "pnl": (0.001 if alpha == "alpha001" else -0.0005) * (date + 1)}
            for date in range(5)
            for alpha in ("alpha001", "alpha002")
        ]
    )
    pnl_path, weights_path = tmp_path / "pnl.png", tmp_path / "weights.png"
    plot_alpha101_pnl(summary, daily, pnl_path)
    plot_alpha101_weights(weights, weights_path)
    assert pnl_path.stat().st_size > 10_000
    assert weights_path.stat().st_size > 10_000
