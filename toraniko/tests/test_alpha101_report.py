import numpy as np
import polars as pl

from toraniko.alpha101_report import (
    _corr,
    analyze_alpha101,
    analyze_alpha101_paper,
    latest_alpha101_weights,
    plot_alpha101_paper_figures,
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
    prices = pl.DataFrame(
        [{"date": row["date"], "symbol": row["symbol"], "close": 10.0} for row in return_rows]
    )
    summary, daily = analyze_alpha101(pl.DataFrame(score_rows), pl.DataFrame(return_rows), prices_df=prices)
    assert summary.height == 1
    assert summary["sharpe"][0] > 5
    assert summary["mean_rank_ic"][0] > 0.8
    assert summary["paper_turnover"][0] > 0
    assert summary["cents_per_share"][0] > 0
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


def test_empty_analysis_renders_without_numeric_formatting_errors():
    scores = pl.DataFrame(
        [{"date": 1, "symbol": f"S{symbol}", "alpha001": float(symbol)} for symbol in range(10)]
    )
    returns = scores.select("date", "symbol").with_columns(pl.lit(0.0).alias("asset_returns"))

    summary, daily = analyze_alpha101(scores, returns)
    report = render_alpha101_report(summary)

    assert summary.is_empty()
    assert daily.is_empty()
    assert "Alphas analyzed: 0" in report
    assert "No analyzable forward-return observations" in report


def test_paper_turnover_and_cents_per_share_follow_section_three_definitions():
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "alpha001": float(symbol)}
            for date in range(3)
            for symbol in range(10)
        ]
    )
    returns = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"S{symbol}",
                "asset_returns": 0.01 if symbol >= 8 else (-0.01 if symbol < 2 else 0.0),
            }
            for date in range(3)
            for symbol in range(10)
        ]
    )
    prices = returns.select("date", "symbol").with_columns(pl.lit(10.0).alias("close"))
    summary, daily = analyze_alpha101(scores, returns, prices_df=prices)
    np.testing.assert_allclose(daily["dollar_volume"], [1.0, 0.0])
    np.testing.assert_allclose(daily["shares_traded"], [0.1, 0.0])
    np.testing.assert_allclose(summary["paper_turnover"], 0.5)
    np.testing.assert_allclose(summary["cents_per_share"], 20.0)


def test_portfolio_membership_does_not_use_future_return_availability():
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "alpha001": float(symbol)}
            for date in range(3)
            for symbol in range(10)
        ]
    )
    returns = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"S{symbol}",
                "asset_returns": (
                    np.nan
                    if date == 1 and symbol == 9
                    else (0.01 if symbol >= 8 else (-0.01 if symbol < 2 else 0.0))
                ),
            }
            for date in range(3)
            for symbol in range(10)
        ]
    )
    _, daily = analyze_alpha101(scores, returns)
    np.testing.assert_allclose(daily["pnl"][0], 0.0075)
    np.testing.assert_allclose(daily["return_coverage"][0], 0.9)


def test_sparse_forward_returns_are_zero_filled_without_suppressing_pnl():
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "alpha001": float(symbol)}
            for date in range(2)
            for symbol in range(10)
        ]
    )
    returns = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"S{symbol}",
                "asset_returns": (-0.01 if date == 1 and symbol < 2 else (0.0 if date == 1 and symbol < 4 else np.nan)),
            }
            for date in range(2)
            for symbol in range(10)
        ]
    )

    _, daily = analyze_alpha101(scores, returns)

    assert daily.height == 1
    np.testing.assert_allclose(daily["pnl"], [0.005])
    np.testing.assert_allclose(daily["return_coverage"], [0.4])
    assert daily["ic"].is_nan().all()
    assert daily["rank_ic"].is_nan().all()


def test_constant_scores_do_not_create_symbol_order_portfolios():
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "alpha001": 1.0}
            for date in range(3)
            for symbol in range(10)
        ]
    )
    returns = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "asset_returns": (symbol - 4.5) / 100}
            for date in range(3)
            for symbol in range(10)
        ]
    )

    _, daily = analyze_alpha101(scores, returns)
    weights = latest_alpha101_weights(scores)

    np.testing.assert_allclose(daily["pnl"], 0.0)
    np.testing.assert_allclose(daily["turnover"], 0.0)
    np.testing.assert_allclose(weights["weight"], 0.0)


def test_tail_portfolios_include_boundary_ties_without_splitting_them():
    values = [0.0] * 3 + [1.0] * 4 + [2.0] * 3
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"S{symbol}", "alpha001": values[symbol]}
            for date in range(2)
            for symbol in range(10)
        ]
    )

    weights = latest_alpha101_weights(scores)
    active = weights.filter(pl.col("weight") != 0)

    assert active.height == 6
    np.testing.assert_allclose(active.filter(pl.col("weight") < 0)["weight"], -1 / 6)
    np.testing.assert_allclose(active.filter(pl.col("weight") > 0)["weight"], 1 / 6)


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


def test_paper_analysis_and_all_four_figures_are_generated(tmp_path):
    alphas = [f"alpha{i:03d}" for i in range(1, 5)]
    summary = pl.DataFrame(
        {
            "alpha": alphas,
            "annual_return": [0.05, 0.08, 0.03, 0.11],
            "annual_volatility": [0.12, 0.15, 0.10, 0.17],
            "sharpe": [0.42, 0.53, 0.30, 0.65],
            "paper_turnover": [0.3, 0.5, 0.8, 1.1],
            "cents_per_share": [0.2, 0.4, 0.3, 0.5],
        }
    )
    daily = pl.DataFrame(
        [
            {
                "date": date,
                "alpha": alpha,
                "pnl": 0.001 * np.sin(date / (index + 2)) + 0.0001 * (index + 1),
            }
            for date in range(40)
            for index, alpha in enumerate(alphas)
        ]
    )
    metrics, pairs, regressions = analyze_alpha101_paper(summary, daily)
    assert metrics.height == 4
    assert pairs.height == 6
    assert regressions["model"].n_unique() == 4
    paths = plot_alpha101_paper_figures(metrics, pairs, tmp_path)
    assert set(paths) == {"figure1", "figure2", "figure3", "figure4"}
    assert all(path.stat().st_size > 10_000 for path in paths.values())


def test_large_universe_weight_plot_aggregates_by_sector(tmp_path):
    symbols = [f"S{i:03d}" for i in range(100)]
    weights = pl.DataFrame(
        [
            {
                "date": 1,
                "alpha": alpha,
                "symbol": symbol,
                "weight": (1 if index % 2 else -1) / 100,
            }
            for alpha in ("alpha001", "alpha002")
            for index, symbol in enumerate(symbols)
        ]
    )
    classifications = pl.DataFrame(
        {"symbol": symbols, "sector": [f"sector{index % 5}" for index in range(len(symbols))]}
    )
    output = tmp_path / "large_weights.png"
    plot_alpha101_weights(weights, output, classifications)
    assert output.stat().st_size > 10_000
