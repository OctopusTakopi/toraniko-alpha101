# Alpha101 paper figures remade with current data

Definitions follow [*101 Formulaic Alphas*](https://arxiv.org/pdf/1601.00991), Section 3.
Turnover is gross daily dollars traded per dollar of gross investment; cents-per-share is
100 times mean daily PnL divided by mean daily shares traded (buys plus sells).

- Alphas: 101
- Pairwise return correlations: 5,050
- Mean / median correlation: 0.2106 / 0.1945
- Positive-return alphas used in log-return panels: 78/101
- Positive-CPS alphas used in the log-CPS panel: 78/101

Nonpositive return and CPS observations are excluded only from panels where the paper takes a natural log;
signals are not flipped and absolute values are not substituted.

## Regressions

| Model | Term | Estimate | Std. error | t-stat | n | R² | Adjusted R² | F-stat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| log_return_on_log_volatility | intercept | 2.108066 | 1.424170 | 1.480 | 78 | 0.4641 | 0.4571 | 65.818 |
| log_return_on_log_volatility | log_volatility | 1.909751 | 0.235398 | 8.113 | 78 | 0.4641 | 0.4571 | 65.818 |
| log_return_on_log_volatility_and_turnover | intercept | 2.321992 | 1.605395 | 1.446 | 78 | 0.4647 | 0.4505 | 32.558 |
| log_return_on_log_volatility_and_turnover | log_volatility | 1.947434 | 0.268993 | 7.240 | 78 | 0.4647 | 0.4505 | 32.558 |
| log_return_on_log_volatility_and_turnover | log_turnover | -0.064427 | 0.218085 | -0.295 | 78 | 0.4647 | 0.4505 | 32.558 |
| correlation_on_turnover_tensors | intercept | 0.211066 | 0.003332 | 63.352 | 5050 | 0.0375 | 0.0371 | 98.316 |
| correlation_on_turnover_tensors | turnover_sum | 0.013171 | 0.005167 | 2.549 | 5050 | 0.0375 | 0.0371 | 98.316 |
| correlation_on_turnover_tensors | turnover_product | 0.220099 | 0.015975 | 13.778 | 5050 | 0.0375 | 0.0371 | 98.316 |
| log_volatility_on_log_turnover | intercept | -6.003660 | 0.038804 | -154.716 | 101 | 0.2014 | 0.1934 | 24.971 |
| log_volatility_on_log_turnover | log_turnover | 0.374794 | 0.075003 | 4.997 | 101 | 0.2014 | 0.1934 | 24.971 |

## Figures

![Figure 1: empirical distributions](paper_figure1_distributions.png)

![Figure 2: return vs volatility](paper_figure2_return_vs_volatility.png)

![Figure 3: turnover vs demeaned correlation](paper_figure3_turnover_vs_correlation.png)

![Figure 4: volatility vs turnover](paper_figure4_volatility_vs_turnover.png)
