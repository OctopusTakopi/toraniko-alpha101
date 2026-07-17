# Alpha101 paper figures remade with current data

Definitions follow [*101 Formulaic Alphas*](https://arxiv.org/pdf/1601.00991), Section 3.
Turnover is gross daily dollars traded per dollar of gross investment; cents-per-share is
100 times mean daily PnL divided by mean daily shares traded (buys plus sells).

- Alphas: 101
- Pairwise return correlations: 5,050
- Mean / median correlation: 0.1942 / 0.1711
- Positive-return alphas used in log-return panels: 74/101
- Positive-CPS alphas used in the log-CPS panel: 74/101

Nonpositive return and CPS observations are excluded only from panels where the paper takes a natural log;
signals are not flipped and absolute values are not substituted.

## Regressions

| Model | Term | Estimate | Std. error | t-stat | n | R² | Adjusted R² | F-stat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| log_return_on_log_volatility | intercept | 1.756573 | 1.298361 | 1.353 | 74 | 0.5095 | 0.5027 | 74.789 |
| log_return_on_log_volatility | log_volatility | 1.857007 | 0.214731 | 8.648 | 74 | 0.5095 | 0.5027 | 74.789 |
| log_return_on_log_volatility_and_turnover | intercept | 1.807576 | 1.486823 | 1.216 | 74 | 0.5095 | 0.4957 | 36.880 |
| log_return_on_log_volatility_and_turnover | log_volatility | 1.866027 | 0.249868 | 7.468 | 74 | 0.5095 | 0.4957 | 36.880 |
| log_return_on_log_volatility_and_turnover | log_turnover | -0.016156 | 0.224278 | -0.072 | 74 | 0.5095 | 0.4957 | 36.880 |
| correlation_on_turnover_tensors | intercept | 0.194626 | 0.003228 | 60.297 | 5050 | 0.0331 | 0.0327 | 86.377 |
| correlation_on_turnover_tensors | turnover_sum | 0.009170 | 0.004939 | 1.857 | 5050 | 0.0331 | 0.0327 | 86.377 |
| correlation_on_turnover_tensors | turnover_product | 0.195946 | 0.015070 | 13.002 | 5050 | 0.0331 | 0.0327 | 86.377 |
| log_volatility_on_log_turnover | intercept | -5.959443 | 0.047920 | -124.362 | 101 | 0.2140 | 0.2061 | 26.955 |
| log_volatility_on_log_turnover | log_turnover | 0.474437 | 0.091382 | 5.192 | 101 | 0.2140 | 0.2061 | 26.955 |

## Figures

![Figure 1: empirical distributions](paper_figure1_distributions.png)

![Figure 2: return vs volatility](paper_figure2_return_vs_volatility.png)

![Figure 3: turnover vs demeaned correlation](paper_figure3_turnover_vs_correlation.png)

![Figure 4: volatility vs turnover](paper_figure4_volatility_vs_turnover.png)
