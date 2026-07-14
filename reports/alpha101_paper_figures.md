# Alpha101 paper figures remade with current data

Definitions follow [*101 Formulaic Alphas*](https://arxiv.org/pdf/1601.00991), Section 3. 
Turnover is gross daily dollars traded per dollar of gross investment; cents-per-share is 
100 times mean daily PnL divided by mean daily shares traded (buys plus sells).

- Alphas: 101
- Pairwise return correlations: 5,050
- Mean / median correlation: 0.1149 / 0.0995
- Positive-return alphas used in log-return panels: 65/101
- Positive-CPS alphas used in the log-CPS panel: 65/101

Nonpositive return and CPS observations are excluded only from panels where the paper takes a natural log; 
signals are not flipped and absolute values are not substituted.

## Regressions

| Model | Term | Estimate | Std. error | t-stat | n | R² | Adjusted R² | F-stat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| log_return_on_log_volatility | intercept | 9.146493 | 8.556425 | 1.069 | 65 | 0.0642 | 0.0493 | 4.319 |
| log_return_on_log_volatility | log_volatility | 3.781312 | 1.819427 | 2.078 | 65 | 0.0642 | 0.0493 | 4.319 |
| log_return_on_log_volatility_and_turnover | intercept | 9.044340 | 8.711452 | 1.038 | 65 | 0.0643 | 0.0341 | 2.129 |
| log_return_on_log_volatility_and_turnover | log_volatility | 3.759198 | 1.853060 | 2.029 | 65 | 0.0643 | 0.0341 | 2.129 |
| log_return_on_log_volatility_and_turnover | log_turnover | 0.027154 | 0.326049 | 0.083 | 65 | 0.0643 | 0.0341 | 2.129 |
| correlation_on_turnover_tensors | intercept | 0.115377 | 0.002019 | 57.159 | 5050 | 0.0832 | 0.0828 | 228.884 |
| correlation_on_turnover_tensors | turnover_sum | -0.001349 | 0.003688 | -0.366 | 5050 | 0.0832 | 0.0828 | 228.884 |
| correlation_on_turnover_tensors | turnover_product | 0.286805 | 0.013407 | 21.393 | 5050 | 0.0832 | 0.0828 | 228.884 |
| log_volatility_on_log_turnover | intercept | -4.705716 | 0.007682 | -612.561 | 101 | 0.0783 | 0.0690 | 8.413 |
| log_volatility_on_log_turnover | log_turnover | 0.054570 | 0.018813 | 2.901 | 101 | 0.0783 | 0.0690 | 8.413 |

## Figures

![Figure 1: empirical distributions](paper_figure1_distributions.png)

![Figure 2: return vs volatility](paper_figure2_return_vs_volatility.png)

![Figure 3: turnover vs demeaned correlation](paper_figure3_turnover_vs_correlation.png)

![Figure 4: volatility vs turnover](paper_figure4_volatility_vs_turnover.png)
