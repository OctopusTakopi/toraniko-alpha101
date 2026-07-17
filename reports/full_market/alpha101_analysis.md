# Full-Market WorldQuant Alpha101 Analysis

Dataset: current-constituent S&P Composite 1500 snapshot (1506 securities; 1501 with Yahoo OHLCV; 1501 with historical shares), analysis 2023-01-03 to 2026-07-16, warm-up from 2021-12-30. Yahoo adjusted daily OHLCV; typical price proxies VWAP; historical shares and raw closes form market cap. Current GICS classifications and membership introduce survivorship bias.

Method: signals at *t*, next-session returns at *t+1*, equal-weight top/bottom quintiles, 50% long and 50% short. Boundary ties remain together; constant signals hold no position. Returns exclude costs.

Alphas analyzed: 101

## Highest Sharpe alphas

| Alpha | Ann. return | Ann. vol | Sharpe | Max DD | Rank IC | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| alpha021 | 6.41% | 3.03% | 2.120 | -2.10% | 0.0130 | 39.86% |
| alpha043 | 6.38% | 4.22% | 1.512 | -3.29% | 0.0134 | 54.50% |
| alpha042 | 8.47% | 5.84% | 1.451 | -4.27% | 0.0026 | 42.34% |
| alpha032 | 6.37% | 4.59% | 1.387 | -3.63% | 0.0122 | 19.38% |
| alpha011 | 4.77% | 3.56% | 1.339 | -4.06% | 0.0073 | 56.72% |
| alpha007 | 8.74% | 6.61% | 1.322 | -3.82% | 0.0098 | 45.40% |
| alpha079 | 2.42% | 2.00% | 1.208 | -2.68% | 0.0051 | 29.32% |
| alpha037 | 5.58% | 4.67% | 1.194 | -3.79% | 0.0105 | 54.97% |
| alpha030 | 5.42% | 4.64% | 1.168 | -3.56% | 0.0095 | 43.11% |
| alpha096 | 1.87% | 1.62% | 1.153 | -2.56% | 0.0035 | 34.31% |
| alpha035 | 7.88% | 6.89% | 1.143 | -5.13% | 0.0129 | 60.65% |
| alpha054 | 5.00% | 4.63% | 1.082 | -8.07% | 0.0113 | 79.25% |
| alpha024 | 5.57% | 5.15% | 1.081 | -4.17% | 0.0120 | 22.36% |
| alpha049 | 5.43% | 5.53% | 0.983 | -5.64% | 0.0083 | 55.44% |
| alpha047 | 6.00% | 6.22% | 0.965 | -5.61% | 0.0085 | 38.89% |
| alpha071 | 2.76% | 2.93% | 0.943 | -2.63% | 0.0048 | 41.61% |
| alpha038 | 6.09% | 6.51% | 0.935 | -5.93% | 0.0141 | 65.11% |
| alpha005 | 4.55% | 5.12% | 0.888 | -4.04% | 0.0123 | 47.55% |
| alpha057 | 4.38% | 5.01% | 0.874 | -4.12% | 0.0088 | 69.28% |
| alpha053 | 3.60% | 4.20% | 0.857 | -8.90% | 0.0103 | 80.18% |

## Lowest Sharpe alphas

| Alpha | Ann. return | Ann. vol | Sharpe | Max DD | Rank IC | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| alpha088 | -6.43% | 6.78% | -0.948 | -22.78% | -0.0002 | 26.23% |
| alpha023 | -13.56% | 14.51% | -0.934 | -45.10% | 0.0053 | 55.87% |
| alpha001 | -4.21% | 4.64% | -0.909 | -18.77% | -0.0087 | 51.19% |
| alpha101 | -3.45% | 5.60% | -0.615 | -17.25% | -0.0103 | 79.39% |
| alpha045 | -1.14% | 2.24% | -0.509 | -7.15% | 0.0020 | 62.24% |
| alpha099 | -0.83% | 1.70% | -0.486 | -4.59% | 0.0005 | 17.51% |
| alpha058 | -0.80% | 1.78% | -0.451 | -4.52% | 0.0003 | 43.88% |
| alpha081 | -0.65% | 1.47% | -0.445 | -3.33% | 0.0008 | 19.08% |
| alpha012 | -1.20% | 2.91% | -0.411 | -7.47% | 0.0031 | 67.95% |
| alpha026 | -1.12% | 3.22% | -0.347 | -5.32% | 0.0029 | 27.63% |
| alpha073 | -1.25% | 4.81% | -0.260 | -10.90% | -0.0038 | 46.66% |
| alpha080 | -0.60% | 2.53% | -0.238 | -5.83% | -0.0035 | 50.04% |
| alpha089 | -0.55% | 2.44% | -0.225 | -6.15% | 0.0006 | 35.94% |
| alpha062 | -0.49% | 2.32% | -0.209 | -5.14% | 0.0008 | 29.40% |
| alpha072 | -0.52% | 2.70% | -0.191 | -4.89% | 0.0024 | 23.93% |
| alpha085 | -0.46% | 2.65% | -0.172 | -5.33% | 0.0011 | 29.77% |
| alpha075 | -0.27% | 1.59% | -0.168 | -3.51% | 0.0037 | 23.83% |
| alpha046 | -2.66% | 16.31% | -0.163 | -29.29% | 0.0056 | 39.30% |
| alpha091 | -0.33% | 2.71% | -0.123 | -4.39% | -0.0020 | 35.12% |
| alpha097 | -0.30% | 2.60% | -0.115 | -4.13% | -0.0034 | 25.93% |

## Cross-alpha diagnostics

- Median annual return: 1.61%
- Median Sharpe: 0.518
- Median rank IC: 0.0039
- Median one-sided turnover: 39.30%

## Charts

![Alpha101 cumulative PnL](alpha101_pnl.png)

![Alpha101 weights](alpha101_weights.png)
