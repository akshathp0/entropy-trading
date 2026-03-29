# Regime-Aware Mean Reversion Portfolio
 
**Author:** Akshath Pasam

**Stack:** Python (NumPy, Pandas) — no external quant libraries
 
---
 
## Overview
 
This strategy is a long-only risk-adjusted portfolio that combines **time-series mean reversion**, **Markov chain regime detection**, and **volatility-based position sizing** to dynamically adjust exposure based on market conditions.
 
The core finding: the strategy achieves **less than half the drawdown of SPY** (-15.6% vs -33.7%) from 2018 to 2026 by automatically reducing exposure during tenuous market conditions. During the 2008 financial crisis, average portfolio exposure dropped from 63% to 33%, and during COVID, it dropped from 63% to 51%. Volatility-based sizing and signal generation across various regimes drives this dynamic risk adjustment.
 
Shannon entropy was tested as both a signal modifier and portfolio allocation mechanism in order to act as a third risk-adjustor during regime shifts. It had a minimal performance impact (~0.01 Sharpe improvement), indicating that the mean reversion signal and volatility sizing are the main features managing risk.
 
---
 
## Methodology
 
### 1. Universe Selection
 
An initial universe of 28 liquid ETFs spanning U.S. equities, sectors, international markets, and commodities was selected, with the following pipeline applied in these assets:
 
1. **Sharpe filter:** Each asset is run through the full strategy pipeline in-sample (2008–2018). Assets with strategy Sharpe ratio > 0.7 are retained.
2. **PCA:** Principal component analysis identifies independent risk factors. The most positive and most negative loading asset from each principal component is selected to maximize factor diversity.
3. **Correlation filter:** Remaining pairwise correlations above 0.85 are flagged. The asset with lower PCA loading is replaced by the next best candidate from the same side of that component.
 
**Final universe (8 assets):** VUG, IBB, XLV, XLU, XLP, VNQ, XLF, SPY. These assets represent equity, biotech, healthcare, utilities, real estate, and a variety of other sectors, representing a diverse ETF universe suitable for a portfolio.
 
### 2. Regime Detection
 
Each asset is given a discrete market regime on each trading day based on two features (both lagged by one day to avoid look-ahead bias):
 
- **Realized volatility** (60-day rolling) — classified as high or low relative to a rolling percentile threshold
- **T-statistic of returns** (200-day) — classified as positive or negative
 
This produces **4 regime states**: low vol / bull, low vol / bear, high vol / bull, high vol / bear.
 
A **Markov chain** transition matrix is estimated over a rolling 750-day window. **Shannon entropy** of the transition distribution quantifies regime predictability at each timestep:
 
$$H(X) = -\sum_{i} p_i \log(p_i)$$
 
Low entropy indicates a stable, predictable regime, and high entropy indicates disorder or regime transition.
 
### 3. Signal Generation
 
The **mean reversion signal** is computed for each asset using a z-score of log returns relative to its rolling mean. This z-score is then passed through a sigmoid function to produce a continuous signal between 0 and 1.
 
### 4. Volatility-Based Position Sizing
 
Positions for individual assets are scaled by the ratio of a target volatility (mean of rolling realized vol used in the Markov chain) to the current realized volatility, calculated as such:
 
```
sized_signal = mr_signal × (target_vol / current_vol)
```
 
This position scaling reduces exposure during high-volatility markets and increases it during calm markets, which becomes the primary mechanism behind the strategy's drawdown reduction.
 
### 5. Portfolio Construction
 
Assets are dynamically sized through their individual entropy values. Daily portfolio return is the aggregate of individual asset strategy returns, compounded to produce the portfolio equity curve.
 
---
 
## Results
 
### In-Sample (2008–2018)
 
| Metric             | Strategy | SPY Benchmark |
|--------------------|----------|---------------|
| Annualized Return  | 12.04%   | ~8.7%         |
| Sharpe Ratio       | 1.102    | ~0.55         |
| Max Drawdown       | -14.66%  | ~-55%         |
 
### Out-of-Sample (2018–2025)
 
| Metric             | Strategy   | SPY Benchmark |
|--------------------|------------|---------------|
| Annualized Return  | 4.29%      | 11.07%        |
| Sharpe Ratio       | 0.632      | 0.774         |
| Sortino Ratio      | 0.751      | 0.944         |
| Calmar Ratio       | 0.275      | 0.328         |
| **Max Drawdown**   | **-15.58%**| **-33.72%**   |
 
### Dynamic Exposure Analysis
 
The strategy's average exposure is 63%. During crises, it drops automatically:
 
| Period                  | Average Exposure |
|-------------------------|-----------------|
| Full period             | 63.2%           |
| 2008 Financial Crisis   | 32.9%           |
| 2020 COVID Crash        | 50.8%           |
 
### Monte Carlo Analysis (Block Bootstrap)
 
A block bootstrap simulation (1000 trials, 20-day blocks) shows that across random arrangements of segmented market conditions, the strategy consistently outperforms SPY in negative-return scenarios and underperforms in strong bull markets. This confirms the strategy functions primarily as a risk management framework rather than an alpha-generation strategy.
 
### Parameter Robustness
 
A grid search over entropy scaling (gamma), entropy application mode, and rolling window length showed Sharpe ratio variations of ~0.02 across all parameter combinations, indicating the strategy is robust to parameter choices and not overfit to specific settings.
 
### Entropy Findings
 
Entropy-weighted portfolio allocation was tested against equal-weight allocation. The difference was negligible (~0.01 Sharpe), suggesting that entropy captures regime information that is already reflected in the volatility sizing mechanism. This is reported as a negative research finding.
 
---
 
## Repository Structure
 
```
.
├── data/                   # ETF price data loading
├── features/               # T-stat, volatility, z-score computation
├── regime/                 # Markov chain, state labeling, entropy
├── backtests/              # Signal generation, blending, backtesting
├── pipeline/               # Run pipeline, portfolio aggregation, grid search, Monte Carlo
├── evaluation/             # Metrics computation and plotting
├── universe/               # PCA, correlation filtering, Sharpe screening
├── results/
│   ├── universe_selection/ # Sharpe rankings, final asset list
│   ├── assets/             # Per-asset equity curves and metrics
│   ├── in_sample_portfolio/
|   ├── no_entropy_portfolio/
|   ├── out_of_sample_portfolio/
|   ├── spy_metrics.csv # Comparison metrics
│   └── grid_search_results.csv # Grid search configurations and final Sharpes
├── notebooks/              # Analysis notebooks
├── config.yml              # All parameters and settings
└── README.md
```
 
---
 
## Setup & Usage
 
**Requirements:** Python 3.12+, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, PyYAML
 
```bash
git clone https://github.com/akshathpasam/entropy-trading.git
cd entropy-trading
pip install -r requirements.txt
```
 
Pipeline is run through Jupyter notebooks in `notebooks/`.
 
---
 
## Limitations & Future Work
 
- **Transaction costs:** Backtest does not incorporate bid-ask spreads or slippage
- **Flash crash sensitivity:** The regime detection framework responds better to prolonged crises (2008) than sudden crashes (COVID), due to the nature of Markov chains and rolling window estimation approach
- **Long-only constraint:** A long/short extension could capture alpha during bear regimes (tested in-sample, results were mixed)
- **Momentum overlay:** Regime-conditional momentum (active during low-entropy bull markets) is a natural extension that could address the bull market underperformance
- **Alternative entropy measures:** Rényi entropy or conditional entropy could provide more nuanced regime characterization
 
---
 
## Contact
 
**Akshath Pasam**
[akshath@pasam.com] · [GitHub](https://github.com/akshathp0)
