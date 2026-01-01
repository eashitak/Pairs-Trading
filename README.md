# ML-Enhanced Pairs Trading Strategy

## Overview

This repository implements a **machine learning–enhanced pairs trading strategy** that combines classical statistical arbitrage with unsupervised learning techniques for pair selection and robust backtesting.

The implementation is inspired by:
- Classical pairs trading literature (Gatev et al., 2006)
- Sarmento & Horta (2020): *Enhancing a Pairs Trading Strategy with the Application of Machine Learning*

The strategy focuses on identifying statistically related asset pairs, constructing mean-reverting spreads, and generating systematic trading signals while accounting for transaction costs and risk metrics.

---

## Strategy Intuition

Pairs trading is a **market-neutral relative-value strategy**.

Instead of predicting market direction:
- One asset is bought (long)
- The other is sold (short)
- Profit is generated when the price relationship reverts to its historical equilibrium

Because returns are driven by **relative price movements**, overall market trends have limited impact on strategy performance.

---

## Methodology

### 1. Data Collection & Preprocessing

- Historical adjusted close prices are fetched using `yfinance`
- Assets are aligned by date and cleaned for missing values
- Log prices and returns are used to stabilize variance
- Only sufficiently liquid assets are considered


### 2. Feature Engineering using PCA

To capture common underlying market drivers:
- Asset returns are standardized
- **Principal Component Analysis (PCA)** is applied
- Reduced-dimensional representations are used for clustering

This step helps group assets driven by similar latent factors.


### 3. Unsupervised Pair Selection (OPTICS)

Following the motivation of Sarmento & Horta (2020):

- Assets are clustered using **OPTICS**
- Candidate pairs are formed *within clusters*
- This constrains the search space and improves robustness compared to naive correlation filtering


### 4. Statistical Validation (Cointegration)

Candidate pairs are filtered using:
- **Engle–Granger cointegration tests**
- Stationarity checks using the **ADF test**

Only pairs with statistically significant long-term relationships are traded.


### 5. Hedge Ratio Estimation (OLS)

For each validated pair, the hedge ratio is estimated using **Ordinary Least Squares (OLS)**:

Pᴀₜ = α + β · Pʙₜ + εₜ

- The hedge ratio β determines position sizing
- This assumes a stable linear relationship over the trading window


### 6. Spread Construction

The spread is defined as:

Spreadₜ = Pᴀₜ − β · Pʙₜ

where:
- `Pᴀₜ` and `Pʙₜ` are the **log-transformed prices** of assets A and B at time `t`
- `β` is the **hedge ratio**, estimated using **Ordinary Least Squares (OLS)**

The spread is expected to exhibit **mean-reverting behavior**.


### 7. Signal Generation (Z-Score Framework)

The spread is normalized using rolling statistics to compute a z-score.

Trading logic:
- **Enter Long–Short** when z-score < −2
- **Enter Short–Long** when z-score > +2
- **Exit** when z-score reverts toward zero
- **Risk control** via hard thresholds on extreme divergence

Positions are updated sequentially to avoid look-ahead bias.


### 8. ARMA-Based Forecasting Filter

To reduce prolonged drawdowns during non-reverting regimes, the strategy incorporates **ARMA-based spread forecasting**, as motivated by Sarmento & Horta (2020).

- An **ARMA model** is fitted to the spread
- Short-horizon forecasts are generated
- Trades are **filtered or exited** when forecasts indicate continued divergence rather than mean reversion

This introduces a trade-off:
- Reduced drawdowns and adverse regimes
- Potential reduction in overall trade frequency and profitability


### 9. Transaction Costs

- Transaction costs are explicitly modeled (~0.1%)
- Costs are applied whenever positions change
- Ensures realistic and conservative performance estimates


### 10. Backtesting & Performance Evaluation

- Spread-based PnL is computed using lagged positions
- Trade counts and holding periods are tracked
- Results are aggregated across pairs

Reported metrics include:
- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Number of Trades
- Average Trade Return

---

## Alignment with Research Paper

| Research Component | Implementation |
|------------------|----------------|
Pair search space reduction | PCA + OPTICS clustering |
Statistical validation | Cointegration & ADF tests |
Hedge ratio estimation | OLS regression |
Forecasting-based filtering | ARMA on spreads |
Signal generation | Z-score mean reversion |
Transaction costs | Explicitly modeled |

---

## Limitations

- Cointegration relationships may break during regime shifts
- OLS assumes a static hedge ratio
- Short-selling constraints may apply in certain markets
- Strategy performance is sensitive to parameter choices

---

## Possible Extensions

- Time-varying hedge ratios
- Intraday implementation
- Portfolio-level capital allocation
- Regime detection mechanisms

---

