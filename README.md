# Wasserstein-based Clustering for Market Regime Detection

This project explores the identification of financial market regimes (Bull, Bear, and Lateral) in **Bitcoin (BTC)** using unsupervised learning. The core of the project is the implementation of **Wasserstein K-Means (WK-Means)**, a clustering algorithm that leverages the Optimal Transport distance to better capture the distributional characteristics of financial time series.

## Overview

Traditional K-Means often fails to capture the temporal morphology of financial data. By using the **Wasserstein distance**, this model groups periods of price action based on their statistical distributions, providing more robust regime identification for trading strategies.

---

## Tech Stack

* **Python**
* **Data Science:** `Pandas`, `NumPy`, `Scikit-Learn`.
* **Clustering:** `WKMeans` (Wasserstein-based) & standard `K-Means`.
* **Visualization:** `Matplotlib`, `Seaborn` (static) and `Plotly` (interactive financial charts).
* **Addicional Libraries:** `pyarrow`, `POT`.

---

## Methodology & Experiments

The project is divided into two main analytical approaches:

### 1. Regime K=2
* Focuses on identifying **Uptrend vs. Downtrend**.

### 2. Regime K=3
* Aims to distinguish between **Bull, Bear, and Sideways (Neutral)** markets.

### Backtesting: Naive Cluster Strategy
The notebooks include a backtest comparing a **"Buy & Hold"** strategy against a **Cluster-based Strategy**. In this strategy, the model only takes a "Long" position when the market is classified into a specific favorable cluster.

**Simple backtest with all the metrics missing, it is just to illustrate a simple way to use the clustering.**


