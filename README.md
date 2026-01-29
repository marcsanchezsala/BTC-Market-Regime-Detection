# Wasserstein-based Clustering for Market Regime Detection

This project explores the identification of financial market regimes (Bull, Bear, and Lateral) in **Bitcoin (BTC)** using unsupervised learning. The core of the project is the implementation of **Wasserstein K-Means (WK-Means)**, a clustering algorithm that leverages the Optimal Transport distance to better capture the distributional characteristics of financial time series.

## Overview

Traditional K-Means often fails to capture the temporal morphology of financial data. By using the **Wasserstein distance**, this model groups periods of price action based on their statistical distributions, providing more robust regime identification for trading strategies.

---

## Project Overview

The goal is to identify latent states in the market to improve trading decision-making. The workflow includes feature engineering (momentum slopes), custom algorithm implementation, clustering analysis, and a backtesting engine to validate the regimes against a "Buy & Hold" strategy.

### Key Features
* **Momentum Feature Engineering:** Calculation of Moving Average Slopes (velocity of price changes) to normalize non-stationary price data.
* **Custom Algorithm:** A proprietary implementation of **Wasserstein K-Means (`WKMeans`)** utilizing the `POT` (Python Optimal Transport) library.
* **Regime Classification:** Grouping market periods into clusters based on distributional similarity (Wasserstein Barycenters).
* **Backtesting Engine:** Validates the predictive power of regimes by simulating a trading strategy that switches exposure based on the detected cluster.
* **Automated Profiling:** Deep Exploratory Data Analysis (EDA) using `ydata-profiling`.

---

## Repository Structure

### Scripts & Modules
* **`wkmeans.py`**: The core algorithmic module. It contains the `WKMeans` class. Instead of standard Euclidean centroids, it computes **Wasserstein Barycenters** (the distribution that minimizes the transport cost to all samples in the cluster).
* **`MAslope.py`**: A utility module to calculate the slope of Simple (SMA) or Exponential (EMA) Moving Averages. This captures the "speed" of the trend.

### Jupyter Notebooks
* **`features.ipynb`** *(Step 1 - Preprocessing)*: Loads raw OHLCV data, computes technical indicators (slopes), generates the `report.html` for data profiling, and saves the `btc_features.parquet` file.
* **`regime.ipynb`** *(Step 2 - Training)*: Loads processed features, trains the `WKMeans` model to find optimal clusters, and visualizes the regimes overlaid on the price chart.
* **`test.ipynb`** *(Step 3 - Backtesting)*: Simulates a trading strategy based on the clusters found in the training step. It compares the "Naive Cluster Strategy" cumulative returns against the "Buy & Hold" benchmark.
* **`tutorial.ipynb`**: A sandbox notebook for understanding the mathematical concepts of Wasserstein distance and testing the algorithm on smaller data subsets.

### Data & Reports
* **`btc_binance_1d.parquet`**: Historical OHLCV data for Bitcoin (Daily timeframe).
* **`btc_features.parquet`**: The processed dataset containing engineering features ready for clustering.
* **`report.html`**: An automated HTML report detailing data distribution statistics and correlations.

---

## Installation & Requirements

To run this project, you need **Python 3.8+**. The critical dependency is `POT` (Python Optimal Transport).

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn pot ydata-profiling pyarrow