import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from scipy import stats
import ot


class WKMeans:
    def __init__(self, k, p=1, tolerance=1e-4, max_iter=100, seed=42):
        """
        Wasserstein K-means algorithm.

        Parameters:
        - k: Number of clusters
        - p: Order of the Wasserstein distance
        - tolerance: Convergence threshold
        - max_iter: Maximum number of iterations
        """
        self.k = k
        self.p = p
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None

    def wasserstein_distance(self, mu1, mu2):
        """Compute the p-Wasserstein distance between two empirical distributions."""
        n = len(mu1)
        a = np.ones(n) / n
        M = ot.dist(mu1.reshape(-1, 1), mu2.reshape(-1, 1), metric="minkowski", p=self.p)
        return ot.emd2(a, a, M)

    def wasserstein_barycenter(self, cluster_samples):
        """Compute the Wasserstein barycenter (median of sorted distributions)."""
        sorted_samples = np.sort(cluster_samples, axis=0)
        return np.median(sorted_samples, axis=0)

    def fit(self, samples, inertia=False):
        """
        Fit the WK-means clustering algorithm.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)
        """
        # Initialize centroids by randomly selecting k samples
        np.random.seed(self.seed)
        self.centroids = [samples[i] for i in np.random.choice(len(samples), self.k, replace=False)]
        
        for iteration in range(self.max_iter):
            clusters = {i: [] for i in range(self.k)}

            # Assign each sample to the closest centroid
            for sample in samples:
                distances = [self.wasserstein_distance(sample, centroid) for centroid in self.centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(sample)

            # Update centroids as Wasserstein barycenters
            new_centroids = []
            for i in range(self.k):
                if clusters[i]:
                    new_centroids.append(self.wasserstein_barycenter(np.array(clusters[i])))
                else:
                    new_centroids.append(self.centroids[i])  # Keep previous centroid if no samples assigned

            # Compute loss function (sum of Wasserstein distances)
            loss = sum(self.wasserstein_distance(self.centroids[i], new_centroids[i]) for i in range(self.k))

            # Check convergence
            if loss < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break

            self.centroids = new_centroids
        
        if inertia:
            self.inertia_ = 0
            for sample in samples:
                dist = min([self.wasserstein_distance(sample, c) for c in self.centroids])
                self.inertia_ += dist**2

    def predict(self, samples):
        """
        Predict the cluster for each sample.

        Parameters:
        - samples: List of empirical distributions (numpy arrays)

        Returns:
        - List of cluster indices
        """
        return [np.argmin([self.wasserstein_distance(sample, centroid) for centroid in self.centroids]) for sample in samples]


def wkmeans_pipeline(df, df_features, window_size=20, k=3, p=1, scale=True, split=0.6, 
                     state_labels=['Bearish', 'Sideways', 'Bullish'], tolerance=1e-4, 
                     max_iter=100, seed=42):
    """
    Pipeline for WKMeans-based market regime detection.
    
    Parameters:
    - df: DataFrame with OHLC data (must have 'Close' column and datetime index)
    - df_features: DataFrame with features for clustering (same index as df)
    - window_size: Size of the rolling window for creating distributions
    - k: Number of clusters
    - p: Order of Wasserstein distance
    - scale: Whether to scale features
    - split: Train/test split ratio
    - state_labels: Labels for the states (ordered from lowest to highest mean)
    - tolerance: Convergence threshold for WKMeans
    - max_iter: Maximum iterations for WKMeans
    - seed: Random seed for reproducibility
    
    Returns:
    - model: Trained WKMeans model
    - log_returns_dist: Dictionary with log returns by regime (out-of-sample)
    """
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    df_features.index = pd.to_datetime(df_features.index)
    print(f"Shapes before alignment: df={df.shape}, df_features={df_features.shape}")
    
    # Align dataframes
    df, df_features = df.align(df_features, join='inner', axis=0)
    print(f"Shapes after alignment: df={df.shape}, df_features={df_features.shape}")
    
    # Train/test split
    split_index = int(len(df) * split)
    train_data = df.iloc[:split_index].copy()
    test_data = df.iloc[split_index:].copy()
    train_features = df_features.iloc[:split_index].copy()
    test_features = df_features.iloc[split_index:].copy()
    
    print(f"Training data: {len(train_data)} candles ({train_data.index[0].date()} - {train_data.index[-1].date()})")
    print(f"Test data: {len(test_data)} candles ({test_data.index[0].date()} - {test_data.index[-1].date()})")
    
    # Feature scaling
    if scale:
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        obs_data_train = scaler.fit_transform(train_features.values)
        obs_data_test = scaler.transform(test_features.values)
    else:
        print("Using raw features (No Scaling)...")
        obs_data_train = train_features.values
        obs_data_test = test_features.values
    
    # Create rolling window distributions for training
    print(f"Creating rolling window distributions (window_size={window_size})...")
    train_samples = []
    
    for i in range(window_size, len(obs_data_train)):
        window = obs_data_train[i-window_size:i].flatten()
        train_samples.append(window)
    
    print(f"Created {len(train_samples)} training samples")
    
    # Fit WKMeans model
    print(f"Training WKMeans with k={k}, p={p}...")
    model = WKMeans(k=k, p=p, tolerance=tolerance, max_iter=max_iter, seed=seed)
    model.fit(train_samples, inertia=True)
    
    print(f"Training completed. Inertia: {model.inertia_:.4f}")
    
    # Predict on ALL training data (creating samples for each point)
    train_data['Regime'] = np.nan
    for i in range(window_size, len(obs_data_train)):
        window = obs_data_train[i-window_size:i].flatten()
        regime = model.predict([window])[0]
        train_data.iloc[i, train_data.columns.get_loc('Regime')] = regime
    
    # Order states by mean returns for this regime
    valid_train = train_data.dropna(subset=['Regime'])
    state_means = []
    for i in range(k):
        mask = valid_train['Regime'] == i
        if mask.sum() > 0:
            regime_returns = valid_train.loc[mask, 'Close'].pct_change().mean()
            state_means.append(regime_returns)
        else:
            state_means.append(0)
    
    state_order = np.argsort(state_means)
    state_map = {state_order[i]: state_labels[i] for i in range(k)}
    train_data['Regime_Label'] = train_data['Regime'].map(state_map)
    
    print(f"\nState mapping: {state_map}")
    print(f"State distribution (train):")
    print(train_data['Regime_Label'].value_counts())
    
    # Plot training regimes
    _plot_regimes(df, train_data, "Market Regime Detector (Train)", state_labels)
    
    # Predict on ALL test data
    test_data['Regime'] = np.nan
    for i in range(window_size, len(obs_data_test)):
        window = obs_data_test[i-window_size:i].flatten()
        regime = model.predict([window])[0]
        test_data.iloc[i, test_data.columns.get_loc('Regime')] = regime
    
    test_data['Regime_Label'] = test_data['Regime'].map(state_map)
    
    print(f"\nState distribution (test):")
    print(test_data['Regime_Label'].value_counts())
    
    # Plot test regimes
    _plot_regimes(df, test_data, "Market Regime Detector (Test)", state_labels)

    # Calculate log returns distribution for each cluster (out-of-sample)
    print("\n" + "="*60)
    print("LOG RETURNS DISTRIBUTION BY CLUSTER (OUT-OF-SAMPLE)")
    print("="*60)
    
    # Calculate log returns for test data
    test_data['Log_Return'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
    
    # Group log returns by regime
    log_returns_by_cluster = {}
    for label in state_labels:
        mask = test_data['Regime_Label'] == label
        log_returns_by_cluster[label] = test_data.loc[mask, 'Log_Return'].dropna()
        
        if len(log_returns_by_cluster[label]) > 0:
            mean_ret = log_returns_by_cluster[label].mean()
            std_ret = log_returns_by_cluster[label].std()
            skew_ret = log_returns_by_cluster[label].skew()
            kurt_ret = log_returns_by_cluster[label].kurtosis()
            
            print(f"\n{label}:")
            print(f"  Count: {len(log_returns_by_cluster[label])}")
            print(f"  Mean: {mean_ret:.6f}")
            print(f"  Std: {std_ret:.6f}")
            print(f"  Skewness: {skew_ret:.4f}")
            print(f"  Kurtosis: {kurt_ret:.4f}")
    
    # Plot log returns distributions
    _plot_log_returns_distributions(log_returns_by_cluster, state_labels)
    
    return model, log_returns_by_cluster


def _plot_regimes(df, data, title, state_labels):
    """Helper function to plot market regimes."""
    close_neutral = data['Close']
    colors = {'Bearish': "red", 'Sideways': "blue", 'Bullish': "green",
              'Strong Bearish': "darkred", 'Weak Bearish': "orange",
              'Weak Bullish': "lightgreen", 'Strong Bullish': "darkgreen"}
    close_cluster = {}
    
    for label in state_labels:
        if label in colors:
            mask = (data['Regime_Label'] == label) | (data['Regime_Label'].shift(-1) == label)
            series = data['Close'].copy()
            series[~mask] = np.nan
            close_cluster[label] = series
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price by State", "Assigned State")
    )
    
    fig.add_trace(go.Scatter(
        x=df.index, y=close_neutral,
        mode='lines',
        line=dict(color='gray', width=1),
        name='Neutral',
        hoverinfo='skip'
    ), row=1, col=1)
    
    for label, series in close_cluster.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            mode='lines',
            line=dict(color=colors.get(label, 'white'), width=2.5),
            name=f"{label}",
            connectgaps=False
        ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=data['Regime_Label'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name='State'
    ), row=2, col=1)
    
    fig.update_layout(
        template='plotly_dark',
        height=700,
        title_text=title,
        hovermode="x unified",
        dragmode='zoom',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    fig.show()


def _sharpe_ratio(returns, risk_free=0.0):
    """Calculate Sharpe ratio."""
    excess = returns - risk_free / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def _max_drawdown(cum_returns):
    """Calculate maximum drawdown."""
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min()


def _plot_log_returns_distributions(log_returns_by_cluster, state_labels):
    """
    Plot log returns distributions for each cluster.
    
    Parameters:
    - log_returns_by_cluster: Dictionary with regime labels as keys and log returns as values
    - state_labels: List of state labels
    """
    colors_dist = {'Bearish': "red", 'Sideways': "blue", 'Bullish': "green",
                   'Strong Bearish': "darkred", 'Weak Bearish': "orange",
                   'Weak Bullish': "lightgreen", 'Strong Bullish': "darkgreen"}
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histogram by Regime", "Box Plot by Regime", 
                       "Violin Plot by Regime", "Q-Q Plot vs Normal"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Histogram
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            fig.add_trace(
                go.Histogram(
                    x=log_returns_by_cluster[label],
                    name=label,
                    opacity=0.7,
                    marker_color=colors_dist.get(label, 'gray'),
                    nbinsx=50,
                    histnorm='probability density'
                ),
                row=1, col=1
            )
    
    # 2. Box Plot
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            fig.add_trace(
                go.Box(
                    y=log_returns_by_cluster[label],
                    name=label,
                    marker_color=colors_dist.get(label, 'gray'),
                    boxmean='sd'
                ),
                row=1, col=2
            )
    
    # 3. Violin Plot
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            fig.add_trace(
                go.Violin(
                    y=log_returns_by_cluster[label],
                    name=label,
                    marker_color=colors_dist.get(label, 'gray'),
                    box_visible=True,
                    meanline_visible=True
                ),
                row=2, col=1
            )
    
    # 4. Q-Q Plot vs Normal
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            data = log_returns_by_cluster[label].values
            theoretical_quantiles = stats.probplot(data, dist="norm")[0][0]
            sample_quantiles = stats.probplot(data, dist="norm")[0][1]
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name=label,
                    marker=dict(color=colors_dist.get(label, 'gray'), size=4),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Add diagonal line for Q-Q plot
    all_returns = np.concatenate([log_returns_by_cluster[label].values 
                                  for label in state_labels 
                                  if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0])
    if len(all_returns) > 0:
        qq_min, qq_max = all_returns.min(), all_returns.max()
        fig.add_trace(
            go.Scatter(
                x=[qq_min, qq_max],
                y=[qq_min, qq_max],
                mode='lines',
                line=dict(color='white', dash='dash'),
                name='Normal',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        template='plotly_dark',
        height=800,
        title_text="Log Returns Distribution Analysis by Regime (Out-of-Sample)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(title_text="Log Returns", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    
    fig.update_xaxes(title_text="Regime", row=1, col=2)
    fig.update_yaxes(title_text="Log Returns", row=1, col=2)
    
    fig.update_xaxes(title_text="Regime", row=2, col=1)
    fig.update_yaxes(title_text="Log Returns", row=2, col=1)
    
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2)
    
    fig.show()
    
    # Additional: Individual distribution plots for each regime
    fig2 = go.Figure()
    
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            # Add histogram
            fig2.add_trace(
                go.Histogram(
                    x=log_returns_by_cluster[label],
                    name=f'{label} (n={len(log_returns_by_cluster[label])})',
                    opacity=0.6,
                    marker_color=colors_dist.get(label, 'gray'),
                    nbinsx=50,
                    histnorm='probability density'
                )
            )
            
            # Add fitted normal distribution
            mean = log_returns_by_cluster[label].mean()
            std = log_returns_by_cluster[label].std()
            x_range = np.linspace(log_returns_by_cluster[label].min(), 
                                 log_returns_by_cluster[label].max(), 100)
            normal_dist = stats.norm.pdf(x_range, mean, std)
            
            fig2.add_trace(
                go.Scatter(
                    x=x_range,
                    y=normal_dist,
                    name=f'{label} Normal Fit',
                    line=dict(color=colors_dist.get(label, 'gray'), dash='dash', width=2),
                    showlegend=True
                )
            )
    
    fig2.update_layout(
        template='plotly_dark',
        height=600,
        title_text="Log Returns Distributions with Normal Fits (Out-of-Sample)",
        xaxis_title="Log Returns",
        yaxis_title="Probability Density",
        barmode='overlay',
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    fig2.show()
    
    # Statistical comparison table
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON OF LOG RETURNS BY REGIME")
    print("="*80)
    
    stats_data = []
    for label in state_labels:
        if label in log_returns_by_cluster and len(log_returns_by_cluster[label]) > 0:
            data = log_returns_by_cluster[label]
            stats_data.append({
                'Regime': label,
                'Count': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Skewness': data.skew(),
                'Kurtosis': data.kurtosis(),
                'Sharpe (Ann.)': np.sqrt(252) * data.mean() / data.std() if data.std() > 0 else 0
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        print("="*80)
