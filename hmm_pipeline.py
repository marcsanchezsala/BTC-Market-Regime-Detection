import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats


def hmm_pipeline(df, df_features, model_type='GaussianHMM', n_states=3, n_mix=2, 
                 scale=True, split=0.6, state_labels=['Bearish', 'Sideways', 'Bullish'],
                 covariance_type='full', n_iter=10000, tol=1e-4, random_state=42):
    """
    Improved HMM-based market regime detection pipeline with comprehensive analysis.
    
    Parameters:
    - df: DataFrame with OHLC data (must have 'Close' column and datetime index)
    - df_features: DataFrame with features for HMM (same index as df)
    - model_type: 'GaussianHMM' or 'GMMHMM'
    - n_states: Number of hidden states
    - n_mix: Number of Gaussian mixtures (only for GMMHMM)
    - scale: Whether to scale features with StandardScaler
    - split: Train/test split ratio
    - state_labels: Labels for the states (ordered from lowest to highest mean returns)
    - covariance_type: Type of covariance ('full', 'diag', 'spherical', 'tied')
    - n_iter: Maximum number of iterations
    - tol: Convergence threshold
    - random_state: Random seed for reproducibility
    
    Returns:
    - model: Trained HMM model
    - log_returns_dist: Dictionary with log returns by regime (out-of-sample)
    - train_data: Training data with regime labels
    - test_data: Test data with regime labels
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
    
    print(f"\nTraining data: {len(train_data)} candles ({train_data.index[0].date()} - {train_data.index[-1].date()})")
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
    
    # Initialize and train HMM model
    print(f"\nTraining {model_type} with {n_states} states...")
    if model_type == 'GaussianHMM':
        model = GaussianHMM(
            n_components=n_states, 
            covariance_type=covariance_type, 
            n_iter=n_iter, 
            tol=tol, 
            algorithm='map', 
            random_state=random_state, 
            verbose=False
        )
    elif model_type == 'GMMHMM':
        model = GMMHMM(
            n_components=n_states, 
            n_mix=n_mix, 
            covariance_type=covariance_type, 
            n_iter=n_iter, 
            tol=tol, 
            algorithm='map', 
            random_state=random_state, 
            verbose=False
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'GaussianHMM' or 'GMMHMM'")
    
    # Fit model
    model.fit(obs_data_train)
    
    print(f"Converged: {model.monitor_.converged}")
    print(f"Iterations: {model.monitor_.iter}")
    print(f"Final log-likelihood: {model.score(obs_data_train):.4f}")
    
    # Predict hidden states for training data
    hidden_states_train = model.predict(obs_data_train)
    train_data['Regime'] = hidden_states_train
    
    # Order states by mean returns for this regime
    state_means = []
    for i in range(n_states):
        mask = train_data['Regime'] == i
        if mask.sum() > 0:
            regime_returns = train_data.loc[mask, 'Close'].pct_change().mean()
            state_means.append(regime_returns)
        else:
            state_means.append(0)
    
    state_order = np.argsort(state_means)
    state_map = {state_order[i]: state_labels[i] for i in range(n_states)}
    train_data['Regime_Label'] = train_data['Regime'].map(state_map)
    
    print(f"\nState mapping: {state_map}")
    print(f"\nState distribution (train):")
    print(train_data['Regime_Label'].value_counts())
    
    # Plot training regimes
    _plot_regimes(df, train_data, "Market Regime Detector (Train)", state_labels)
    
    # Predict hidden states for test data
    hidden_states_test = model.predict(obs_data_test)
    test_data['Regime'] = hidden_states_test
    test_data['Regime_Label'] = test_data['Regime'].map(state_map)
    
    print(f"\nState distribution (test):")
    print(test_data['Regime_Label'].value_counts())
    
    # Plot test regimes
    _plot_regimes(df, test_data, "Market Regime Detector (Test)", state_labels)
    
    # Performance analysis on out-of-sample data (test set)
    print("\n" + "="*80)
    print("OUT-OF-SAMPLE PERFORMANCE ANALYSIS (Test Set)")
    print("="*80)
    
    # Calculate log returns for test data
    test_data['Log_Returns'] = np.log(test_data['Close'] / test_data['Close'].shift(1))
    
    # Group log returns by regime
    log_returns_by_regime = {}
    for label in state_labels:
        mask = test_data['Regime_Label'] == label
        if mask.sum() > 0:
            log_returns_by_regime[label] = test_data.loc[mask, 'Log_Returns'].dropna()
    
    # Plot distribution analysis
    _plot_log_returns_distributions(log_returns_by_regime, state_labels)
    
    # Regime statistics comparison
    _print_regime_statistics(test_data, state_labels)
    
    # Transition matrix analysis
    _analyze_transition_matrix(model, state_map, state_labels)
    
    # Return results
    return model, log_returns_by_regime, train_data, test_data


def _plot_regimes(df, data, title, state_labels):
    """
    Plot market regimes with price coloring and state timeline.
    
    Parameters:
    - df: Full DataFrame with all dates
    - data: DataFrame with regime labels
    - title: Plot title
    - state_labels: List of state labels
    """
    colors = {'Bearish': "red", 'Sideways': "blue", 'Bullish': "green",
              'Strong Bearish': "darkred", 'Weak Bearish': "orange",
              'Weak Bullish': "lightgreen", 'Strong Bullish': "darkgreen"}
    
    close_neutral = data['Close']
    close_cluster = {}
    
    # Create colored series for each regime
    for c in state_labels:
        mask = (data['Regime_Label'] == c) | (data['Regime_Label'].shift(-1) == c)
        series = data['Close'].copy()
        series[~mask] = np.nan
        close_cluster[c] = series
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price by State", "Assigned State")
    )
    
    # Add neutral line
    fig.add_trace(go.Scatter(
        x=data.index, y=close_neutral,
        mode='lines',
        line=dict(color='gray', width=1),
        name='Neutral',
        hoverinfo='skip'
    ), row=1, col=1)
    
    # Add colored regime lines
    for c, series in close_cluster.items():
        fig.add_trace(go.Scatter(
            x=data.index, y=series,
            mode='lines',
            line=dict(color=colors.get(c, 'white'), width=2.5),
            name=f"{c}",
            connectgaps=False
        ), row=1, col=1)
    
    # Add state timeline
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Regime_Label'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name='State'
    ), row=2, col=1)
    
    # Update layout
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
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess = returns - risk_free / 252
    return np.sqrt(252) * excess.mean() / excess.std()


def _max_drawdown(cum_returns):
    """Calculate maximum drawdown."""
    if len(cum_returns) == 0:
        return 0.0
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    return drawdown.min()


def _plot_log_returns_distributions(log_returns_by_cluster, state_labels):
    """
    Plot comprehensive log returns distribution analysis for each regime.
    
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
                'Sharpe (Ann.)': _sharpe_ratio(data)
            })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False))
        print("="*80)


def _print_regime_statistics(data, state_labels):
    """
    Print comprehensive regime statistics.
    
    Parameters:
    - data: DataFrame with regime labels and returns
    - state_labels: List of state labels
    """
    print("\n" + "="*80)
    print("REGIME PERFORMANCE METRICS")
    print("="*80)
    
    regime_stats = []
    
    for label in state_labels:
        mask = data['Regime_Label'] == label
        if mask.sum() > 0:
            regime_data = data[mask].copy()
            
            # Calculate returns
            regime_data['Returns'] = regime_data['Close'].pct_change()
            regime_data['Cum_Returns'] = (1 + regime_data['Returns']).cumprod()
            
            # Calculate metrics
            total_return = (regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0] - 1) * 100
            avg_return = regime_data['Returns'].mean() * 100
            volatility = regime_data['Returns'].std() * 100
            sharpe = _sharpe_ratio(regime_data['Returns'].dropna())
            max_dd = _max_drawdown(regime_data['Cum_Returns'].dropna()) * 100
            
            # Win rate
            wins = (regime_data['Returns'] > 0).sum()
            total = (regime_data['Returns'] != 0).sum()
            win_rate = (wins / total * 100) if total > 0 else 0
            
            regime_stats.append({
                'Regime': label,
                'Periods': mask.sum(),
                'Total Return (%)': total_return,
                'Avg Return (%)': avg_return,
                'Volatility (%)': volatility,
                'Sharpe Ratio': sharpe,
                'Max DD (%)': max_dd,
                'Win Rate (%)': win_rate
            })
    
    if regime_stats:
        stats_df = pd.DataFrame(regime_stats)
        print(stats_df.to_string(index=False))
        print("="*80)


def _analyze_transition_matrix(model, state_map, state_labels):
    """
    Analyze and display the state transition matrix.
    
    Parameters:
    - model: Trained HMM model
    - state_map: Dictionary mapping state indices to labels
    - state_labels: List of state labels in order
    """
    print("\n" + "="*80)
    print("STATE TRANSITION MATRIX ANALYSIS")
    print("="*80)
    
    # Get transition matrix
    trans_matrix = model.transmat_
    
    # Create ordered matrix based on state labels
    n_states = len(state_labels)
    ordered_matrix = np.zeros((n_states, n_states))
    
    for i, from_label in enumerate(state_labels):
        for j, to_label in enumerate(state_labels):
            # Find original indices
            from_idx = [k for k, v in state_map.items() if v == from_label][0]
            to_idx = [k for k, v in state_map.items() if v == to_label][0]
            ordered_matrix[i, j] = trans_matrix[from_idx, to_idx]
    
    # Create DataFrame for better visualization
    trans_df = pd.DataFrame(
        ordered_matrix,
        index=[f"From {label}" for label in state_labels],
        columns=[f"To {label}" for label in state_labels]
    )
    
    print("\nTransition Probabilities:")
    print(trans_df.to_string())
    
    # Calculate average duration in each state
    print("\n" + "-"*80)
    print("Expected Duration in Each State (periods):")
    print("-"*80)
    for i, label in enumerate(state_labels):
        duration = 1 / (1 - ordered_matrix[i, i]) if ordered_matrix[i, i] < 1 else float('inf')
        print(f"{label}: {duration:.2f}")
    
    print("="*80)
    
    # Visualize transition matrix
    fig = go.Figure(data=go.Heatmap(
        z=ordered_matrix,
        x=[f"To {label}" for label in state_labels],
        y=[f"From {label}" for label in state_labels],
        colorscale='Blues',
        text=np.round(ordered_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 14},
        colorbar=dict(title="Probability")
    ))
    
    fig.update_layout(
        template='plotly_dark',
        title='State Transition Matrix Heatmap',
        height=500,
        xaxis_title="To State",
        yaxis_title="From State"
    )
    
    fig.show()


# Example usage function
def example_usage():
    """
    Example of how to use the improved HMM pipeline.
    """
    # This is just for documentation purposes
    pass
    
    # Example call:
    # model, log_returns, train_data, test_data = hmm_pipeline(
    #     df=price_data,
    #     df_features=features_data,
    #     model_type='GaussianHMM',
    #     n_states=3,
    #     scale=True,
    #     split=0.6,
    #     state_labels=['Bearish', 'Sideways', 'Bullish']
    # )
