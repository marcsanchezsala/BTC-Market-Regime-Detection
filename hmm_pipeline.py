import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.preprocessing import StandardScaler

def hmm_pipeline(df, df_features, model_type = 'GaussianHMM', n_states = 3, scale = True, split = 0.6, state_labels = ['Bearish', 'Sideways', 'Bullish']):
    df.index = pd.to_datetime(df.index)
    df_features.index = pd.to_datetime(df_features.index)
    print(f"Shapes before alignment: df={df.shape}, df_features={df_features.shape}")
    
    df, df_features = df.align(df_features, join='inner', axis=0)
    
    print(f"Shapes after alignment: df={df.shape}, df_features={df_features.shape}")
    
    N_STATES = n_states
    split_index = int(len(df) * split)

    train_data = df.iloc[:split_index].copy()
    test_data = df.iloc[split_index:].copy()

    train_features = df_features.iloc[:split_index].copy()
    test_features = df_features.iloc[split_index:].copy()

    print(f"Training data: {len(train_data)} candles ({train_data.index[0].date()} - {train_data.index[-1].date()})")
    print(f"Test data: {len(test_data)} candles ({test_data.index[0].date()} - {test_data.index[-1].date()})")

    if scale:
        print("Scaling features using StandardScaler...")
        scaler = StandardScaler()
        # Fit solo en train
        obs_data_train = scaler.fit_transform(train_features.values)
        # Transform en test usando las medias/std de train
        obs_data_test = scaler.transform(test_features.values)
    else:
        print("Using raw features (No Scaling)...")
        obs_data_train = train_features.values
        obs_data_test = test_features.values

    if model_type == 'GaussianHMM':
        model = GaussianHMM(n_components=N_STATES, covariance_type='full', n_iter=10000, tol=1e-4, algorithm='map', random_state=42, verbose=False)
    elif model_type == 'GMMHMM':
        # n_mix es el número de gaussianas dentro de cada estado (mezcla)
        model = GMMHMM(n_components=N_STATES, n_mix=2, covariance_type='full', n_iter=10000, tol=1e-4, algorithm='map', random_state=42, verbose=False)

    model.fit(obs_data_train)
    print("Converged:", model.monitor_.converged)
    print("Iterations:", model.monitor_.iter)
    print("Final log-likelihood:", model.score(obs_data_train))

    hidden_states = model.predict(obs_data_train)
    train_data['Regime'] = hidden_states
    state_means = [obs_data_train[hidden_states == i].mean() for i in range(3)]
    state_order = np.argsort(state_means)

    # Map state numbers to regime names
    state_map = {state_order[0]: state_labels[0], state_order[1]: state_labels[1], state_order[2]: state_labels[2]}
    train_data['Regime_Label'] = train_data['Regime'].map(state_map)
    
    # PLOT REGIMES
    close_neutral = train_data['Close']
    colors = {'Bearish': "red", 'Sideways': "blue", 'Bullish': "green"}
    close_cluster = {}

    for c in colors.keys():
        mask = (train_data['Regime_Label'] == c) | (train_data['Regime_Label'].shift(-1) == c)
        series = train_data['Close'].copy()
        series[~mask] = np.nan
        close_cluster[c] = series

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

    for c, series in close_cluster.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            mode='lines',
            line=dict(color=colors[c], width=2.5),
            name=f"{c}",
            connectgaps=False
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=train_data['Regime_Label'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name='State'
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=700,
        title_text="Market Regime Detector (Train)",
        hovermode="x unified",
        dragmode='zoom',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(rangeslider_visible=False)

    fig.show();

    hidden_states = model.predict(obs_data_test)
    test_data['Regime'] = hidden_states
    test_data['Regime_Label'] = test_data['Regime'].map(state_map)

    close_neutral = test_data['Close']
    colors = {'Bearish': "red", 'Sideways': "blue", 'Bullish': "green"}
    close_cluster = {}

    for c in colors.keys():
        mask = (test_data['Regime_Label'] == c) | (test_data['Regime_Label'].shift(-1) == c)
        series = test_data['Close'].copy()
        series[~mask] = np.nan
        close_cluster[c] = series

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

    for c, series in close_cluster.items():
        fig.add_trace(go.Scatter(
            x=df.index, y=series,
            mode='lines',
            line=dict(color=colors[c], width=2.5),
            name=f"{c}",
            connectgaps=False
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=test_data['Regime_Label'],
        mode='lines',
        line=dict(color='cyan', width=1),
        name='State'
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        height=700,
        title_text="Market Regime Detector (Test)",
        hovermode="x unified",
        dragmode='zoom',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(rangeslider_visible=False)

    fig.show();

    return model
