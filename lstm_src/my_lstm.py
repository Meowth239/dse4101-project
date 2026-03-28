import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

#################################################################################################
# this is a more complex LSTM architecture with two LSTM layers and dropout, 
# inspired by common practices in time series forecasting
def build_lstm_model(lookback, n_features, units, dropout, lr=0.001):
    model = Sequential([
        LSTM(units, input_shape=(lookback, n_features), return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )
    
    return model

# model architecture inspired by Ly et al. (2021) "Forecasting Commodity Prices Using Long-Short-Term Memory Neural Networks" 
# using a single LSTM layer with 50 units, dropout of 0.1, and learning rate of 0.001
def build_lstm_model_paper(lookback, n_features, units=50, dropout=0.1, lr=0.001):
    model = Sequential([
        LSTM(units, input_shape=(lookback, n_features)),
        Dropout(dropout),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    return model
#################################################################################################

def create_sequences(data, feature_cols, target_col, lookback):
    X, y = [], []

    values_x = data[feature_cols].values
    values_y = data[target_col].values

    for t in range(lookback, len(data)):
        X.append(values_x[t - lookback:t])
        y.append(values_y[t])

    return np.array(X), np.array(y)


#################################################################################################

def expanding_window_lstm_forecast(
    df,
    feature_cols,
    target_col,
    date_col=None,
    initial_train_size=200,
    end_idx=None,
    lookback=20,
    units=50,
    dropout=0.2,
    epochs=20,
    batch_size=32,
    verbose=0,
    scale=True,
    seed=42
):
    """
    Expanding window one-step-ahead forecast using LSTM.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe in time order.
    feature_cols : list
        Input feature columns.
    target_col : str
        Target column to predict.
    date_col : str or None
        Optional date column to keep in output.
    initial_train_size : int
        First forecast origin. Training uses all rows before this index.
    end_idx : int or None
        Forecast until this index (exclusive). If None, forecast to end of df.
    lookback : int
        Number of past timesteps per sequence.
    units, dropout, epochs, batch_size : model/training params
    verbose : int
        Keras fit verbosity.
    scale : bool
        Whether to fit StandardScaler on training data at each step.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with forecast date, actual, predicted.

    Notes
    -----
    - This method is computationally expensive because it retrains the model at each step.
    - We use an expanding window approach where the model is trained on all data
      available up to the current time, converted into overlapping lookback sequences.
    - Each prediction uses only the most recent sequence, but model parameters are
      estimated using all past sequences.
    - Scaling is done within each training step to prevent leakage. Scalers are fit
      only on the training data and then applied to the prediction sequence.
    """
    set_seed(seed)

    results = []
    df = df.copy().reset_index(drop=True)

    start_idx = max(initial_train_size, lookback)
    stop_idx = end_idx if end_idx is not None else len(df)

    for test_idx in range(start_idx, stop_idx):
        # all data up to test_idx - 1
        train_df = df.iloc[:test_idx].copy()

        if scale:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
            train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])

        # build training sequences
        X_train, y_train = create_sequences(train_df, feature_cols, target_col, lookback)

        if len(X_train) == 0:
            continue

        # latest sequence: [test_idx-lookback, ..., test_idx-1]
        hist_window = df.iloc[test_idx - lookback:test_idx].copy()

        if scale:
            hist_window[feature_cols] = x_scaler.transform(hist_window[feature_cols])

        X_test = hist_window[feature_cols].values.reshape(1, lookback, len(feature_cols))

        tf.keras.backend.clear_session()
        model = build_lstm_model(
            lookback=lookback,
            n_features=len(feature_cols),
            units=units,
            dropout=dropout
        )

        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop]
        )

        pred = model.predict(X_test, verbose=0).flatten()[0]

        if scale:
            pred = y_scaler.inverse_transform([[pred]])[0, 0]

        actual = df.iloc[test_idx][target_col]

        result = {
            "test_index": test_idx,
            "actual": actual,
            "predicted": pred
        }

        if date_col is not None:
            result[date_col] = df.iloc[test_idx][date_col]

        results.append(result)

    return pd.DataFrame(results)

#################################################################################################
def expanding_window_lstm_forecast2(
    df,
    feature_cols,
    target_col,
    date_col=None,
    train_start_idx=0,
    initial_train_size=200,
    end_idx=None,
    lookback=20,
    units=50,
    dropout=0.2,
    epochs=20,
    batch_size=32,
    verbose=0,
    scale=True,
    seed=42
):
    """
    Expanding window one-step-ahead forecast using LSTM.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe in time order.
    feature_cols : list
        Input feature columns.
    target_col : str
        Target column to predict.
    date_col : str or None
        Optional date column to keep in output.
    train_start_idx : int
        First row allowed in the training sample.
    initial_train_size : int
        First forecast origin. Training uses rows from train_start_idx
        up to initial_train_size - 1.
    end_idx : int or None
        Forecast until this index (exclusive). If None, forecast to end of df.
    lookback : int
        Number of past timesteps per sequence.
    units, dropout, epochs, batch_size : model/training params
    verbose : int
        Keras fit verbosity.
    scale : bool
        Whether to fit StandardScaler on training data at each step.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with forecast date, actual, predicted.
    """
    set_seed(seed)

    results = []
    df = df.copy().reset_index(drop=True)

    start_idx = max(initial_train_size, train_start_idx + lookback)
    stop_idx = end_idx if end_idx is not None else len(df)

    for test_idx in range(start_idx, stop_idx):
        # training data now starts at train_start_idx instead of 0
        train_df = df.iloc[train_start_idx:test_idx].copy()

        if scale:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
            train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])

        # create sequences from training data only
        X_train, y_train = create_sequences(train_df, feature_cols, target_col, lookback)

        if len(X_train) == 0:
            continue

        # latest sequence for prediction still comes from actual history
        hist_window = df.iloc[test_idx - lookback:test_idx].copy()

        if scale:
            hist_window[feature_cols] = x_scaler.transform(hist_window[feature_cols])

        X_test = hist_window[feature_cols].values.reshape(1, lookback, len(feature_cols))

        tf.keras.backend.clear_session()

        model = build_lstm_model(
            lookback=lookback,
            n_features=len(feature_cols),
            units=units,
            dropout=dropout
        )

        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop]
        )

        pred = model.predict(X_test, verbose=0).flatten()[0]

        if scale:
            pred = y_scaler.inverse_transform([[pred]])[0, 0]

        actual = df.iloc[test_idx][target_col]

        result = {
            "test_index": test_idx,
            "actual": actual,
            "predicted": pred
        }

        if date_col is not None:
            result[date_col] = df.iloc[test_idx][date_col]

        results.append(result)

    return pd.DataFrame(results)