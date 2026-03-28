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
import gc
from tensorflow.keras import backend as K




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

# for jittering

def create_jittered_sequences(data, feature_cols, target_col, lookback,
                    jitter_std=0.3, jitter_cols=None, num_samples=0):
    """
    Creates sequences for time series modeling with optional internal jittering on X.

    Args:
        data:         Input DataFrame.
        feature_cols: List of feature column names.
        target_col:   Target column name.
        lookback:     Number of past timesteps to include in each sequence.
        jitter_std:   Noise scale as a fraction of each column's std (default 0.3).
                      Set to 0 to disable jittering.
        jitter_cols:  Indices (int) or names (str) of feature_cols to jitter.
                      If None, jitter is applied to all features.
        num_samples:  Number of jittered copies of each window to append per sequence.
                      0 disables augmentation entirely.
    """
    values_x = data[feature_cols].values
    values_y = data[target_col].values

    # Resolve jitter column indices once
    if jitter_cols is None:
        col_indices = list(range(values_x.shape[1]))
    else:
        col_indices = [
            feature_cols.index(c) if isinstance(c, str) else c
            for c in jitter_cols
        ]

    X, y = [], []

    for t in range(lookback, len(data)):
        window = values_x[t - lookback:t]   # (lookback, features)

        if jitter_std and jitter_std > 0 and num_samples > 0:
            # Per-column std computed over the current window
            col_stds = window[:, col_indices].std(axis=0)   # (len(col_indices),)
            scaled_std = jitter_std * col_stds               # fraction of each col's std

            # Build num_samples jittered copies of the window
            # noise shape: (num_samples, lookback, len(col_indices))
            noise = np.random.normal(0, scaled_std, (num_samples, lookback, len(col_indices)))

            augmented = np.tile(window, (num_samples, 1, 1))  # (num_samples, lookback, features)
            augmented[:, :, col_indices] += noise

            # Stack original + augmented copies
            combined = np.concatenate([window[np.newaxis], augmented], axis=0)  # (1+num_samples, lookback, features)
        else:
            combined = window[np.newaxis]  # (1, lookback, features)

        X.append(combined)
        y.append(np.full(combined.shape[0], values_y[t]))  # same target for all copies

    # X: list of (1+num_samples, lookback, features) → (total_rows, lookback, features)
    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


#################################################################################################

import gc
import tensorflow as tf
from tensorflow.keras import backend as K

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
    set_seed(seed)

    results = []
    df = df.copy().reset_index(drop=True)

    start_idx = max(initial_train_size, lookback)
    stop_idx = end_idx if end_idx is not None else len(df)

    for test_idx in range(start_idx, stop_idx):
        train_df = df.iloc[:test_idx].copy()

        if scale:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
            train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])

        X_train, y_train = create_sequences(train_df, feature_cols, target_col, lookback)

        if len(X_train) == 0:
            continue

        hist_window = df.iloc[test_idx - lookback:test_idx].copy()

        if scale:
            hist_window[feature_cols] = x_scaler.transform(hist_window[feature_cols])

        X_test = hist_window[feature_cols].values.reshape(1, lookback, len(feature_cols))

        # cleanup before new model
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()

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

        # cleanup after using this model
        del model, train_df, X_train, y_train, hist_window, X_test
        if scale:
            del x_scaler, y_scaler
        gc.collect()

    K.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()

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
    set_seed(seed)

    results = []
    df = df.copy().reset_index(drop=True)

    start_idx = max(initial_train_size, train_start_idx + lookback)
    stop_idx = end_idx if end_idx is not None else len(df)

    for test_idx in range(start_idx, stop_idx):
        train_df = df.iloc[train_start_idx:test_idx].copy()

        if scale:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
            train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])

        X_train, y_train = create_sequences(train_df, feature_cols, target_col, lookback)

        if len(X_train) == 0:
            continue

        hist_window = df.iloc[test_idx - lookback:test_idx].copy()

        if scale:
            hist_window[feature_cols] = x_scaler.transform(hist_window[feature_cols])

        X_test = hist_window[feature_cols].values.reshape(1, lookback, len(feature_cols))

        # clear any leftover TF state before building a new model
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()

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

        # explicitly delete large objects from this iteration
        del model, train_df, X_train, y_train, hist_window, X_test
        if scale:
            del x_scaler, y_scaler
        gc.collect()

    # optional final cleanup
    K.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()

    return pd.DataFrame(results)

####################################################################################

# for jittering

def expanding_window_lstm_forecast3(
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
    seed=42,
    nojitter=True,
    jitter_std=0.3,
    jitter_cols=None,
    num_samples=0
):
    
    """
    Creates augmented LSTM sequences with optional jittering applied to input features.

    Args:
        data:         Input DataFrame in time order.
        feature_cols: List of feature column names used as inputs.
        target_col:   Target column name to predict.
        lookback:     Number of past timesteps per sequence.
        jitter_std:   Noise scale as a fraction of each column’s standard deviation.
        jitter_cols:  Subset of feature columns to apply jitter to (default: all).
        num_samples:  Number of jittered copies generated per original sequence.

    Returns:
        X: Array of shape (n_samples, lookback, n_features) containing original and jittered sequences.
        y: Array of shape (n_samples,) containing corresponding targets (same target repeated for augmented copies).
    """

    set_seed(seed)

    results = []
    df = df.copy().reset_index(drop=True)

    start_idx = max(initial_train_size, lookback)
    stop_idx = end_idx if end_idx is not None else len(df)

    for test_idx in range(start_idx, stop_idx):
        train_df = df.iloc[:test_idx].copy()

        # -----------------------------
        # scale training data
        # -----------------------------
        if scale:
            x_scaler = StandardScaler()
            y_scaler = StandardScaler()

            train_df[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
            train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])

        # -----------------------------
        # build training sequences
        # -----------------------------
        if nojitter:
            X_train, y_train = create_sequences(
                train_df,
                feature_cols,
                target_col,
                lookback
            )
        else:
            X_train, y_train = create_jittered_sequences(
                train_df,
                feature_cols=feature_cols,
                target_col=target_col,
                lookback=lookback,
                jitter_std=jitter_std,
                jitter_cols=jitter_cols,
                num_samples=num_samples
            )

        if len(X_train) == 0:
            if scale:
                del x_scaler, y_scaler
            continue

        # -----------------------------
        # create clean test window
        # -----------------------------
        hist_window = df.iloc[test_idx - lookback:test_idx].copy()

        if scale:
            hist_window[feature_cols] = x_scaler.transform(hist_window[feature_cols])

        X_test = hist_window[feature_cols].values.reshape(1, lookback, len(feature_cols))

        # -----------------------------
        # clear TF state before fit
        # -----------------------------
        K.clear_session()
        tf.keras.backend.clear_session()
        gc.collect()

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

        # clear memory
        del model, train_df, X_train, y_train, hist_window, X_test
        if scale:
            del x_scaler, y_scaler
        gc.collect()

    K.clear_session()
    tf.keras.backend.clear_session()
    gc.collect()

    return pd.DataFrame(results)

###########