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