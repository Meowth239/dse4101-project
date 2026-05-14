import numpy as np

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