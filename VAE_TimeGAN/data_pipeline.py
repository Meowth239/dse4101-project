"""
data_pipeline.py
================
Module 1: Data pipeline for multivariate time series.
Handles loading, validation, normalisation, sliding window construction,
and scaler persistence for downstream use in Modules 2 and 3.

Assumes the input DataFrame contains only the training portion of the data.
Train/test splitting is handled upstream by the researcher before calling
this module.

Usage
-----
from data_pipeline import build_pipeline, inverse_transform, load_scaler

# Build windowed dataset from raw training DataFrame
windows, scaler = build_pipeline(
    df=my_dataframe,            # rows = timesteps, columns = variables
    window_length=24,
    scaler_save_path="scaler.pkl"
)
# windows shape: (num_windows, window_length, num_variables)

# After generation, invert normalisation on synthetic data
original_scale = inverse_transform(synthetic_windows, scaler)

# Load a saved scaler in a separate session
scaler = load_scaler("scaler.pkl")

Notes
-----
- One row of the input DataFrame = one timestep (e.g. one month).
- All columns are treated as variables. Non-numeric columns are dropped
  with a warning.
- Normalisation is fit on the full training data before windowing, using
  MinMaxScaler by default. The fitted scaler is saved to disk so it can
  be reused consistently at generation and evaluation time.
- Sliding windows use a configurable stride (default 1).
"""

# =============================================================================
# Imports
# =============================================================================

import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class PipelineConfig:
    """
    Configuration for the data pipeline.

    Parameters
    ----------
    window_length : int
        Number of timesteps per sliding window. Must be less than the
        number of observations in the DataFrame.
    stride : int
        Step size between consecutive windows. Default is 1 (maximum
        overlap). Increase to reduce overlap between windows, which
        may help with the memorisation concern during generation.
    scaler_type : str
        Normalisation method. One of 'minmax' or 'standard'.
        'minmax' scales each variable to [0, 1].
        'standard' scales each variable to zero mean and unit variance.
    feature_range : Tuple[float, float]
        Only used when scaler_type is 'minmax'. Defines the target range.
        Default is (0, 1).
    verbose : bool
        If True, prints pipeline progress and summary statistics.
    """
    window_length: int = 24
    stride: int = 1
    scaler_type: str = "minmax"
    feature_range: Tuple[float, float] = (0, 1)
    verbose: bool = True


# =============================================================================
# Internal Utilities
# =============================================================================

def _validate_dataframe(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Validate and clean the input DataFrame before processing.

    - Drops non-numeric columns with a warning.
    - Checks for NaN and infinite values.
    - Checks that sufficient observations exist for at least one window.
    - Checks that the index is sensibly ordered (warns if not).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame containing only numeric columns.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.
    ValueError
        If the DataFrame has insufficient rows, all-NaN columns,
        or fewer than 2 numeric columns after cleaning.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(df).__name__}. "
            "Pass your training data as a DataFrame where rows are "
            "timesteps and columns are variables."
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Drop non-numeric columns with a warning
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        warnings.warn(
            f"The following non-numeric columns were dropped: {non_numeric}. "
            "Only numeric columns are used as variables.",
            UserWarning,
        )
        df = df.select_dtypes(include=[np.number])

    if df.shape[1] == 0:
        raise ValueError(
            "No numeric columns found in the DataFrame after dropping "
            "non-numeric columns."
        )

    if df.shape[1] < 2:
        warnings.warn(
            "Only one numeric variable found. The VAE will not learn "
            "cross-variable relationships with a single variable. "
            "Consider adding more variables.",
            UserWarning,
        )

    # Check for all-NaN columns
    all_nan_cols = df.columns[df.isna().all()].tolist()
    if all_nan_cols:
        raise ValueError(
            f"The following columns are entirely NaN and cannot be used: "
            f"{all_nan_cols}. Please remove or impute them before calling "
            "this function."
        )

    # Warn about partial NaN values
    partial_nan_cols = df.columns[df.isna().any()].tolist()
    if partial_nan_cols:
        warnings.warn(
            f"NaN values found in columns: {partial_nan_cols}. "
            "These will cause errors in downstream modules. "
            "Consider imputing missing values before calling this function.",
            UserWarning,
        )

    # Check for infinite values
    if np.isinf(df.values).any():
        raise ValueError(
            "Infinite values found in the DataFrame. "
            "Please clean your data before calling this function."
        )

    # Check sufficient observations
    n_obs = df.shape[0]
    if n_obs < config.window_length:
        raise ValueError(
            f"DataFrame has {n_obs} rows but window_length is "
            f"{config.window_length}. Need at least {config.window_length} "
            "observations to create one window."
        )

    # Warn if very few windows will be created
    n_windows = (n_obs - config.window_length) // config.stride + 1
    if n_windows < 20:
        warnings.warn(
            f"Only {n_windows} windows will be created with window_length="
            f"{config.window_length} and stride={config.stride}. "
            "This may be insufficient for the VAE to learn meaningful "
            "cross-variable structure. Consider reducing window_length "
            "or using a smaller stride.",
            UserWarning,
        )

    # Warn if index does not appear monotonically ordered
    if hasattr(df.index, "is_monotonic_increasing"):
        if not df.index.is_monotonic_increasing:
            warnings.warn(
                "DataFrame index is not monotonically increasing. "
                "Ensure your data is sorted chronologically before "
                "calling this function, as window construction assumes "
                "temporal order.",
                UserWarning,
            )

    return df


def _build_scaler(config: PipelineConfig):
    """
    Instantiate the appropriate sklearn scaler based on config.

    Parameters
    ----------
    config : PipelineConfig

    Returns
    -------
    scaler : MinMaxScaler or StandardScaler
    """
    if config.scaler_type == "minmax":
        return MinMaxScaler(feature_range=config.feature_range)
    elif config.scaler_type == "standard":
        return StandardScaler()
    else:
        raise ValueError(
            f"scaler_type must be 'minmax' or 'standard', "
            f"got '{config.scaler_type}'."
        )


def _normalise(
    data: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, object]:
    """
    Fit a scaler on the full training data and return normalised data
    and the fitted scaler.

    The scaler is fit on the 2D array of shape (num_observations, num_variables)
    before windowing. This ensures that normalisation is consistent across
    all windows and that the scaler is not influenced by window boundaries.

    Parameters
    ----------
    data : np.ndarray
        Shape (num_observations, num_variables). Raw unnormalised values.
    config : PipelineConfig

    Returns
    -------
    normalised : np.ndarray
        Shape (num_observations, num_variables). Scaled values.
    scaler : fitted sklearn scaler
        Must be saved and reused for inverse_transform at generation time.
    """
    scaler = _build_scaler(config)
    normalised = scaler.fit_transform(data)
    return normalised, scaler


def _build_windows(
    data: np.ndarray,
    window_length: int,
    stride: int,
) -> np.ndarray:
    """
    Construct sliding windows from a 2D array of normalised observations.

    Parameters
    ----------
    data : np.ndarray
        Shape (num_observations, num_variables). Normalised time series.
    window_length : int
        Number of timesteps per window.
    stride : int
        Step size between windows.

    Returns
    -------
    windows : np.ndarray
        Shape (num_windows, window_length, num_variables).
    """
    num_obs, num_vars = data.shape
    indices = range(0, num_obs - window_length + 1, stride)
    windows = np.stack(
        [data[i : i + window_length] for i in indices],
        axis=0,
    )
    return windows


# =============================================================================
# Scaler Persistence
# =============================================================================

def save_scaler(scaler, path: str) -> None:
    """
    Save a fitted sklearn scaler to disk using pickle.

    Parameters
    ----------
    scaler : fitted sklearn scaler
        As returned by build_pipeline.
    path : str
        File path (e.g., 'scaler.pkl').
    """
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str):
    """
    Load a saved sklearn scaler from disk.

    Use this to reload the fitted scaler in a separate session —
    for example when inverting normalisation on generated synthetic data.

    Parameters
    ----------
    path : str
        Path to a scaler saved by save_scaler or build_pipeline.

    Returns
    -------
    scaler : fitted sklearn scaler

    Raises
    ------
    FileNotFoundError
        If no file exists at the given path.

    Examples
    --------
    >>> scaler = load_scaler("scaler.pkl")
    >>> original_scale = inverse_transform(synthetic_windows, scaler)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No scaler found at '{path}'. "
            "Ensure you have run build_pipeline and saved the scaler first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Inverse Transform
# =============================================================================

def inverse_transform(
    windows: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Invert normalisation on windowed data to recover original scale.

    Use this after decoding synthetic latent sequences through the VAE
    decoder (Module 2) to convert normalised synthetic series back to
    their original units.

    Parameters
    ----------
    windows : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Normalised data as produced by build_pipeline or VAE decode.
    scaler : fitted sklearn scaler
        The same scaler object that was used during normalisation.
        Load with load_scaler if in a new session.

    Returns
    -------
    original_scale : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Data in original units.

    Raises
    ------
    ValueError
        If windows is not a 3D array.

    Examples
    --------
    >>> reconstructed = decode(latent, vae_model)          # Module 2
    >>> original = inverse_transform(reconstructed, scaler) # Module 1
    """
    if not isinstance(windows, np.ndarray) or windows.ndim != 3:
        raise ValueError(
            "windows must be a 3D numpy array of shape "
            "(num_windows, window_length, num_variables)."
        )

    num_windows, window_length, num_variables = windows.shape

    # Reshape to 2D for sklearn: (num_windows * window_length, num_variables)
    flat = windows.reshape(-1, num_variables)
    flat_original = scaler.inverse_transform(flat)

    # Reshape back to 3D
    return flat_original.reshape(num_windows, window_length, num_variables)


# =============================================================================
# Public API
# =============================================================================

def build_pipeline(
    df: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
    scaler_save_path: Optional[str] = "scaler.pkl",
) -> Tuple[np.ndarray, object]:
    """
    Full data pipeline: validate, normalise, and construct sliding windows
    from a raw training DataFrame.

    This is the primary entry point for Module 1. Pass in your training
    DataFrame and receive a windowed numpy array ready for Module 2 (VAE).

    Parameters
    ----------
    df : pd.DataFrame
        Training data. Rows are timesteps (e.g. months), columns are
        variables. Must contain only the training portion — train/test
        splitting should be done upstream before calling this function.
    config : Optional[PipelineConfig]
        Pipeline configuration. If None, sensible defaults are used
        (window_length=24, stride=1, MinMaxScaler to [0,1]).
    scaler_save_path : Optional[str]
        Path to save the fitted scaler. If None, scaler is not saved to
        disk. Strongly recommended to save for use at generation time.

    Returns
    -------
    windows : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Normalised, windowed training data ready for Module 2.
    scaler : fitted sklearn scaler
        Fitted on the full training data. Use inverse_transform() to
        convert generated data back to original scale.

    Examples
    --------
    >>> import pandas as pd
    >>> from data_pipeline import build_pipeline, inverse_transform, load_scaler
    >>>
    >>> df = pd.read_csv("my_training_data.csv", index_col=0, parse_dates=True)
    >>> windows, scaler = build_pipeline(df, scaler_save_path="scaler.pkl")
    >>> print(windows.shape)  # (num_windows, 24, num_variables)
    """
    if config is None:
        config = PipelineConfig()

    # --- Validate and clean DataFrame ---
    df_clean = _validate_dataframe(df, config)

    num_obs, num_vars = df_clean.shape
    n_windows = (num_obs - config.window_length) // config.stride + 1

    if config.verbose:
        print(f"[Pipeline] Observations: {num_obs} | Variables: {num_vars}")
        print(f"[Pipeline] Columns: {df_clean.columns.tolist()}")
        print(f"[Pipeline] Window length: {config.window_length} | Stride: {config.stride}")
        print(f"[Pipeline] Windows to be created: {n_windows}")
        print(f"[Pipeline] Scaler: {config.scaler_type}")
        print("-" * 60)

    # --- Normalise on raw 2D observations before windowing ---
    raw = df_clean.values.astype(np.float32)
    normalised, scaler = _normalise(raw, config)

    if config.verbose:
        print(f"[Pipeline] Normalisation complete.")
        if config.scaler_type == "minmax":
            print(f"[Pipeline] Data range after scaling: "
                  f"[{normalised.min():.4f}, {normalised.max():.4f}]")
        else:
            print(f"[Pipeline] Mean after scaling: {normalised.mean():.4f} "
                  f"| Std: {normalised.std():.4f}")

    # --- Build sliding windows ---
    windows = _build_windows(normalised, config.window_length, config.stride)

    if config.verbose:
        print(f"[Pipeline] Windows created: {windows.shape[0]}")
        print(f"[Pipeline] Output shape: {windows.shape}  "
              f"(num_windows, window_length, num_variables)")

    # --- Save scaler ---
    if scaler_save_path is not None:
        save_scaler(scaler, scaler_save_path)
        if config.verbose:
            print(f"[Pipeline] Scaler saved to: {scaler_save_path}")

    if config.verbose:
        print("-" * 60)
        print("[Pipeline] Pipeline complete. Ready for Module 2 (VAE).")

    return windows, scaler


def get_pipeline_summary(
    df: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
) -> dict:
    """
    Return a summary of what the pipeline will produce without running it.

    Useful for quickly checking how many windows will be generated for
    different window_length and stride combinations before committing
    to a configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame (same format as build_pipeline).
    config : Optional[PipelineConfig]
        Pipeline configuration. If None, uses defaults.

    Returns
    -------
    summary : dict
        Dictionary with keys:
        - 'num_observations': int
        - 'num_variables': int
        - 'column_names': list
        - 'window_length': int
        - 'stride': int
        - 'num_windows': int
        - 'scaler_type': str
        - 'has_nan': bool
        - 'has_inf': bool

    Examples
    --------
    >>> summary = get_pipeline_summary(df)
    >>> print(summary['num_windows'])
    """
    if config is None:
        config = PipelineConfig()

    numeric_df = df.select_dtypes(include=[np.number])
    num_obs, num_vars = numeric_df.shape
    n_windows = max(0, (num_obs - config.window_length) // config.stride + 1)

    return {
        "num_observations": num_obs,
        "num_variables": num_vars,
        "column_names": numeric_df.columns.tolist(),
        "window_length": config.window_length,
        "stride": config.stride,
        "num_windows": n_windows,
        "scaler_type": config.scaler_type,
        "has_nan": bool(numeric_df.isna().any().any()),
        "has_inf": bool(np.isinf(numeric_df.values).any()),
    }