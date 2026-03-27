"""
evaluation_module.py
====================
Module 5: Evaluation suite for synthetic multivariate time series.

Provides both individual metric functions and a unified evaluate_all()
wrapper that runs all metrics, produces a multi-panel figure, and saves
a text and JSON report.

Metrics included
----------------
1. Statistical Moments       — mean, std, min, max per variable
2. Full ACF Comparison       — autocorrelation up to lag K per variable
3. Cross-Correlation Matrix  — Frobenius norm between real and synthetic
4. MMD                       — Maximum Mean Discrepancy on flattened windows
5. Discriminative Score      — logistic regression two-sample test
6. t-SNE / PCA Visualisation — qualitative diversity and overlap check
7. NN Distance Ratio         — memorisation check (optional)

Design principles
-----------------
- All functions accept pre-generated numpy arrays in normalised space.
  Pass real windows from Module 1 and synthetic windows from Module 3
  generate(). Decoupled from any specific generator.
- Individual metric functions are independently callable for flexibility
  during development.
- evaluate_all() calls all enabled metrics, prints a verbose summary,
  saves a multi-panel figure (.png), a structured text report (.txt),
  and a machine-readable results file (.json).
- Designed for ablation studies: call evaluate_all() with outputs from
  jittering, standard TimeGAN, and VAE-TimeGAN using identical inputs.

Usage
-----
from evaluation_module import EvaluationConfig, evaluate_all

config = EvaluationConfig(
    n_lags=12,
    report_save_path="eval_report.txt",
    json_save_path="eval_results.json",
    figure_save_path="eval_figure.png",
)

results = evaluate_all(
    real=real_windows,          # (n, seq_len, n_vars) normalised
    synthetic=synthetic_windows, # (n, seq_len, n_vars) normalised
    config=config,
    held_out=oos_windows,       # optional — enables NN distance ratio
    variable_names=["TB3MS", "WTI_price"],
)
"""

# =============================================================================
# Imports
# =============================================================================

import json
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation suite.

    Parameters
    ----------
    n_lags : int
        Number of lags for ACF comparison. Default 12.
    mmd_kernel_bandwidth : Optional[float]
        Bandwidth for the RBF kernel in MMD. If None, uses the median
        heuristic (recommended — adapts to data scale automatically).
    tsne_perplexity : float
        Perplexity for t-SNE. Typical range 5-50. Default 30.
    tsne_max_samples : int
        Maximum number of samples to use for t-SNE (capped for speed).
        Default 300.
    nn_n_neighbours : int
        Number of nearest neighbours for the NN distance ratio metric.
        Default 5.
    discriminative_n_splits : int
        Number of train/test splits for discriminative score averaging.
        Default 3. Averaging over multiple splits reduces variance.
    random_seed : Optional[int]
        Seed for reproducibility across t-SNE and discriminative score.
    verbose : bool
        If True, prints metric results as they are computed. Default True.
    report_save_path : Optional[str]
        Path to save the text report. If None, not saved.
    json_save_path : Optional[str]
        Path to save the JSON results. If None, not saved.
    figure_save_path : Optional[str]
        Path to save the multi-panel figure. If None, not saved.
    """
    n_lags: int = 12
    mmd_kernel_bandwidth: Optional[float] = None
    tsne_perplexity: float = 30.0
    tsne_max_samples: int = 300
    nn_n_neighbours: int = 5
    discriminative_n_splits: int = 3
    random_seed: Optional[int] = None
    verbose: bool = True
    report_save_path: Optional[str] = "eval_report.txt"
    json_save_path: Optional[str] = "eval_results.json"
    figure_save_path: Optional[str] = "eval_figure.png"


# =============================================================================
# Internal Utilities
# =============================================================================

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _validate_inputs(
    real: np.ndarray,
    synthetic: np.ndarray,
    held_out: Optional[np.ndarray] = None,
) -> None:
    """Validate shapes and contents of input arrays."""
    for name, arr in [("real", real), ("synthetic", synthetic)]:
        if not isinstance(arr, np.ndarray) or arr.ndim != 3:
            raise ValueError(
                f"'{name}' must be a 3D numpy array of shape "
                "(n_windows, seq_len, n_vars)."
            )
        if np.isnan(arr).any():
            raise ValueError(f"'{name}' contains NaN values.")
        if np.isinf(arr).any():
            raise ValueError(f"'{name}' contains infinite values.")

    if real.shape[1:] != synthetic.shape[1:]:
        raise ValueError(
            f"real and synthetic must have the same seq_len and n_vars. "
            f"Got real {real.shape[1:]} vs synthetic {synthetic.shape[1:]}."
        )

    if held_out is not None:
        if not isinstance(held_out, np.ndarray) or held_out.ndim != 3:
            raise ValueError(
                "held_out must be a 3D numpy array of shape "
                "(n_windows, seq_len, n_vars)."
            )
        if held_out.shape[1:] != real.shape[1:]:
            raise ValueError(
                "held_out must have the same seq_len and n_vars as real."
            )


def _get_variable_names(
    n_vars: int,
    variable_names: Optional[List[str]],
) -> List[str]:
    """Return variable names, defaulting to var_0, var_1, ..."""
    if variable_names is not None:
        if len(variable_names) != n_vars:
            warnings.warn(
                f"variable_names has {len(variable_names)} entries but data "
                f"has {n_vars} variables. Defaulting to var_0, var_1, ...",
                UserWarning,
            )
            return [f"var_{i}" for i in range(n_vars)]
        return variable_names
    return [f"var_{i}" for i in range(n_vars)]


# =============================================================================
# Metric 1: Statistical Moments
# =============================================================================

def compute_statistical_moments(
    real: np.ndarray,
    synthetic: np.ndarray,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute and compare per-variable statistical moments between real
    and synthetic data.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars). Normalised real windows.
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars). Normalised synthetic windows.
    variable_names : Optional[List[str]]

    Returns
    -------
    results : dict
        Per-variable mean, std, min, max for real and synthetic,
        plus absolute differences.
    """
    n_vars = real.shape[2]
    var_names = _get_variable_names(n_vars, variable_names)
    results = {"per_variable": {}}

    for v in range(n_vars):
        real_v = real[:, :, v].flatten()
        fake_v = synthetic[:, :, v].flatten()

        r_mean, r_std = float(real_v.mean()), float(real_v.std())
        r_min,  r_max = float(real_v.min()),  float(real_v.max())
        f_mean, f_std = float(fake_v.mean()), float(fake_v.std())
        f_min,  f_max = float(fake_v.min()),  float(fake_v.max())

        results["per_variable"][var_names[v]] = {
            "real":       {"mean": r_mean, "std": r_std, "min": r_min, "max": r_max},
            "synthetic":  {"mean": f_mean, "std": f_std, "min": f_min, "max": f_max},
            "abs_diff":   {
                "mean": abs(r_mean - f_mean),
                "std":  abs(r_std  - f_std),
                "min":  abs(r_min  - f_min),
                "max":  abs(r_max  - f_max),
            },
            "std_ratio": f_std / r_std if r_std > 1e-8 else float("nan"),
        }

    return results


# =============================================================================
# Metric 2: ACF Comparison
# =============================================================================

def compute_acf_comparison(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_lags: int = 12,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute and compare full ACF profiles up to n_lags for each variable.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    n_lags : int
        Number of lags to compute. Default 12.
    variable_names : Optional[List[str]]

    Returns
    -------
    results : dict
        Per-variable real and synthetic ACF arrays, plus mean absolute
        error between them across all lags.
    """
    n_vars = real.shape[2]
    var_names = _get_variable_names(n_vars, variable_names)
    results = {"per_variable": {}, "mean_acf_mae": 0.0}

    maes = []
    for v in range(n_vars):
        real_series  = real[:, :, v].flatten()
        fake_series  = synthetic[:, :, v].flatten()

        real_acf = acf(real_series, nlags=n_lags, fft=True)
        fake_acf = acf(fake_series, nlags=n_lags, fft=True)

        # Skip lag 0 (always 1.0) when computing MAE
        mae = float(np.mean(np.abs(real_acf[1:] - fake_acf[1:])))
        maes.append(mae)

        results["per_variable"][var_names[v]] = {
            "real_acf":      real_acf.tolist(),
            "synthetic_acf": fake_acf.tolist(),
            "mae":           mae,
        }

    results["mean_acf_mae"] = float(np.mean(maes))
    return results


# =============================================================================
# Metric 3: Cross-Correlation Matrix Distance
# =============================================================================

def compute_cross_correlation(
    real: np.ndarray,
    synthetic: np.ndarray,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute cross-variable correlation matrices for real and synthetic data
    and measure their distance via Frobenius norm.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    variable_names : Optional[List[str]]

    Returns
    -------
    results : dict
        Real and synthetic correlation matrices and their Frobenius distance.
        Frobenius norm closer to 0 = better cross-variable preservation.
    """
    var_names = _get_variable_names(real.shape[2], variable_names)

    # Flatten to (n_windows * seq_len, n_vars) for correlation computation
    real_flat = real.reshape(-1, real.shape[2])
    fake_flat = synthetic.reshape(-1, synthetic.shape[2])

    real_corr = np.corrcoef(real_flat.T)
    fake_corr = np.corrcoef(fake_flat.T)

    frob_norm = float(np.linalg.norm(real_corr - fake_corr, "fro"))

    return {
        "real_correlation":      real_corr.tolist(),
        "synthetic_correlation": fake_corr.tolist(),
        "frobenius_norm":        frob_norm,
        "variable_names":        var_names,
        "interpretation":        "closer to 0 = better cross-variable preservation",
    }


# =============================================================================
# Metric 4: Maximum Mean Discrepancy (MMD)
# =============================================================================

def _rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Radial basis function kernel matrix between X and Y."""
    # ||x - y||^2 via expansion: ||x||^2 + ||y||^2 - 2 x^T y
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    sq_dists = XX + YY.T - 2 * XY
    return np.exp(-sq_dists / (2 * bandwidth ** 2))


def compute_mmd(
    real: np.ndarray,
    synthetic: np.ndarray,
    bandwidth: Optional[float] = None,
    max_samples: int = 500,
) -> Dict:
    """
    Compute Maximum Mean Discrepancy (MMD) between real and synthetic
    distributions using a Gaussian RBF kernel on flattened windows.

    MMD = 0 indicates identical distributions. Higher values indicate
    greater distributional divergence.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    bandwidth : Optional[float]
        RBF kernel bandwidth. If None, uses median heuristic.
    max_samples : int
        Cap on samples used for MMD computation (quadratic complexity).
        Default 500.

    Returns
    -------
    results : dict
        MMD value, kernel bandwidth used, and interpretation.
        MMD closer to 0 = better distributional match.
    """
    # Flatten windows to (n, seq_len * n_vars)
    real_flat = real.reshape(len(real), -1).astype(np.float64)
    fake_flat = synthetic.reshape(len(synthetic), -1).astype(np.float64)

    # Cap samples for computational tractability
    n = min(len(real_flat), len(fake_flat), max_samples)
    idx_r = np.random.choice(len(real_flat), n, replace=False)
    idx_f = np.random.choice(len(fake_flat), n, replace=False)
    X = real_flat[idx_r]
    Y = fake_flat[idx_f]

    # Median heuristic for bandwidth if not provided
    if bandwidth is None:
        combined = np.concatenate([X, Y], axis=0)
        pairwise_sq = np.sum(
            (combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1
        )
        median_sq = np.median(pairwise_sq[pairwise_sq > 0])
        bandwidth = float(np.sqrt(0.5 * median_sq))
        bandwidth = max(bandwidth, 1e-6)  # prevent zero bandwidth

    K_XX = _rbf_kernel(X, X, bandwidth)
    K_YY = _rbf_kernel(Y, Y, bandwidth)
    K_XY = _rbf_kernel(X, Y, bandwidth)

    # Unbiased MMD estimate — exclude diagonal terms
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd = (
        K_XX.sum() / (n * (n - 1))
        + K_YY.sum() / (n * (n - 1))
        - 2 * K_XY.mean()
    )

    return {
        "mmd": float(mmd),
        "bandwidth_used": float(bandwidth),
        "n_samples_used": n,
        "interpretation": "closer to 0 = better distributional match",
    }


# =============================================================================
# Metric 5: Discriminative Score
# =============================================================================

def compute_discriminative_score(
    real: np.ndarray,
    synthetic: np.ndarray,
    n_splits: int = 3,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Compute discriminative score using logistic regression.

    Trains a classifier to distinguish real from synthetic sequences
    using mean-pooled features. Accuracy close to 0.5 means the synthetic
    data is indistinguishable from real (best). Accuracy close to 1.0
    means it is easily distinguishable (worst).

    Averaged over n_splits random train/test splits to reduce variance.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars).
    n_splits : int
        Number of random splits to average over. Default 3.
    random_seed : Optional[int]

    Returns
    -------
    results : dict
        Mean accuracy, std across splits, and per-split scores.
        Accuracy closer to 0.5 = better.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = min(len(real), len(synthetic))

    # Mean-pool across time: (n, seq_len, n_vars) -> (n, n_vars)
    # Also include std-pool to capture variance information
    real_mean = real[:n].mean(axis=1)
    fake_mean = synthetic[:n].mean(axis=1)
    real_std  = real[:n].std(axis=1)
    fake_std  = synthetic[:n].std(axis=1)

    real_feats = np.concatenate([real_mean, real_std], axis=1)
    fake_feats = np.concatenate([fake_mean, fake_std], axis=1)

    X = np.concatenate([real_feats, fake_feats], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    split_scores = []
    for split in range(n_splits):
        idx = np.random.permutation(len(y))
        X_s, y_s = X[idx], y[idx]
        cut = int(0.8 * len(y_s))
        X_train, X_test = X_s[:cut], X_s[cut:]
        y_train, y_test = y_s[:cut], y_s[cut:]

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=split)),
        ])
        try:
            clf.fit(X_train, y_train)
            score = float(clf.score(X_test, y_test))
        except Exception:
            score = 1.0
        split_scores.append(score)

    mean_score = float(np.mean(split_scores))
    std_score  = float(np.std(split_scores))

    return {
        "mean_accuracy":  mean_score,
        "std_accuracy":   std_score,
        "per_split":      split_scores,
        "interpretation": "0.5 = indistinguishable (best), 1.0 = perfectly distinguishable (worst)",
    }


# =============================================================================
# Metric 6: NN Distance Ratio (Memorisation Check)
# =============================================================================

def compute_nn_distance_ratio(
    real_train: np.ndarray,
    synthetic: np.ndarray,
    held_out: Optional[np.ndarray] = None,
    n_neighbours: int = 5,
    max_samples: int = 300,
) -> Dict:
    """
    Compute nearest-neighbour distance ratio to detect memorisation.

    For each synthetic sample, computes:
    - d_train: distance to nearest neighbour in training set
    - d_ref:   distance to nearest neighbour in held_out set (if provided)
              or in other synthetic samples (if held_out not provided)

    A ratio d_train / d_ref close to 1.0 means synthetic samples are
    as close to training data as to unseen data — good generalisation.
    A ratio much less than 1.0 means synthetic samples are much closer
    to training data than to unseen data — potential memorisation.

    Parameters
    ----------
    real_train : np.ndarray
        Shape (n, seq_len, n_vars). Training windows.
    synthetic : np.ndarray
        Shape (n, seq_len, n_vars). Generated windows.
    held_out : Optional[np.ndarray]
        Shape (n, seq_len, n_vars). OOS real windows if available.
        If None, uses other synthetic samples as reference.
    n_neighbours : int
        Number of nearest neighbours. Default 5.
    max_samples : int
        Cap on samples for computational tractability. Default 300.

    Returns
    -------
    results : dict
        Mean ratio, std, and interpretation.
        Ratio close to 1.0 = good generalisation.
        Ratio << 1.0 = potential memorisation.
    """
    # Flatten all arrays
    train_flat = real_train.reshape(len(real_train), -1).astype(np.float32)
    fake_flat  = synthetic.reshape(len(synthetic), -1).astype(np.float32)

    n = min(len(fake_flat), max_samples)
    idx = np.random.choice(len(fake_flat), n, replace=False)
    fake_sub = fake_flat[idx]

    # Distance to training set
    nn_train = NearestNeighbors(n_neighbors=n_neighbours, metric="euclidean")
    nn_train.fit(train_flat)
    d_train, _ = nn_train.kneighbors(fake_sub)
    d_train_mean = d_train.mean(axis=1)

    # Distance to reference set
    if held_out is not None:
        ref_flat = held_out.reshape(len(held_out), -1).astype(np.float32)
        nn_ref = NearestNeighbors(n_neighbors=n_neighbours, metric="euclidean")
        nn_ref.fit(ref_flat)
        d_ref, _ = nn_ref.kneighbors(fake_sub)
        ref_label = "held_out"
    else:
        # Use other synthetic samples as reference
        nn_ref = NearestNeighbors(n_neighbors=n_neighbours + 1, metric="euclidean")
        nn_ref.fit(fake_flat)
        d_ref_all, _ = nn_ref.kneighbors(fake_sub)
        d_ref = d_ref_all[:, 1:]  # exclude self (distance 0)
        ref_label = "other_synthetic"

    d_ref_mean = d_ref.mean(axis=1)

    # Avoid division by zero
    valid = d_ref_mean > 1e-8
    ratios = d_train_mean[valid] / d_ref_mean[valid]

    return {
        "mean_ratio":    float(ratios.mean()) if len(ratios) > 0 else float("nan"),
        "std_ratio":     float(ratios.std())  if len(ratios) > 0 else float("nan"),
        "reference_set": ref_label,
        "n_samples":     int(valid.sum()),
        "interpretation": (
            "ratio ~ 1.0 = good generalisation | "
            "ratio << 1.0 = potential memorisation of training data"
        ),
    }


# =============================================================================
# Metric 7: t-SNE and PCA Visualisation (returns figure axes data)
# =============================================================================

def compute_dimensionality_reduction(
    real: np.ndarray,
    synthetic: np.ndarray,
    tsne_perplexity: float = 30.0,
    max_samples: int = 300,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Compute PCA and t-SNE embeddings of real and synthetic windows
    for visualisation.

    Both methods flatten the temporal dimension before embedding:
    (n, seq_len, n_vars) -> (n, seq_len * n_vars)

    Parameters
    ----------
    real : np.ndarray
    synthetic : np.ndarray
    tsne_perplexity : float
    max_samples : int
        Cap on samples for t-SNE speed. Default 300.
    random_seed : Optional[int]

    Returns
    -------
    results : dict
        PCA and t-SNE 2D coordinates for real and synthetic samples.
    """
    seed = random_seed if random_seed is not None else 42

    n = min(len(real), len(synthetic), max_samples)
    real_sub = real[:n].reshape(n, -1)
    fake_sub = synthetic[:n].reshape(n, -1)
    combined = np.concatenate([real_sub, fake_sub], axis=0)

    # PCA
    pca = PCA(n_components=2, random_state=seed)
    pca_embedded = pca.fit_transform(combined)

    # t-SNE — adjust perplexity if sample size is too small
    effective_perplexity = min(tsne_perplexity, (len(combined) - 1) / 3)
    effective_perplexity = max(effective_perplexity, 5.0)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=seed,
        max_iter=1000,
    )
    tsne_embedded = tsne.fit_transform(combined)

    return {
        "pca": {
            "real":      pca_embedded[:n].tolist(),
            "synthetic": pca_embedded[n:].tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "tsne": {
            "real":      tsne_embedded[:n].tolist(),
            "synthetic": tsne_embedded[n:].tolist(),
            "perplexity_used": float(effective_perplexity),
        },
        "n_samples": n,
    }


# =============================================================================
# Figure Generation
# =============================================================================

def _build_evaluation_figure(
    real: np.ndarray,
    synthetic: np.ndarray,
    results: Dict,
    variable_names: List[str],
    config: EvaluationConfig,
) -> plt.Figure:
    """
    Build the multi-panel evaluation figure combining all visual diagnostics.

    Layout:
    Row 1: PCA | t-SNE
    Row 2: ACF per variable (up to 4 variables shown)
    Row 3: Real correlation heatmap | Synthetic correlation heatmap | Moment comparison
    """
    n_vars = real.shape[2]
    acf_cols = min(n_vars, 4)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, max(acf_cols, 3), figure=fig, hspace=0.45, wspace=0.35)

    # --- Row 1: PCA and t-SNE ---
    dr = results.get("dimensionality_reduction", {})

    ax_pca = fig.add_subplot(gs[0, :2])
    if dr:
        pca_real = np.array(dr["pca"]["real"])
        pca_fake = np.array(dr["pca"]["synthetic"])
        ax_pca.scatter(pca_real[:, 0], pca_real[:, 1],
                       c="steelblue", alpha=0.4, s=12, label="Real")
        ax_pca.scatter(pca_fake[:, 0], pca_fake[:, 1],
                       c="tomato", alpha=0.4, s=12, label="Synthetic")
        ev = dr["pca"]["explained_variance_ratio"]
        ax_pca.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax_pca.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax_pca.set_title("PCA: Real vs Synthetic")
    ax_pca.legend(markerscale=2)
    ax_pca.grid(True, alpha=0.3)

    ax_tsne = fig.add_subplot(gs[0, 2:])
    if dr:
        tsne_real = np.array(dr["tsne"]["real"])
        tsne_fake = np.array(dr["tsne"]["synthetic"])
        ax_tsne.scatter(tsne_real[:, 0], tsne_real[:, 1],
                        c="steelblue", alpha=0.4, s=12, label="Real")
        ax_tsne.scatter(tsne_fake[:, 0], tsne_fake[:, 1],
                        c="tomato", alpha=0.4, s=12, label="Synthetic")
    ax_tsne.set_title("t-SNE: Real vs Synthetic")
    ax_tsne.legend(markerscale=2)
    ax_tsne.grid(True, alpha=0.3)

    # --- Row 2: ACF comparison per variable ---
    acf_data = results.get("acf", {}).get("per_variable", {})
    for v_idx, v_name in enumerate(variable_names[:acf_cols]):
        ax_acf = fig.add_subplot(gs[1, v_idx])
        if v_name in acf_data:
            lags = list(range(len(acf_data[v_name]["real_acf"])))
            ax_acf.plot(lags, acf_data[v_name]["real_acf"],
                        marker="o", markersize=3, label="Real", linewidth=1.5)
            ax_acf.plot(lags, acf_data[v_name]["synthetic_acf"],
                        marker="s", markersize=3, label="Synthetic",
                        linewidth=1.5, linestyle="--")
            mae = acf_data[v_name]["mae"]
            ax_acf.set_title(f"ACF: {v_name}\n(MAE={mae:.4f})")
        else:
            ax_acf.set_title(f"ACF: {v_name}")
        ax_acf.axhline(0, color="black", linewidth=0.5)
        ax_acf.set_xlabel("Lag")
        ax_acf.legend(fontsize=7)
        ax_acf.grid(True, alpha=0.3)

    # --- Row 3: Correlation heatmaps and moment comparison ---
    corr_data = results.get("cross_correlation", {})

    ax_corr_real = fig.add_subplot(gs[2, 0])
    if corr_data:
        real_corr = np.array(corr_data["real_correlation"])
        im1 = ax_corr_real.imshow(real_corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_corr_real.set_title("Real Correlation Matrix")
        ax_corr_real.set_xticks(range(n_vars))
        ax_corr_real.set_yticks(range(n_vars))
        ax_corr_real.set_xticklabels(variable_names, rotation=45, ha="right", fontsize=8)
        ax_corr_real.set_yticklabels(variable_names, fontsize=8)
        plt.colorbar(im1, ax=ax_corr_real, fraction=0.046)
        for i in range(n_vars):
            for j in range(n_vars):
                ax_corr_real.text(j, i, f"{real_corr[i,j]:.2f}",
                                  ha="center", va="center", fontsize=7)

    ax_corr_fake = fig.add_subplot(gs[2, 1])
    if corr_data:
        fake_corr = np.array(corr_data["synthetic_correlation"])
        frob = corr_data["frobenius_norm"]
        im2 = ax_corr_fake.imshow(fake_corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_corr_fake.set_title(f"Synthetic Correlation Matrix\n(Frob norm={frob:.4f})")
        ax_corr_fake.set_xticks(range(n_vars))
        ax_corr_fake.set_yticks(range(n_vars))
        ax_corr_fake.set_xticklabels(variable_names, rotation=45, ha="right", fontsize=8)
        ax_corr_fake.set_yticklabels(variable_names, fontsize=8)
        plt.colorbar(im2, ax=ax_corr_fake, fraction=0.046)
        for i in range(n_vars):
            for j in range(n_vars):
                ax_corr_fake.text(j, i, f"{fake_corr[i,j]:.2f}",
                                  ha="center", va="center", fontsize=7)

    # Moment comparison bar chart
    ax_moments = fig.add_subplot(gs[2, 2:])
    moments_data = results.get("statistical_moments", {}).get("per_variable", {})
    if moments_data:
        x = np.arange(len(variable_names))
        width = 0.35
        real_stds = [moments_data[v]["real"]["std"] for v in variable_names if v in moments_data]
        fake_stds = [moments_data[v]["synthetic"]["std"] for v in variable_names if v in moments_data]
        ax_moments.bar(x - width/2, real_stds, width, label="Real std", color="steelblue", alpha=0.8)
        ax_moments.bar(x + width/2, fake_stds, width, label="Synthetic std", color="tomato", alpha=0.8)
        ax_moments.set_xticks(x)
        ax_moments.set_xticklabels(
            [v for v in variable_names if v in moments_data],
            rotation=45, ha="right", fontsize=8
        )
        ax_moments.set_title("Std Comparison (Real vs Synthetic)")
        ax_moments.set_ylabel("Std")
        ax_moments.legend()
        ax_moments.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Synthetic Data Evaluation Report", fontsize=14, fontweight="bold")
    return fig


# =============================================================================
# Report Generation
# =============================================================================

def _format_text_report(
    results: Dict,
    variable_names: List[str],
    config: EvaluationConfig,
    timestamp: str,
) -> str:
    """Format a human-readable text report from the results dictionary."""
    lines = []
    sep = "=" * 65
    thin = "-" * 65

    lines.append(sep)
    lines.append("  SYNTHETIC TIME SERIES EVALUATION REPORT")
    lines.append(f"  Generated: {timestamp}")
    lines.append(sep)

    # Statistical Moments
    lines.append("\n1. STATISTICAL MOMENTS")
    lines.append(thin)
    moments = results.get("statistical_moments", {}).get("per_variable", {})
    for v in variable_names:
        if v not in moments:
            continue
        m = moments[v]
        lines.append(f"\n  Variable: {v}")
        lines.append(f"  {'':10s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
        lines.append(f"  {'Real':10s} {m['real']['mean']:>10.4f} {m['real']['std']:>10.4f} "
                     f"{m['real']['min']:>10.4f} {m['real']['max']:>10.4f}")
        lines.append(f"  {'Synthetic':10s} {m['synthetic']['mean']:>10.4f} {m['synthetic']['std']:>10.4f} "
                     f"{m['synthetic']['min']:>10.4f} {m['synthetic']['max']:>10.4f}")
        lines.append(f"  Std ratio (synth/real): {m['std_ratio']:.4f}  "
                     f"(1.0 = perfect match)")

    # ACF
    lines.append("\n2. ACF COMPARISON")
    lines.append(thin)
    acf_data = results.get("acf", {})
    lines.append(f"  Mean ACF MAE across all variables: "
                 f"{acf_data.get('mean_acf_mae', float('nan')):.4f}")
    for v, vdata in acf_data.get("per_variable", {}).items():
        lines.append(f"  {v}: ACF MAE = {vdata['mae']:.4f}")

    # Cross-correlation
    lines.append("\n3. CROSS-CORRELATION MATRIX DISTANCE")
    lines.append(thin)
    corr = results.get("cross_correlation", {})
    lines.append(f"  Frobenius norm: {corr.get('frobenius_norm', float('nan')):.4f}")
    lines.append(f"  (0.0 = perfect match | >0.5 = meaningful divergence)")

    # MMD
    lines.append("\n4. MAXIMUM MEAN DISCREPANCY (MMD)")
    lines.append(thin)
    mmd = results.get("mmd", {})
    lines.append(f"  MMD: {mmd.get('mmd', float('nan')):.6f}")
    lines.append(f"  Kernel bandwidth: {mmd.get('bandwidth_used', float('nan')):.4f}")
    lines.append(f"  Samples used: {mmd.get('n_samples_used', 'N/A')}")
    lines.append(f"  (0.0 = identical distributions)")

    # Discriminative score
    lines.append("\n5. DISCRIMINATIVE SCORE")
    lines.append(thin)
    disc = results.get("discriminative_score", {})
    lines.append(f"  Mean accuracy: {disc.get('mean_accuracy', float('nan')):.4f} "
                 f"(+/- {disc.get('std_accuracy', float('nan')):.4f})")
    lines.append(f"  Per-split: {[round(s, 4) for s in disc.get('per_split', [])]}")
    lines.append(f"  (0.5 = indistinguishable | 1.0 = perfectly distinguishable)")

    # NN Distance Ratio
    nn = results.get("nn_distance_ratio")
    if nn is not None:
        lines.append("\n6. NN DISTANCE RATIO (MEMORISATION CHECK)")
        lines.append(thin)
        lines.append(f"  Mean ratio: {nn.get('mean_ratio', float('nan')):.4f} "
                     f"(+/- {nn.get('std_ratio', float('nan')):.4f})")
        lines.append(f"  Reference set: {nn.get('reference_set', 'N/A')}")
        lines.append(f"  (~1.0 = good generalisation | <<1.0 = memorisation risk)")

    lines.append(f"\n{sep}")
    lines.append("  END OF REPORT")
    lines.append(sep)

    return "\n".join(lines)


# =============================================================================
# Public API — Individual Metrics (already defined above as public functions)
# =============================================================================

def evaluate_all(
    real: np.ndarray,
    synthetic: np.ndarray,
    config: Optional[EvaluationConfig] = None,
    held_out: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    run_nn: bool = False,
) -> Dict:
    """
    Run all evaluation metrics and produce a report, figure, and JSON output.

    Parameters
    ----------
    real : np.ndarray
        Shape (n_windows, seq_len, n_vars). Normalised real training windows.
    synthetic : np.ndarray
        Shape (n_windows, seq_len, n_vars). Normalised synthetic windows
        from any generator (jittering, TimeGAN, VAE-TimeGAN).
    config : Optional[EvaluationConfig]
        Evaluation configuration. If None, uses defaults.
    held_out : Optional[np.ndarray]
        Shape (n_windows, seq_len, n_vars). OOS real windows for the
        NN distance ratio memorisation check. Only used if run_nn=True.
    variable_names : Optional[List[str]]
        Names of variables for reports and figures.
    run_nn : bool
        If True, runs the NN distance ratio metric. Requires more compute.
        Default False. Pass held_out for a more rigorous memorisation check.

    Returns
    -------
    results : Dict
        Dictionary containing all metric results. Keys:
        - 'statistical_moments'
        - 'acf'
        - 'cross_correlation'
        - 'mmd'
        - 'discriminative_score'
        - 'dimensionality_reduction'
        - 'nn_distance_ratio' (only if run_nn=True)
        - 'metadata'

    Examples
    --------
    >>> from evaluation_module import EvaluationConfig, evaluate_all
    >>> config = EvaluationConfig(n_lags=12, random_seed=42)
    >>> results = evaluate_all(real_windows, synthetic_windows, config=config,
    ...                        variable_names=["TB3MS", "WTI_price"])
    >>> print(results["discriminative_score"]["mean_accuracy"])
    >>> print(results["cross_correlation"]["frobenius_norm"])
    """
    if config is None:
        config = EvaluationConfig()

    _validate_inputs(real, synthetic, held_out)

    if config.random_seed is not None:
        _set_seed(config.random_seed)

    n_vars = real.shape[2]
    var_names = _get_variable_names(n_vars, variable_names)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results: Dict = {"metadata": {
        "timestamp":      timestamp,
        "n_real":         len(real),
        "n_synthetic":    len(synthetic),
        "seq_len":        real.shape[1],
        "n_vars":         n_vars,
        "variable_names": var_names,
        "n_lags":         config.n_lags,
    }}

    def _log(msg: str) -> None:
        if config.verbose:
            print(msg)

    _log(f"\n{'='*65}")
    _log("  EVALUATION SUITE — running all metrics")
    _log(f"  Real: {len(real)} windows | Synthetic: {len(synthetic)} windows")
    _log(f"  Variables: {var_names}")
    _log(f"{'='*65}")

    # --- Metric 1: Statistical Moments ---
    _log("\n[1/6] Computing statistical moments...")
    results["statistical_moments"] = compute_statistical_moments(
        real, synthetic, var_names
    )
    if config.verbose:
        for v in var_names:
            m = results["statistical_moments"]["per_variable"][v]
            print(f"  {v}: real_std={m['real']['std']:.4f} | "
                  f"synth_std={m['synthetic']['std']:.4f} | "
                  f"std_ratio={m['std_ratio']:.4f}")

    # --- Metric 2: ACF Comparison ---
    _log("\n[2/6] Computing ACF comparison...")
    results["acf"] = compute_acf_comparison(
        real, synthetic, config.n_lags, var_names
    )
    _log(f"  Mean ACF MAE: {results['acf']['mean_acf_mae']:.4f}")
    for v in var_names:
        mae = results["acf"]["per_variable"][v]["mae"]
        _log(f"  {v}: ACF MAE = {mae:.4f}")

    # --- Metric 3: Cross-Correlation ---
    _log("\n[3/6] Computing cross-correlation matrix distance...")
    results["cross_correlation"] = compute_cross_correlation(
        real, synthetic, var_names
    )
    _log(f"  Frobenius norm: {results['cross_correlation']['frobenius_norm']:.4f}")

    # --- Metric 4: MMD ---
    _log("\n[4/6] Computing MMD...")
    results["mmd"] = compute_mmd(
        real, synthetic,
        bandwidth=config.mmd_kernel_bandwidth,
    )
    _log(f"  MMD: {results['mmd']['mmd']:.6f} "
         f"(bandwidth={results['mmd']['bandwidth_used']:.4f})")

    # --- Metric 5: Discriminative Score ---
    _log("\n[5/6] Computing discriminative score...")
    results["discriminative_score"] = compute_discriminative_score(
        real, synthetic,
        n_splits=config.discriminative_n_splits,
        random_seed=config.random_seed,
    )
    disc = results["discriminative_score"]
    _log(f"  Accuracy: {disc['mean_accuracy']:.4f} (+/- {disc['std_accuracy']:.4f})")
    _log(f"  (0.5=indistinguishable best | 1.0=perfectly distinguishable worst)")

    # --- Metric 6: Dimensionality Reduction ---
    _log("\n[6/6] Computing PCA and t-SNE...")
    results["dimensionality_reduction"] = compute_dimensionality_reduction(
        real, synthetic,
        tsne_perplexity=config.tsne_perplexity,
        max_samples=config.tsne_max_samples,
        random_seed=config.random_seed,
    )
    _log(f"  Done. Samples used: {results['dimensionality_reduction']['n_samples']}")

    # --- Optional: NN Distance Ratio ---
    if run_nn:
        _log("\n[Optional] Computing NN distance ratio (memorisation check)...")
        results["nn_distance_ratio"] = compute_nn_distance_ratio(
            real_train=real,
            synthetic=synthetic,
            held_out=held_out,
            n_neighbours=config.nn_n_neighbours,
        )
        nn = results["nn_distance_ratio"]
        _log(f"  Mean ratio: {nn['mean_ratio']:.4f} (+/- {nn['std_ratio']:.4f})")
        _log(f"  Reference: {nn['reference_set']}")
    else:
        results["nn_distance_ratio"] = None

    # --- Summary ---
    _log(f"\n{'='*65}")
    _log("  SUMMARY")
    _log(f"{'='*65}")
    _log(f"  Discriminative score:    {results['discriminative_score']['mean_accuracy']:.4f}  (target: ~0.5)")
    _log(f"  MMD:                     {results['mmd']['mmd']:.6f}  (target: ~0.0)")
    _log(f"  Frobenius norm (corr):   {results['cross_correlation']['frobenius_norm']:.4f}  (target: ~0.0)")
    _log(f"  Mean ACF MAE:            {results['acf']['mean_acf_mae']:.4f}  (target: ~0.0)")
    _log(f"{'='*65}\n")

    # --- Build and save figure ---
    fig = _build_evaluation_figure(real, synthetic, results, var_names, config)
    if config.figure_save_path is not None:
        fig.savefig(config.figure_save_path, bbox_inches="tight", dpi=150)
        _log(f"[Evaluation] Figure saved to: {config.figure_save_path}")
    plt.close(fig)

    # --- Save text report ---
    report_text = _format_text_report(results, var_names, config, timestamp)
    if config.verbose:
        print(report_text)
    if config.report_save_path is not None:
        with open(config.report_save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        _log(f"[Evaluation] Text report saved to: {config.report_save_path}")

    # --- Save JSON ---
    if config.json_save_path is not None:
        # Convert numpy types for JSON serialisation
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} not JSON serialisable")

        with open(config.json_save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_json_safe)
        _log(f"[Evaluation] JSON results saved to: {config.json_save_path}")

    return results
