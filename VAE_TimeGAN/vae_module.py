"""
vae_module.py
=============
Module 2: Variational Autoencoder (VAE) for cross-variable relationship learning
in multivariate time series. Designed to encode per-timestep multivariate
observations into a structured latent space for downstream use with TimeGAN (Module 3).

Usage
-----
from vae_module import train_vae, encode, decode, load_vae

# Train
model, scaler, history = train_vae(
    data=my_numpy_array,        # shape: (num_windows, window_len, num_variables)
    save_path="vae_checkpoint.pt"
)

# Encode new data
latent_sequences = encode(my_numpy_array, model)

# Decode latent sequences back to data space
reconstructed = decode(latent_sequences, model)

# Load a saved model
model = load_vae("vae_checkpoint.pt")

Notes
-----
- Expects data that has already been normalised upstream (Module 1).
- Per-timestep MLP encoding: each timestep (num_variables,) -> (latent_dim,)
  independently. Temporal relationships are handled by Module 3 (TimeGAN).
- KL annealing uses a linear warmup schedule to mitigate posterior collapse.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for both CPU and GPU envs
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class VAEConfig:
    """
    Configuration for the VAE architecture and training procedure.

    Parameters
    ----------
    input_dim : int
        Number of variables (features) per timestep. Inferred automatically
        from data if not set explicitly.
    latent_dim : int
        Dimensionality of the latent space. Defaults to max(input_dim // 2, 2).
    hidden_dim : int
        Size of the MLP hidden layer. Defaults to max(input_dim * 4, 16).
    num_epochs : int
        Total number of training epochs.
    batch_size : int
        Mini-batch size for training.
    learning_rate : float
        Adam optimiser learning rate.
    kl_warmup_epochs : int
        Number of epochs over which the KL weight is linearly annealed from
        0 to kl_weight_max. Mitigates posterior collapse.
    kl_weight_max : float
        Maximum weight applied to the KL divergence term in the ELBO loss.
    val_fraction : float
        Fraction of windows held out for validation. Must be in (0, 0.5).
    verbose_every : int
        Print training progress every this many epochs. Set to 0 to silence.
    random_seed : Optional[int]
        If provided, seeds Python, NumPy, and PyTorch for reproducibility.
    device : str
        Compute device. 'auto' selects GPU if available, otherwise CPU.
    """
    input_dim: int = 0                  # Set automatically from data if left as 0
    latent_dim: int = 0                 # Set automatically if left as 0
    hidden_dim: int = 0                 # Set automatically if left as 0
    num_epochs: int = 200
    batch_size: int = 32
    learning_rate: float = 1e-3
    kl_warmup_epochs: int = 50
    kl_weight_max: float = 1.0
    val_fraction: float = 0.15
    verbose_every: int = 50
    random_seed: Optional[int] = None
    device: str = "auto"


# =============================================================================
# Internal Utilities
# =============================================================================

def _resolve_device(device_str: str) -> torch.device:
    """Select compute device based on config string."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config_defaults(config: VAEConfig, input_dim: int) -> VAEConfig:
    """
    Fill in any auto-derived config fields based on the actual input dimensionality.
    Returns a new VAEConfig with all fields resolved.
    """
    input_dim_resolved = input_dim
    latent_dim_resolved = config.latent_dim if config.latent_dim > 0 else max(input_dim_resolved // 2, 2)
    hidden_dim_resolved = config.hidden_dim if config.hidden_dim > 0 else max(input_dim_resolved * 4, 16)

    return VAEConfig(
        input_dim=input_dim_resolved,
        latent_dim=latent_dim_resolved,
        hidden_dim=hidden_dim_resolved,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        kl_warmup_epochs=config.kl_warmup_epochs,
        kl_weight_max=config.kl_weight_max,
        val_fraction=config.val_fraction,
        verbose_every=config.verbose_every,
        random_seed=config.random_seed,
        device=config.device,
    )


def _validate_inputs(data: np.ndarray, config: VAEConfig) -> None:
    """
    Validate that the input data and config are well-formed before training.
    Raises informative errors early rather than silently failing mid-training.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Expected data to be a numpy array, got {type(data).__name__}. "
            "Pass your windowed data as a numpy array of shape "
            "(num_windows, window_length, num_variables)."
        )

    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D array of shape (num_windows, window_length, num_variables), "
            f"got shape {data.shape}."
        )

    num_windows, window_len, num_variables = data.shape

    if num_windows < 10:
        warnings.warn(
            f"Very few windows ({num_windows}). VAE may not learn meaningful "
            "cross-variable structure. Consider reducing window length or using "
            "a smaller validation fraction.",
            UserWarning,
        )

    if np.isnan(data).any():
        raise ValueError(
            "Input data contains NaN values. Please handle missing values in "
            "Module 1 before passing data to the VAE."
        )

    if np.isinf(data).any():
        raise ValueError(
            "Input data contains infinite values. Please check your normalisation "
            "in Module 1."
        )

    if not (0.05 <= config.val_fraction <= 0.4):
        raise ValueError(
            f"val_fraction must be between 0.05 and 0.40, got {config.val_fraction}."
        )

    val_size = int(num_windows * config.val_fraction)
    if val_size < 10:
        warnings.warn(
            f"Validation set is very small ({val_size} windows). "
            "Consider increasing val_fraction or collecting more data.",
            UserWarning,
        )


def _prepare_dataloaders(
    data: np.ndarray,
    config: VAEConfig,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """
    Split windowed data into training and validation sets and return DataLoaders.

    The split is done as a clean temporal split — the last val_fraction of windows
    are held out as validation. This avoids leakage from overlapping windows.

    Parameters
    ----------
    data : np.ndarray
        Shape (num_windows, window_length, num_variables).
    config : VAEConfig
        Resolved configuration object.
    device : torch.device
        Compute device.

    Returns
    -------
    train_loader, val_loader : Tuple[DataLoader, DataLoader]
    """
    num_windows = data.shape[0]
    val_size = max(int(num_windows * config.val_fraction), 1)
    train_size = num_windows - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    # Flatten to (num_windows * window_length, num_variables) for per-timestep VAE
    # We keep the window axis for reconstruction but train on flattened timesteps
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# =============================================================================
# VAE Architecture
# =============================================================================

class _Encoder(nn.Module):
    """
    MLP encoder that maps a single timestep observation (num_variables,)
    to the mean and log-variance of the latent distribution.

    Architecture: input -> hidden (ReLU) -> [mu, log_var] (linear)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (..., input_dim). Accepts any leading batch dimensions.

        Returns
        -------
        mu : torch.Tensor
            Shape (..., latent_dim).
        log_var : torch.Tensor
            Shape (..., latent_dim).
        """
        h = self.shared(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var


class _Decoder(nn.Module):
    """
    MLP decoder that maps a latent vector back to the original data space.

    Architecture: latent -> hidden (ReLU) -> output (linear)
    No activation on the output layer — assumes data is normalised upstream
    and reconstruction targets are unbounded.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : torch.Tensor
            Shape (..., latent_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Shape (..., output_dim).
        """
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder for per-timestep multivariate encoding.

    Encodes each timestep independently. Cross-variable relationships are
    captured implicitly by the joint encoder seeing all variables simultaneously
    at each timestep. Temporal relationships are intentionally left for
    Module 3 (TimeGAN) to learn.

    Parameters
    ----------
    config : VAEConfig
        Fully resolved configuration object (all fields must be non-zero).
    """

    def __init__(self, config: VAEConfig):
        super().__init__()

        if config.input_dim == 0:
            raise ValueError(
                "VAEConfig.input_dim must be set before instantiating VAE. "
                "Use _resolve_config_defaults() first."
            )

        self.config = config
        self.encoder = _Encoder(config.input_dim, config.hidden_dim, config.latent_dim)
        self.decoder = _Decoder(config.latent_dim, config.hidden_dim, config.input_dim)

    def reparameterise(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterisation trick: z = mu + eps * std, where eps ~ N(0, I).
        Allows gradients to flow through the sampling operation.

        Parameters
        ----------
        mu : torch.Tensor
            Shape (..., latent_dim).
        log_var : torch.Tensor
            Shape (..., latent_dim).

        Returns
        -------
        z : torch.Tensor
            Sampled latent vector, shape (..., latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, window_len, input_dim) or (batch, input_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed input, same shape as x.
        mu : torch.Tensor
            Encoder mean, shape (..., latent_dim).
        log_var : torch.Tensor
            Encoder log-variance, shape (..., latent_dim).
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# =============================================================================
# Loss Function
# =============================================================================

def _vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    kl_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ELBO loss: reconstruction loss + KL divergence.

    Reconstruction loss is MSE averaged over all elements.
    KL divergence is the analytical KL between N(mu, sigma) and N(0, I),
    averaged over the batch and latent dimensions.

    Parameters
    ----------
    x : torch.Tensor
        Original input.
    x_hat : torch.Tensor
        Reconstructed input.
    mu : torch.Tensor
        Encoder mean.
    log_var : torch.Tensor
        Encoder log-variance.
    kl_weight : float
        Current KL annealing weight (0 at start, kl_weight_max at end of warmup).

    Returns
    -------
    total_loss, recon_loss, kl_loss : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction="mean")

    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


# =============================================================================
# Training Loop (Internal)
# =============================================================================

def _run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    kl_weight: float,
    device: torch.device,
    training: bool,
) -> Tuple[float, float, float]:
    """
    Run one epoch of training or validation.

    Parameters
    ----------
    model : VAE
    loader : DataLoader
    optimizer : Optional optimiser (None during validation)
    kl_weight : float
    device : torch.device
    training : bool
        If True, runs backprop. If False, runs in torch.no_grad().

    Returns
    -------
    avg_total, avg_recon, avg_kl : Tuple[float, float, float]
        Mean losses over all batches in the epoch.
    """
    model.train(training)
    total_losses, recon_losses, kl_losses = [], [], []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for (batch,) in loader:
            batch = batch.to(device)

            # Flatten window dimension: (batch, window_len, input_dim)
            # -> (batch * window_len, input_dim) for per-timestep encoding
            original_shape = batch.shape
            batch_flat = batch.reshape(-1, original_shape[-1])

            x_hat_flat, mu, log_var = model(batch_flat)

            loss, recon_loss, kl_loss = _vae_loss(
                batch_flat, x_hat_flat, mu, log_var, kl_weight
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — important for stability with small datasets
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

    return (
        float(np.mean(total_losses)),
        float(np.mean(recon_losses)),
        float(np.mean(kl_losses)),
    )


# =============================================================================
# Public API
# =============================================================================

def train_vae(
    data: np.ndarray,
    config: Optional[VAEConfig] = None,
    save_path: Optional[str] = "vae_checkpoint.pt",
    plot_save_path: Optional[str] = "vae_loss_curves.png",
) -> Tuple["VAE", Dict[str, List[float]]]:
    """
    Train the VAE on windowed multivariate time series data.

    Parameters
    ----------
    data : np.ndarray
        Windowed data of shape (num_windows, window_length, num_variables).
        Must be normalised upstream in Module 1 before calling this function.
    config : Optional[VAEConfig]
        Training and architecture configuration. If None, sensible defaults
        are used based on the data dimensionality.
    save_path : Optional[str]
        Path to save the trained model checkpoint. If None, model is not saved.
    plot_save_path : Optional[str]
        Path to save the loss curve plot. If None, plot is not saved.

    Returns
    -------
    model : VAE
        Trained VAE model in eval mode, on CPU.
    history : Dict[str, List[float]]
        Dictionary with keys:
        - 'train_total', 'train_recon', 'train_kl'
        - 'val_total', 'val_recon', 'val_kl'
        Each value is a list of per-epoch losses.

    Raises
    ------
    TypeError
        If data is not a numpy array.
    ValueError
        If data shape is not 3D or contains NaN/Inf values.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(377, 24, 3)  # 377 windows, 24 timesteps, 3 variables
    >>> model, history = train_vae(data, save_path="my_vae.pt")
    """
    if config is None:
        config = VAEConfig()

    # --- Validate inputs before anything else ---
    _validate_inputs(data, config)

    num_windows, window_len, num_variables = data.shape

    # --- Resolve auto-derived config defaults ---
    config = _resolve_config_defaults(config, input_dim=num_variables)

    # --- Seeding ---
    if config.random_seed is not None:
        _set_seed(config.random_seed)

    # --- Device ---
    device = _resolve_device(config.device)

    if config.verbose_every > 0:
        print(f"[VAE] Device: {device}")
        print(f"[VAE] Input dim: {config.input_dim} | Latent dim: {config.latent_dim} | Hidden dim: {config.hidden_dim}")
        print(f"[VAE] Windows: {num_windows} | Window length: {window_len} | Variables: {num_variables}")
        val_size = int(num_windows * config.val_fraction)
        print(f"[VAE] Train windows: {num_windows - val_size} | Val windows: {val_size}")
        print(f"[VAE] KL warmup: {config.kl_warmup_epochs} epochs | Max KL weight: {config.kl_weight_max}")
        print("-" * 60)

    # --- Data preparation ---
    train_loader, val_loader = _prepare_dataloaders(data, config, device)

    # --- Model, optimiser ---
    model = VAE(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- History tracking ---
    history: Dict[str, List[float]] = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total": [],   "val_recon": [],   "val_kl": [],
    }

    # --- Training loop ---
    for epoch in range(1, config.num_epochs + 1):

        # Linear KL annealing: weight rises from 0 to kl_weight_max over warmup period
        if config.kl_warmup_epochs > 0:
            kl_weight = min(
                config.kl_weight_max,
                config.kl_weight_max * (epoch / config.kl_warmup_epochs),
            )
        else:
            kl_weight = config.kl_weight_max

        # Training epoch
        train_total, train_recon, train_kl = _run_epoch(
            model, train_loader, optimizer, kl_weight, device, training=True
        )

        # Validation epoch
        val_total, val_recon, val_kl = _run_epoch(
            model, val_loader, None, kl_weight, device, training=False
        )

        # Record history
        history["train_total"].append(train_total)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_total"].append(val_total)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        # Verbose logging
        if config.verbose_every > 0 and (epoch % config.verbose_every == 0 or epoch == 1):
            print(
                f"[VAE] Epoch {epoch:>4d}/{config.num_epochs} | "
                f"KL weight: {kl_weight:.4f} | "
                f"Train — Total: {train_total:.6f}, Recon: {train_recon:.6f}, KL: {train_kl:.6f} | "
                f"Val   — Total: {val_total:.6f}, Recon: {val_recon:.6f}, KL: {val_kl:.6f}"
            )

    if config.verbose_every > 0:
        print("-" * 60)
        print("[VAE] Training complete.")

    # --- Move model to CPU before saving/returning ---
    model = model.cpu()
    model.eval()

    # --- Save checkpoint ---
    if save_path is not None:
        save_vae(model, save_path)
        if config.verbose_every > 0:
            print(f"[VAE] Model saved to: {save_path}")

    # --- Plot and optionally save loss curves ---
    fig = plot_loss_curves(history)
    if plot_save_path is not None:
        fig.savefig(plot_save_path, bbox_inches="tight", dpi=150)
        if config.verbose_every > 0:
            print(f"[VAE] Loss curves saved to: {plot_save_path}")
    plt.close(fig)

    return model, history


def encode(
    data: np.ndarray,
    model: "VAE",
    use_mean: bool = True,
) -> np.ndarray:
    """
    Encode windowed multivariate data into the VAE latent space.

    Each timestep is encoded independently. The output preserves the window
    and timestep structure, replacing the variable axis with the latent axis.
    This is the primary output consumed by Module 3 (TimeGAN).

    Parameters
    ----------
    data : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Must be normalised with the same scaler used during training.
    model : VAE
        Trained VAE model (as returned by train_vae).
    use_mean : bool
        If True (default), returns the encoder mean (mu) as the latent
        representation — deterministic and recommended for downstream use.
        If False, samples from the posterior — introduces stochasticity.

    Returns
    -------
    latent : np.ndarray
        Shape (num_windows, window_length, latent_dim).

    Examples
    --------
    >>> latent = encode(data, model)
    >>> print(latent.shape)  # (377, 24, 4) for latent_dim=4
    """
    if not isinstance(data, np.ndarray) or data.ndim != 3:
        raise ValueError(
            "data must be a 3D numpy array of shape "
            "(num_windows, window_length, num_variables)."
        )

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        tensor = torch.tensor(data, dtype=torch.float32).to(device)
        original_shape = tensor.shape  # (num_windows, window_len, input_dim)

        # Flatten to (num_windows * window_len, input_dim)
        tensor_flat = tensor.reshape(-1, original_shape[-1])

        mu, log_var = model.encoder(tensor_flat)

        if use_mean:
            latent_flat = mu
        else:
            latent_flat = model.reparameterise(mu, log_var)

        # Reshape back to (num_windows, window_len, latent_dim)
        latent = latent_flat.reshape(original_shape[0], original_shape[1], -1)

    return latent.cpu().numpy()


def decode(
    latent: np.ndarray,
    model: "VAE",
) -> np.ndarray:
    """
    Decode latent sequences back into the original data space.

    Used at generation time: latent sequences produced by Module 3 (TimeGAN)
    are passed through this function to recover synthetic multivariate series.

    Parameters
    ----------
    latent : np.ndarray
        Shape (num_windows, window_length, latent_dim).
    model : VAE
        Trained VAE model (as returned by train_vae).

    Returns
    -------
    reconstructed : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Output is in normalised space — apply inverse scaler from Module 1
        to recover original scale.

    Examples
    --------
    >>> reconstructed = decode(latent, model)
    >>> print(reconstructed.shape)  # (377, 24, 3) for 3 variables
    """
    if not isinstance(latent, np.ndarray) or latent.ndim != 3:
        raise ValueError(
            "latent must be a 3D numpy array of shape "
            "(num_windows, window_length, latent_dim)."
        )

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        tensor = torch.tensor(latent, dtype=torch.float32).to(device)
        original_shape = tensor.shape  # (num_windows, window_len, latent_dim)

        # Flatten to (num_windows * window_len, latent_dim)
        tensor_flat = tensor.reshape(-1, original_shape[-1])

        recon_flat = model.decoder(tensor_flat)

        # Reshape back to (num_windows, window_len, num_variables)
        reconstructed = recon_flat.reshape(original_shape[0], original_shape[1], -1)

    return reconstructed.cpu().numpy()


def save_vae(model: "VAE", path: str) -> None:
    """
    Save the trained VAE model weights and config to disk.

    Parameters
    ----------
    model : VAE
        Trained VAE model.
    path : str
        File path for the checkpoint (e.g., 'vae_checkpoint.pt').
    """
    checkpoint = {
        "config": model.config,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, path)


def load_vae(path: str) -> "VAE":
    """
    Load a saved VAE model from disk.

    Parameters
    ----------
    path : str
        Path to a checkpoint saved by save_vae or train_vae.

    Returns
    -------
    model : VAE
        Loaded VAE model in eval mode on CPU.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.

    Examples
    --------
    >>> model = load_vae("vae_checkpoint.pt")
    >>> latent = encode(new_data, model)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found at '{path}'. "
            "Ensure you have trained and saved a VAE model first."
        )

    # weights_only=False is required because the checkpoint includes the
    # VAEConfig dataclass object. This is safe as long as the checkpoint
    # was produced by save_vae() from this module.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = VAE(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def plot_loss_curves(
    history: Dict[str, List[float]],
) -> plt.Figure:
    """
    Plot training and validation loss curves from a training history dictionary.

    Produces a two-panel figure:
    - Left panel: Total ELBO loss (train vs val)
    - Right panel: Reconstruction and KL components (train only)

    Parameters
    ----------
    history : Dict[str, List[float]]
        As returned by train_vae.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object. Call fig.savefig(...) or plt.show() externally.

    Examples
    --------
    >>> model, history = train_vae(data)
    >>> fig = plot_loss_curves(history)
    >>> fig.savefig("losses.png")
    """
    epochs = list(range(1, len(history["train_total"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Total loss
    axes[0].plot(epochs, history["train_total"], label="Train Total", linewidth=1.5)
    axes[0].plot(epochs, history["val_total"],   label="Val Total",   linewidth=1.5, linestyle="--")
    axes[0].set_title("Total ELBO Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Reconstruction vs KL
    axes[1].plot(epochs, history["train_recon"], label="Train Recon", linewidth=1.5)
    axes[1].plot(epochs, history["train_kl"],    label="Train KL",    linewidth=1.5, linestyle="--")
    axes[1].plot(epochs, history["val_recon"],   label="Val Recon",   linewidth=1.5, linestyle=":")
    axes[1].set_title("Reconstruction vs KL Loss (Train)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("VAE Training Loss Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()

    return fig
