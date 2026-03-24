"""
timegan_module.py
=================
Module 3: TimeGAN for multivariate time series generation.

Implements the TimeGAN architecture (Yoon et al., 2019) with the following
components:
    - Embedder:    Maps real data sequences to latent embedding space
    - Recovery:    Maps latent embeddings back to data space
    - Generator:   Generates synthetic latent sequences from noise
    - Supervisor:  Learns stepwise temporal dynamics in latent space
    - Discriminator: Distinguishes real from synthetic latent sequences

Training proceeds in three sequential phases:
    1. Embedding pretraining  (embedder + recovery)
    2. Supervised pretraining (generator + supervisor, using real embeddings)
    3. Joint GAN training     (all five networks jointly)

Designed for Option B (raw normalised data input) but structured so that
the input can be swapped to VAE latent sequences from Module 2 with minimal
changes when scaling to 5-8 variables.

Usage
-----
from timegan_module import TimeGANConfig, train_timegan, generate, load_timegan

# Train
model, history = train_timegan(
    data=windows,               # shape: (num_windows, window_length, num_variables)
    config=TimeGANConfig(),
    save_path="timegan_checkpoint.pt"
)

# Generate synthetic windows
synthetic = generate(n_samples=200, model=model)
# synthetic shape: (200, window_length, num_variables)

# Load saved model
model = load_timegan("timegan_checkpoint.pt")

Notes
-----
- Expects normalised data from Module 1 (or latent sequences from Module 2).
- Output of generate() is in the same normalised space as the input.
  Apply Module 1's inverse_transform() to recover original scale.
- Early stopping monitors discriminative score during joint training phase.
  Lower discriminative score = harder to distinguish real from synthetic = better.
- Reference: Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series
  generative adversarial networks. NeurIPS.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class TimeGANConfig:
    """
    Configuration for the TimeGAN architecture and training procedure.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for all GRU networks. Default 32.
    num_layers : int
        Number of GRU layers in each network. Default 2.
    noise_dim : int
        Dimension of the random noise input to the generator.
        If 0, defaults to match the input data dimension at runtime.
    embedding_dim : int
        Dimension of the internal embedding space shared across all networks.
        If 0, defaults to hidden_dim at runtime.
    num_epochs : int
        Total epochs across all three training phases combined.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Learning rate for all optimisers (Adam).
    gamma : float
        Weighting factor for the unsupervised GAN loss relative to the
        supervised loss in joint training. Default 1.0.
    phase_split : Tuple[float, float, float]
        Fraction of total epochs allocated to each phase.
        Must sum to 1.0. Default (0.2, 0.2, 0.6).
    early_stopping_patience : int
        Number of discriminative score evaluations without improvement
        before stopping joint training early. Set to 0 to disable.
        Default 5.
    eval_every_n_epochs : int
        Evaluate discriminative score every N epochs during joint training.
        Default 50.
    verbose_every : int
        Print progress every N epochs within each phase. Default 50.
    random_seed : Optional[int]
        Seed for reproducibility.
    device : str
        Compute device. 'auto' selects GPU if available, else CPU.
    """
    hidden_dim: int = 32
    num_layers: int = 2
    noise_dim: int = 0                          # 0 = auto-match input dim
    embedding_dim: int = 0                      # 0 = auto-match hidden_dim
    num_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 1e-3
    gamma: float = 1.0
    phase_split: Tuple[float, float, float] = (0.2, 0.2, 0.6)
    early_stopping_patience: int = 5           # 0 = disabled
    eval_every_n_epochs: int = 50
    verbose_every: int = 50
    random_seed: Optional[int] = None
    device: str = "auto"


# =============================================================================
# Internal Utilities
# =============================================================================

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_config_defaults(
    config: TimeGANConfig, input_dim: int
) -> TimeGANConfig:
    """Fill in auto-derived fields based on actual input dimensionality."""
    noise_dim = config.noise_dim if config.noise_dim > 0 else input_dim
    embedding_dim = config.embedding_dim if config.embedding_dim > 0 else config.hidden_dim

    # Validate phase split
    if abs(sum(config.phase_split) - 1.0) > 1e-6:
        raise ValueError(
            f"phase_split must sum to 1.0, got {sum(config.phase_split):.4f}. "
            f"Current values: {config.phase_split}"
        )

    return TimeGANConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        noise_dim=noise_dim,
        embedding_dim=embedding_dim,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        phase_split=config.phase_split,
        early_stopping_patience=config.early_stopping_patience,
        eval_every_n_epochs=config.eval_every_n_epochs,
        verbose_every=config.verbose_every,
        random_seed=config.random_seed,
        device=config.device,
    )


def _validate_inputs(data: np.ndarray, config: TimeGANConfig) -> None:
    """Validate input data before training."""
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Expected numpy array, got {type(data).__name__}. "
            "Pass windowed data of shape (num_windows, window_length, num_variables)."
        )
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D array (num_windows, window_length, num_variables), "
            f"got shape {data.shape}."
        )
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values.")
    if np.isinf(data).any():
        raise ValueError("Input data contains infinite values.")
    if data.shape[0] < config.batch_size:
        warnings.warn(
            f"Number of windows ({data.shape[0]}) is less than batch_size "
            f"({config.batch_size}). Consider reducing batch_size.",
            UserWarning,
        )


def _compute_phase_epochs(
    num_epochs: int,
    phase_split: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Compute integer epoch counts for each phase from the fractional split.
    Assigns any rounding remainder to the joint training phase.
    """
    embed_epochs = max(1, int(num_epochs * phase_split[0]))
    super_epochs = max(1, int(num_epochs * phase_split[1]))
    joint_epochs = max(1, num_epochs - embed_epochs - super_epochs)
    return embed_epochs, super_epochs, joint_epochs


def _make_dataloader(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    tensor = torch.tensor(data, dtype=torch.float32)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,     # drop_last=True keeps batch sizes consistent
    )                       # which stabilises GAN training


# =============================================================================
# TimeGAN Network Components
# =============================================================================

class _GRUNet(nn.Module):
    """
    Base GRU network used by all five TimeGAN components.
    Takes a sequence input and produces a sequence output of specified dim.

    Architecture: GRU -> Linear projection -> optional activation

    Parameters
    ----------
    input_dim : int
    hidden_dim : int
    output_dim : int
    num_layers : int
    activation : Optional[nn.Module]
        Applied to the output projection. Use nn.Sigmoid() for outputs
        that should be bounded, None for unbounded outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim).

        Returns
        -------
        out : torch.Tensor
            Shape (batch, seq_len, output_dim).
        """
        out, _ = self.gru(x)
        out = self.projection(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class TimeGAN(nn.Module):
    """
    TimeGAN model comprising all five sub-networks.

    All networks operate in sequence space (batch, seq_len, dim).
    The embedder and recovery form an autoencoder pair in latent space.
    The generator and supervisor produce synthetic latent sequences.
    The discriminator operates on latent sequences to distinguish real
    from synthetic.

    Parameters
    ----------
    config : TimeGANConfig
        Fully resolved configuration (all fields non-zero).
    input_dim : int
        Number of variables in the input data (or latent dim from Module 2).
    """

    def __init__(self, config: TimeGANConfig, input_dim: int):
        super().__init__()

        self.config = config
        self.input_dim = input_dim

        emb_dim = config.embedding_dim
        hid_dim = config.hidden_dim
        n_layers = config.num_layers

        # Embedder: data space -> embedding space
        # Sigmoid activation keeps embeddings in (0,1) — consistent with
        # the normalised input range from Module 1
        self.embedder = _GRUNet(
            input_dim=input_dim,
            hidden_dim=hid_dim,
            output_dim=emb_dim,
            num_layers=n_layers,
            activation=nn.Sigmoid(),
        )

        # Recovery: embedding space -> data space
        # Sigmoid activation to match normalised input range
        self.recovery = _GRUNet(
            input_dim=emb_dim,
            hidden_dim=hid_dim,
            output_dim=input_dim,
            num_layers=n_layers,
            activation=nn.Sigmoid(),
        )

        # Generator: noise space -> embedding space
        # Sigmoid activation to match embedding space range
        self.generator = _GRUNet(
            input_dim=config.noise_dim,
            hidden_dim=hid_dim,
            output_dim=emb_dim,
            num_layers=n_layers,
            activation=nn.Sigmoid(),
        )

        # Supervisor: embedding space -> embedding space (stepwise transitions)
        # Sigmoid activation — operates in same space as embedder output
        self.supervisor = _GRUNet(
            input_dim=emb_dim,
            hidden_dim=hid_dim,
            output_dim=emb_dim,
            num_layers=n_layers,
            activation=nn.Sigmoid(),
        )

        # Discriminator: embedding space -> binary sequence (real vs synthetic)
        # No activation here — BCEWithLogitsLoss handles sigmoid internally
        # for numerical stability
        self.discriminator = _GRUNet(
            input_dim=emb_dim,
            hidden_dim=hid_dim,
            output_dim=1,
            num_layers=n_layers,
            activation=None,
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed real data into latent space. Shape: (..., input_dim) -> (..., emb_dim)."""
        return self.embedder(x)

    def recover(self, h: torch.Tensor) -> torch.Tensor:
        """Recover data from latent embedding. Shape: (..., emb_dim) -> (..., input_dim)."""
        return self.recovery(h)

    def generate_noise(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Sample uniform noise for the generator input."""
        return torch.rand(batch_size, seq_len, self.config.noise_dim, device=device)

    def forward_generator(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthetic embedding from noise via generator + supervisor."""
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        return h_hat

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for diagnostic purposes.

        Returns
        -------
        x_tilde : reconstructed real data
        h       : real embeddings
        h_hat   : synthetic embeddings
        y_fake  : discriminator logits on synthetic embeddings
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = self.embedder(x)
        x_tilde = self.recovery(h)

        z = self.generate_noise(batch_size, seq_len, device)
        h_hat = self.forward_generator(z)
        y_fake = self.discriminator(h_hat)

        return x_tilde, h, h_hat, y_fake


# =============================================================================
# Loss Functions
# =============================================================================

def _reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss for embedding pretraining."""
    return nn.functional.mse_loss(x_tilde, x)


def _supervised_loss(
    h: torch.Tensor, h_hat_supervise: torch.Tensor
) -> torch.Tensor:
    """
    Supervised loss: MSE between real embeddings at t+1 and supervisor
    predictions from real embeddings at t. Teaches temporal transitions.
    Uses h[:, 1:, :] as targets and supervisor output h[:, :-1, :] as preds.
    """
    return nn.functional.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])


def _generator_loss(
    y_fake: torch.Tensor,
    y_fake_e: torch.Tensor,
    h: torch.Tensor,
    h_hat: torch.Tensor,
    x: torch.Tensor,
    x_hat: torch.Tensor,
    supervised_loss: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Combined generator loss:
    - Unsupervised loss: fool discriminator on synthetic embeddings
    - Unsupervised loss on raw generator output (before supervisor)
    - Supervised loss: match real temporal transitions
    - Moment matching: match mean and variance of real vs synthetic embeddings
    """
    # Adversarial losses — generator wants discriminator to output 1 (real)
    ones = torch.ones_like(y_fake)
    g_loss_u = nn.functional.binary_cross_entropy_with_logits(y_fake, ones)
    g_loss_u_e = nn.functional.binary_cross_entropy_with_logits(y_fake_e, ones)

    # Moment matching loss — match first and second moments
    # between real and synthetic embeddings
    g_loss_v1 = torch.mean(
        torch.abs(torch.sqrt(h_hat.var(dim=0) + 1e-6) - torch.sqrt(h.var(dim=0) + 1e-6))
    )
    g_loss_v2 = torch.mean(torch.abs(h_hat.mean(dim=0) - h.mean(dim=0)))
    g_loss_v = g_loss_v1 + g_loss_v2

    # Combined
    return g_loss_u + gamma * g_loss_u_e + 100 * supervised_loss + 100 * g_loss_v


def _discriminator_loss(
    y_real: torch.Tensor,
    y_fake: torch.Tensor,
    y_fake_e: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator loss:
    - Real embeddings should be classified as real (1)
    - Synthetic embeddings (from supervisor) should be classified as fake (0)
    - Raw generator embeddings (before supervisor) should be fake (0)
    Only update if discriminator is not already too strong (loss > 0.15).
    """
    ones = torch.ones_like(y_real)
    zeros_fake = torch.zeros_like(y_fake)
    zeros_fake_e = torch.zeros_like(y_fake_e)

    d_loss_real = nn.functional.binary_cross_entropy_with_logits(y_real, ones)
    d_loss_fake = nn.functional.binary_cross_entropy_with_logits(y_fake, zeros_fake)
    d_loss_fake_e = nn.functional.binary_cross_entropy_with_logits(y_fake_e, zeros_fake_e)

    return d_loss_real + d_loss_fake + d_loss_fake_e


# =============================================================================
# Discriminative Score (Lightweight Evaluation)
# =============================================================================

def _compute_discriminative_score(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
) -> float:
    """
    Compute a lightweight discriminative score using logistic regression.

    Train a classifier to distinguish real from synthetic sequences.
    Score of 0.5 = indistinguishable (best). Score of 1.0 = perfectly
    distinguishable (worst).

    Uses mean-pooled sequence features for speed — this is a proxy metric
    for early stopping, not a rigorous evaluation metric (that belongs in
    Module 5).

    Parameters
    ----------
    real_data : np.ndarray
        Shape (n, seq_len, n_vars). Real normalised windows.
    synthetic_data : np.ndarray
        Shape (n, seq_len, n_vars). Generated normalised windows.

    Returns
    -------
    score : float
        Accuracy of the classifier. Closer to 0.5 is better.
    """
    n = min(len(real_data), len(synthetic_data))

    # Mean-pool across time dimension: (n, seq_len, n_vars) -> (n, n_vars)
    real_feats = real_data[:n].mean(axis=1)
    fake_feats = synthetic_data[:n].mean(axis=1)

    X = np.concatenate([real_feats, fake_feats], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Split
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Logistic regression with standard scaling
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, random_state=0)),
    ])

    try:
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
    except Exception:
        # If fitting fails (e.g. degenerate data early in training), return worst score
        score = 1.0

    return float(score)


# =============================================================================
# Training Phases
# =============================================================================

def _train_embedding_phase(
    model: TimeGAN,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    verbose_every: int,
    phase_label: str = "Embedding",
) -> List[float]:
    """
    Phase 1: Train embedder and recovery networks jointly.
    Objective: minimise reconstruction loss through the autoencoder.
    Generator, supervisor, and discriminator are not updated.
    """
    model.train()
    losses = []

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []

        for (batch,) in loader:
            batch = batch.to(device)

            # Forward: embed then recover
            h = model.embedder(batch)
            x_tilde = model.recovery(h)

            # Reconstruction loss + supervised loss on real embeddings
            # Including supervised loss here gives the supervisor
            # a head start before joint training
            h_supervise = model.supervisor(h)
            loss_recon = _reconstruction_loss(batch, x_tilde)
            loss_sup = _supervised_loss(h, h_supervise)
            loss = 10 * loss_recon + 0.1 * loss_sup

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)

        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
            print(
                f"  [{phase_label}] Epoch {epoch:>4d}/{num_epochs} | "
                f"Loss: {avg_loss:.6f}"
            )

    return losses


def _train_supervised_phase(
    model: TimeGAN,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    verbose_every: int,
    phase_label: str = "Supervised",
) -> List[float]:
    """
    Phase 2: Train generator and supervisor using real embeddings as targets.
    Objective: teach the generator to produce embeddings that match the
    temporal transition structure of real data.
    Embedder, recovery, and discriminator are not updated.
    """
    model.train()
    losses = []

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []

        for (batch,) in loader:
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape

            with torch.no_grad():
                h = model.embedder(batch)

            # Generator produces embeddings from noise
            z = model.generate_noise(batch_size, seq_len, device)
            e_hat = model.generator(z)

            # Supervisor predicts next-step transitions
            h_hat_supervise = model.supervisor(e_hat)

            loss = _supervised_loss(h, h_hat_supervise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)

        if verbose_every > 0 and (epoch % verbose_every == 0 or epoch == 1):
            print(
                f"  [{phase_label}] Epoch {epoch:>4d}/{num_epochs} | "
                f"Loss: {avg_loss:.6f}"
            )

    return losses


def _train_joint_phase(
    model: TimeGAN,
    loader: DataLoader,
    optimizers: Dict[str, optim.Optimizer],
    num_epochs: int,
    device: torch.device,
    config: TimeGANConfig,
    real_data: np.ndarray,
) -> Tuple[Dict[str, List[float]], int]:
    """
    Phase 3: Joint adversarial training of all five networks.

    Training alternates between:
    - Generator + Supervisor update (minimise generator loss)
    - Embedder + Recovery update (minimise reconstruction + supervised)
    - Discriminator update (maximise classification accuracy)

    Discriminator is only updated when its loss is above a threshold (0.15)
    to prevent it from overpowering the generator early in training.

    Early stopping monitors discriminative score every eval_every_n_epochs.

    Returns
    -------
    losses : Dict[str, List[float]]
    best_epoch : int
        Epoch at which the best discriminative score was achieved.
    """
    losses = {
        "g_loss": [], "d_loss": [], "e_loss": [], "disc_score": []
    }

    best_disc_score = 1.0        # Start at worst possible score
    best_state_dict = None
    patience_counter = 0
    best_epoch = 0
    eval_counter = 0

    opt_g = optimizers["generator"]
    opt_e = optimizers["embedder"]
    opt_d = optimizers["discriminator"]

    for epoch in range(1, num_epochs + 1):
        g_losses, d_losses, e_losses = [], [], []

        for (batch,) in loader:
            batch = batch.to(device)
            batch_size, seq_len, _ = batch.shape

            # ---- Generator + Supervisor step ----
            z = model.generate_noise(batch_size, seq_len, device)
            e_hat = model.generator(z)
            h_hat = model.supervisor(e_hat)

            # Also get discriminator outputs for generator loss
            y_fake = model.discriminator(h_hat)
            y_fake_e = model.discriminator(e_hat)

            with torch.no_grad():
                h = model.embedder(batch)

            h_hat_supervise = model.supervisor(e_hat.detach())
            sup_loss = _supervised_loss(h, h_hat_supervise)

            x_hat = model.recovery(h_hat)

            g_loss = _generator_loss(
                y_fake, y_fake_e, h, h_hat, batch, x_hat,
                sup_loss, config.gamma
            )

            opt_g.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()

            # ---- Embedder + Recovery step ----
            h = model.embedder(batch)
            x_tilde = model.recovery(h)
            h_supervise = model.supervisor(h)

            loss_recon = _reconstruction_loss(batch, x_tilde)
            loss_sup = _supervised_loss(h, h_supervise)
            e_loss = 10 * loss_recon + 0.1 * loss_sup

            opt_e.zero_grad()
            e_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_e.step()

            # ---- Discriminator step ----
            # Only update if discriminator is not already dominating
            with torch.no_grad():
                h_real = model.embedder(batch)
                z_d = model.generate_noise(batch_size, seq_len, device)
                e_hat_d = model.generator(z_d)
                h_hat_d = model.supervisor(e_hat_d)

            y_real = model.discriminator(h_real)
            y_fake_d = model.discriminator(h_hat_d.detach())
            y_fake_e_d = model.discriminator(e_hat_d.detach())

            d_loss = _discriminator_loss(y_real, y_fake_d, y_fake_e_d)

            # Conditional discriminator update — prevents it from overpowering
            # generator during early joint training
            if d_loss.item() > 0.15:
                opt_d.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt_d.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            e_losses.append(e_loss.item())

        avg_g = float(np.mean(g_losses))
        avg_d = float(np.mean(d_losses))
        avg_e = float(np.mean(e_losses))

        losses["g_loss"].append(avg_g)
        losses["d_loss"].append(avg_d)
        losses["e_loss"].append(avg_e)

        # ---- Verbose logging ----
        if config.verbose_every > 0 and (
            epoch % config.verbose_every == 0 or epoch == 1
        ):
            print(
                f"  [Joint] Epoch {epoch:>4d}/{num_epochs} | "
                f"G: {avg_g:.4f} | D: {avg_d:.4f} | E: {avg_e:.4f}"
            )

        # ---- Early stopping check ----
        if (
            config.early_stopping_patience > 0
            and epoch % config.eval_every_n_epochs == 0
        ):
            eval_counter += 1
            model.eval()
            synthetic = _generate_internal(
                model=model,
                n_samples=min(len(real_data), 200),
                seq_len=real_data.shape[1],
                device=device,
            )
            model.train()

            disc_score = _compute_discriminative_score(real_data, synthetic)
            losses["disc_score"].append(disc_score)

            if config.verbose_every > 0:
                print(
                    f"  [Joint] Eval {eval_counter} @ epoch {epoch} | "
                    f"Discriminative score: {disc_score:.4f} "
                    f"(0.5=best, 1.0=worst)"
                )

            # Save best checkpoint
            if disc_score < best_disc_score:
                best_disc_score = disc_score
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    if config.verbose_every > 0:
                        print(
                            f"  [Joint] Early stopping triggered at epoch {epoch}. "
                            f"Best discriminative score: {best_disc_score:.4f} "
                            f"at epoch {best_epoch}."
                        )
                    break

    # Restore best weights if early stopping found a better checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if config.verbose_every > 0 and best_epoch < num_epochs:
            print(
                f"  [Joint] Restored best weights from epoch {best_epoch}."
            )

    return losses, best_epoch


# =============================================================================
# Internal Generation Helper
# =============================================================================

def _generate_internal(
    model: TimeGAN,
    n_samples: int,
    seq_len: int,
    device: torch.device,
) -> np.ndarray:
    """
    Generate synthetic sequences from a trained TimeGAN.
    Internal helper used by both generate() and early stopping evaluation.

    Returns numpy array of shape (n_samples, seq_len, input_dim).
    """
    model.eval()
    results = []
    batch_size = 64

    with torch.no_grad():
        generated = 0
        while generated < n_samples:
            current_batch = min(batch_size, n_samples - generated)
            z = model.generate_noise(current_batch, seq_len, device)
            h_hat = model.forward_generator(z)
            x_hat = model.recovery(h_hat)
            results.append(x_hat.cpu().numpy())
            generated += current_batch

    return np.concatenate(results, axis=0)[:n_samples]


# =============================================================================
# Public API
# =============================================================================

def train_timegan(
    data: np.ndarray,
    config: Optional[TimeGANConfig] = None,
    save_path: Optional[str] = "timegan_checkpoint.pt",
    plot_save_path: Optional[str] = "timegan_loss_curves.png",
) -> Tuple["TimeGAN", Dict[str, List[float]]]:
    """
    Train TimeGAN on windowed multivariate time series data.

    Runs three sequential training phases:
    1. Embedding pretraining
    2. Supervised pretraining
    3. Joint GAN training with optional early stopping

    Parameters
    ----------
    data : np.ndarray
        Shape (num_windows, window_length, num_variables).
        Normalised data from Module 1, or latent sequences from Module 2.
    config : Optional[TimeGANConfig]
        Training configuration. If None, sensible defaults are used.
    save_path : Optional[str]
        Path to save the trained model checkpoint.
    plot_save_path : Optional[str]
        Path to save the loss curve figure.

    Returns
    -------
    model : TimeGAN
        Trained model in eval mode on CPU.
    history : Dict[str, List[float]]
        Per-epoch losses for all three phases:
        - 'embed_loss'  : Phase 1 losses
        - 'super_loss'  : Phase 2 losses
        - 'g_loss'      : Phase 3 generator losses
        - 'd_loss'      : Phase 3 discriminator losses
        - 'e_loss'      : Phase 3 embedder losses
        - 'disc_score'  : Discriminative scores computed during Phase 3

    Examples
    --------
    >>> from timegan_module import TimeGANConfig, train_timegan, generate
    >>> model, history = train_timegan(windows, save_path="timegan.pt")
    >>> synthetic = generate(200, model)
    >>> print(synthetic.shape)  # (200, 24, num_variables)
    """
    if config is None:
        config = TimeGANConfig()

    _validate_inputs(data, config)

    num_windows, seq_len, input_dim = data.shape
    config = _resolve_config_defaults(config, input_dim)

    if config.random_seed is not None:
        _set_seed(config.random_seed)

    device = _resolve_device(config.device)

    embed_epochs, super_epochs, joint_epochs = _compute_phase_epochs(
        config.num_epochs, config.phase_split
    )

    if config.verbose_every > 0:
        print(f"[TimeGAN] Device: {device}")
        print(f"[TimeGAN] Input dim: {input_dim} | Embedding dim: {config.embedding_dim} | "
              f"Hidden dim: {config.hidden_dim} | Noise dim: {config.noise_dim}")
        print(f"[TimeGAN] Windows: {num_windows} | Seq length: {seq_len}")
        print(f"[TimeGAN] Total epochs: {config.num_epochs} | "
              f"Phase split — Embed: {embed_epochs} | "
              f"Supervised: {super_epochs} | Joint: {joint_epochs}")
        print(f"[TimeGAN] Early stopping: "
              f"{'enabled (patience=' + str(config.early_stopping_patience) + ')' if config.early_stopping_patience > 0 else 'disabled'}")
        print("=" * 70)

    # --- Build model ---
    model = TimeGAN(config, input_dim).to(device)
    model._seq_len = seq_len  # Store for use in generate()

    # --- DataLoader ---
    loader = _make_dataloader(data, config.batch_size, shuffle=True)

    # --- Optimisers ---
    # Separate optimisers per network group for independent updates
    opt_embed = optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=config.learning_rate,
    )
    opt_super = optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()),
        lr=config.learning_rate,
    )
    opt_joint_g = optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()),
        lr=config.learning_rate,
    )
    opt_joint_e = optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()),
        lr=config.learning_rate,
    )
    opt_joint_d = optim.Adam(
        model.discriminator.parameters(),
        lr=config.learning_rate,
    )

    history: Dict[str, List[float]] = {}

    # =========================================================================
    # Phase 1: Embedding pretraining
    # =========================================================================
    if config.verbose_every > 0:
        print(f"\n[Phase 1/3] Embedding Pretraining ({embed_epochs} epochs)")
        print("-" * 70)

    embed_losses = _train_embedding_phase(
        model=model,
        loader=loader,
        optimizer=opt_embed,
        num_epochs=embed_epochs,
        device=device,
        verbose_every=config.verbose_every,
    )
    history["embed_loss"] = embed_losses

    # =========================================================================
    # Phase 2: Supervised pretraining
    # =========================================================================
    if config.verbose_every > 0:
        print(f"\n[Phase 2/3] Supervised Pretraining ({super_epochs} epochs)")
        print("-" * 70)

    super_losses = _train_supervised_phase(
        model=model,
        loader=loader,
        optimizer=opt_super,
        num_epochs=super_epochs,
        device=device,
        verbose_every=config.verbose_every,
    )
    history["super_loss"] = super_losses

    # =========================================================================
    # Phase 3: Joint GAN training
    # =========================================================================
    if config.verbose_every > 0:
        print(f"\n[Phase 3/3] Joint GAN Training ({joint_epochs} epochs)")
        print("-" * 70)

    joint_losses, best_epoch = _train_joint_phase(
        model=model,
        loader=loader,
        optimizers={
            "generator": opt_joint_g,
            "embedder": opt_joint_e,
            "discriminator": opt_joint_d,
        },
        num_epochs=joint_epochs,
        device=device,
        config=config,
        real_data=data,
    )
    history.update(joint_losses)

    if config.verbose_every > 0:
        print("=" * 70)
        print(f"[TimeGAN] Training complete. Best checkpoint at joint epoch {best_epoch}.")

    # --- Move to CPU, eval mode ---
    model = model.cpu()
    model.eval()

    # --- Save checkpoint ---
    if save_path is not None:
        save_timegan(model, save_path)
        if config.verbose_every > 0:
            print(f"[TimeGAN] Model saved to: {save_path}")

    # --- Plot loss curves ---
    fig = plot_loss_curves(history)
    if plot_save_path is not None:
        fig.savefig(plot_save_path, bbox_inches="tight", dpi=150)
        if config.verbose_every > 0:
            print(f"[TimeGAN] Loss curves saved to: {plot_save_path}")
    plt.close(fig)

    return model, history


def generate(
    n_samples: int,
    model: "TimeGAN",
    seq_len: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic time series windows from a trained TimeGAN.

    Output is in the same normalised space as the training data.
    Apply Module 1's inverse_transform() to recover original scale,
    or Module 2's decode() if using the VAE hybrid pipeline.

    Parameters
    ----------
    n_samples : int
        Number of synthetic windows to generate.
    model : TimeGAN
        Trained TimeGAN model (as returned by train_timegan).
    seq_len : Optional[int]
        Length of each generated sequence. If None, must be inferred
        from a prior call context. Typically you should not need to set
        this — it is stored in the model config at training time.
        Provide this explicitly if generating sequences of a different
        length than training (experimental).

    Returns
    -------
    synthetic : np.ndarray
        Shape (n_samples, seq_len, num_variables).
        Normalised synthetic windows ready for inverse_transform or decode.

    Examples
    --------
    >>> synthetic = generate(200, model)
    >>> print(synthetic.shape)  # (200, 24, 3)
    >>>
    >>> # Recover original scale (Module 1)
    >>> from data_pipeline import inverse_transform, load_scaler
    >>> scaler = load_scaler("scaler.pkl")
    >>> original_scale = inverse_transform(synthetic, scaler)
    >>>
    >>> # Or decode through VAE (Module 2 hybrid pipeline)
    >>> from vae_module import decode, load_vae
    >>> vae = load_vae("vae_checkpoint.pt")
    >>> decoded = decode(synthetic, vae)
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")

    device = next(model.parameters()).device

    # Infer seq_len from a dummy forward pass if not provided
    # We store it on the model object after training
    if seq_len is None:
        if not hasattr(model, "_seq_len"):
            raise ValueError(
                "seq_len could not be inferred. Either provide seq_len explicitly "
                "or ensure the model was trained via train_timegan() which sets "
                "model._seq_len automatically."
            )
        seq_len = model._seq_len

    return _generate_internal(model, n_samples, seq_len, device)


def save_timegan(model: "TimeGAN", path: str) -> None:
    """
    Save trained TimeGAN model weights and config to disk.

    Parameters
    ----------
    model : TimeGAN
    path : str
        File path (e.g., 'timegan_checkpoint.pt').
    """
    checkpoint = {
        "config": model.config,
        "input_dim": model.input_dim,
        "state_dict": model.state_dict(),
        "seq_len": getattr(model, "_seq_len", None),
    }
    torch.save(checkpoint, path)


def load_timegan(path: str) -> "TimeGAN":
    """
    Load a saved TimeGAN model from disk.

    Parameters
    ----------
    path : str
        Path to a checkpoint saved by save_timegan or train_timegan.

    Returns
    -------
    model : TimeGAN
        Loaded model in eval mode on CPU.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists at the given path.

    Examples
    --------
    >>> model = load_timegan("timegan_checkpoint.pt")
    >>> synthetic = generate(100, model)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No checkpoint found at '{path}'. "
            "Ensure you have trained and saved a TimeGAN model first."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    input_dim = checkpoint["input_dim"]

    model = TimeGAN(config, input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if checkpoint.get("seq_len") is not None:
        model._seq_len = checkpoint["seq_len"]

    return model


def plot_loss_curves(
    history: Dict[str, List[float]],
) -> plt.Figure:
    """
    Plot training loss curves across all three TimeGAN training phases.

    Produces a figure with up to three panels:
    - Panel 1: Phase 1 embedding loss and Phase 2 supervised loss
    - Panel 2: Phase 3 generator, discriminator, and embedder losses
    - Panel 3: Discriminative scores during Phase 3 (if available)

    Parameters
    ----------
    history : Dict[str, List[float]]
        As returned by train_timegan.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    has_disc_scores = len(history.get("disc_score", [])) > 0
    n_panels = 3 if has_disc_scores else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Phase 1 and Phase 2 losses
    ax = axes[0]
    if "embed_loss" in history:
        ax.plot(history["embed_loss"], label="Phase 1: Embed", linewidth=1.5)
    if "super_loss" in history:
        ax.plot(history["super_loss"], label="Phase 2: Supervised", linewidth=1.5)
    ax.set_title("Pre-training Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Phase 3 GAN losses
    ax = axes[1]
    if "g_loss" in history:
        ax.plot(history["g_loss"], label="Generator", linewidth=1.5)
    if "d_loss" in history:
        ax.plot(history["d_loss"], label="Discriminator", linewidth=1.5)
    if "e_loss" in history:
        ax.plot(history["e_loss"], label="Embedder", linewidth=1.5, linestyle="--")
    ax.set_title("Phase 3: Joint GAN Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Discriminative scores (if available)
    if has_disc_scores:
        ax = axes[2]
        scores = history["disc_score"]
        eval_epochs = [
            (i + 1) * (len(history.get("g_loss", [1])) // max(len(scores), 1))
            for i in range(len(scores))
        ]
        ax.plot(eval_epochs, scores, marker="o", linewidth=1.5,
                label="Discriminative Score")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7,
                   label="Ideal (0.5)")
        ax.set_title("Discriminative Score (lower = better)")
        ax.set_xlabel("Joint Training Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim([0.4, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("TimeGAN Training Loss Curves", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
