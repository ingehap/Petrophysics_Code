"""
Permeability Inference From 3-D Greyscale Images Using Deep Learning
====================================================================

Implements the deep-learning architecture and evaluation metrics of:

    Youssef, S., Feraille, M., Batot, G., Lecomte, J.-F., Cokelaer, F.,
    and Desroziers, S., 2025,
    "Permeability Inference From 3D Grayscale Images Using Deep Learning:
    How a Large-Scale Data Set Can Contribute to Model Generalization",
    Petrophysics, 66(6), 939–955.
    DOI: 10.30632/PJV66N6-2025a2

Key ideas
---------
* Shallow Plain 3-D CNN (SP3D) and deep Residual 3-D CNN (R3D_18)
  architectures for permeability regression from 3-D μCT voxel cubes.
* Greyscale (GLV) vs. binary (Bin) image input comparison.
* Evaluation with R², MAPE, and coefficient of variation (Cv).
* Cartesian-grid upscaling of sub-volume predictions to the mini-plug scale.

Notes
-----
This module provides *architecture definitions* in pure PyTorch together
with helper functions for training, evaluation, and upscaling.  The actual
training data (synchrotron 3-D μCT images paired with lattice-Boltzmann
permeabilities) must be supplied by the user.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ──────────────────────────────────────────────────────────────────────
# 1. Evaluation Metrics
# ──────────────────────────────────────────────────────────────────────
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def coefficient_of_variation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of variation of RMSE (Cv)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse / y_true.mean() if y_true.mean() != 0 else float("inf")


# ──────────────────────────────────────────────────────────────────────
# 2. Upscaling: Cartesian-grid permeability averaging
# ──────────────────────────────────────────────────────────────────────
def harmonic_mean(values: np.ndarray) -> float:
    """Harmonic mean (used for series arrangement of blocks)."""
    values = np.asarray(values, float)
    if np.any(values <= 0):
        raise ValueError("All values must be positive for harmonic mean.")
    return float(len(values) / np.sum(1.0 / values))


def arithmetic_mean(values: np.ndarray) -> float:
    """Arithmetic mean (used for parallel arrangement of blocks)."""
    return float(np.mean(values))


def upscale_permeability_cartesian(
    k_blocks: np.ndarray,
    direction: str = "x",
) -> float:
    """Upscale sub-volume permeabilities on a regular Cartesian grid.

    For flow in a given direction, the effective permeability is obtained
    by taking the *harmonic mean* along that direction and then the
    *arithmetic mean* over the transverse plane.

    Parameters
    ----------
    k_blocks : np.ndarray, shape (nx, ny, nz)
        Permeability of each sub-block (mD).
    direction : {'x', 'y', 'z'}
        Flow direction.

    Returns
    -------
    float
        Effective (upscaled) permeability in the given direction.
    """
    k = np.asarray(k_blocks, float)
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map[direction]

    # Harmonic mean along flow direction for each column
    n_along = k.shape[ax]
    inv_sum = np.sum(1.0 / k, axis=ax)
    k_harmonic = n_along / inv_sum  # shape = transverse plane

    # Arithmetic mean over transverse plane
    return float(np.mean(k_harmonic))


# ──────────────────────────────────────────────────────────────────────
# 3. Neural-Network Architectures (require PyTorch)
# ──────────────────────────────────────────────────────────────────────
if _HAS_TORCH:

    class SP3D(nn.Module):
        """Shallow Plain 3-D CNN for permeability regression.

        Architecture (Table 2 in the paper, simplified):
            Conv3d → PReLU → Conv3d → PReLU → AdaptiveAvgPool → FC → 1
        """

        def __init__(self, in_channels: int = 1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.PReLU(),
                nn.AdaptiveAvgPool3d(1),
            )
            self.regressor = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.  x: (B, C, D, H, W) → (B, 1)."""
            h = self.features(x).view(x.size(0), -1)
            return self.regressor(h)

    class _ResBlock3D(nn.Module):
        """Basic residual block for 3-D convolutions (He et al., 2016)."""

        def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
            super().__init__()
            self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.conv2 = nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                    nn.BatchNorm3d(out_ch),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.prelu(self.bn1(self.conv1(x)),
                          torch.tensor(0.25, device=x.device))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.prelu(out, torch.tensor(0.25, device=x.device))

    class R3D18(nn.Module):
        """Deep 3-D Residual Network (R3D_18) for permeability regression.

        Follows the ResNet-18 architecture adapted to 3-D inputs
        (Tran et al., 2018; He et al., 2016), as used in Youssef et al.

        The model named R3D_18_GLV in the paper achieved R² up to 0.99
        on seen rock types and 0.82 on unseen rock types.
        """

        def __init__(self, in_channels: int = 1):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm3d(64)
            self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

            self.layer1 = self._make_layer(64, 64, 2, stride=1)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)

            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(512, 1)

            # Kaiming initialisation (He et al., 2015)
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="leaky_relu")
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        @staticmethod
        def _make_layer(in_ch: int, out_ch: int, n_blocks: int,
                        stride: int) -> nn.Sequential:
            layers = [_ResBlock3D(in_ch, out_ch, stride)]
            for _ in range(1, n_blocks):
                layers.append(_ResBlock3D(out_ch, out_ch))
            return nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.  x: (B, 1, D, H, W) → (B, 1)."""
            x = self.pool(F.prelu(self.bn1(self.conv1(x)),
                                  torch.tensor(0.25, device=x.device)))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x).view(x.size(0), -1)
            return self.fc(x)

    # ── Training helper ──────────────────────────────────────────────
    def train_one_epoch(
        model: nn.Module,
        loader,
        optimizer,
        device: str = "cpu",
    ) -> float:
        """Train for one epoch and return mean MSE loss.

        Parameters
        ----------
        model : nn.Module
            SP3D or R3D18.
        loader : DataLoader
            Yields (images, log10_permeability) batches.
        optimizer : torch.optim.Optimizer
        device : str
        """
        model.train()
        total_loss = 0.0
        n = 0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device).view(-1, 1)
            preds = model(imgs)
            loss = F.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
        return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────────────
# Quick demo (metrics + upscaling only — no GPU needed)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = 10 ** rng.uniform(0, 3, 50)
    noise = rng.normal(0, 0.1, 50)
    y_pred = 10 ** (np.log10(y_true) + noise)

    print(f"R²  = {r2_score(y_true, y_pred):.4f}")
    print(f"MAPE = {mape(y_true, y_pred):.1f} %")
    print(f"Cv  = {coefficient_of_variation(y_true, y_pred):.4f}")

    # Upscaling demo
    k_grid = rng.uniform(100, 1000, (4, 4, 4))
    for d in ("x", "y", "z"):
        print(f"k_eff ({d}) = {upscale_permeability_cartesian(k_grid, d):.1f} mD")
