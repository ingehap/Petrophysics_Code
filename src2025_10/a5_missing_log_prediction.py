#!/usr/bin/env python3
"""
Article 5: Missing Well-Log Data Prediction Using a Hybrid U-Net and
           LSTM Network Model
Authors: Benard Sasu Oppong, Po Chen, En-Jui Lee, Wu-Yu Liao
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 785-806.
     DOI: 10.30632/PJV66N5-2025a5

Implements (pure-NumPy):
  - Simplified 1-D U-Net encoder–decoder with skip connections
  - LSTM module for sequential depth-trend modelling
  - Hybrid U-Net + LSTM for missing-log prediction
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _relu(x):
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# 1-D Convolution block
# ---------------------------------------------------------------------------

class Conv1DBlock:
    """Single 1-D convolution + ReLU, stride-1, same-padding."""

    def __init__(self, in_ch, out_ch, ks=3, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((out_ch, in_ch, ks)) * 0.1
        self.b = np.zeros(out_ch)
        self.ks = ks

    def forward(self, x):
        """x: (L, C_in) -> (L, C_out)."""
        L, C = x.shape
        pad = self.ks // 2
        xp = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
        out_ch = self.W.shape[0]
        out = np.zeros((L, out_ch))
        for oc in range(out_ch):
            for p in range(L):
                out[p, oc] = np.sum(self.W[oc] * xp[p:p + self.ks].T) + self.b[oc]
        return _relu(out)


# ---------------------------------------------------------------------------
# Simplified 1-D U-Net
# ---------------------------------------------------------------------------

class MiniUNet1D:
    """Encoder-decoder with one skip connection level."""

    def __init__(self, in_ch, mid_ch=8, out_ch=1, seed=10):
        self.enc1 = Conv1DBlock(in_ch, mid_ch, 3, seed)
        self.enc2 = Conv1DBlock(mid_ch, mid_ch * 2, 3, seed + 1)
        self.dec1 = Conv1DBlock(mid_ch * 2 + mid_ch, mid_ch, 3, seed + 2)
        self.dec2 = Conv1DBlock(mid_ch, out_ch, 3, seed + 3)

    @staticmethod
    def _downsample(x):
        return x[::2]

    @staticmethod
    def _upsample(x, target_len):
        return np.repeat(x, 2, axis=0)[:target_len]

    def forward(self, x):
        """x: (L, C_in) -> (L, out_ch)."""
        e1 = self.enc1.forward(x)
        e2 = self.enc2.forward(self._downsample(e1))
        d1 = self._upsample(e2, e1.shape[0])
        d1 = np.concatenate([d1, e1], axis=1)  # skip connection
        d2 = self.dec1.forward(d1)
        return self.dec2.forward(d2)


# ---------------------------------------------------------------------------
# Compact LSTM for depth-wise trends
# ---------------------------------------------------------------------------

class CompactLSTM:
    """Forward-only LSTM returning full sequence of hidden states."""

    def __init__(self, in_dim, hid, seed=20):
        rng = np.random.default_rng(seed)
        s = 0.1
        tot = in_dim + hid
        self.Wf = rng.standard_normal((tot, hid)) * s
        self.Wi = rng.standard_normal((tot, hid)) * s
        self.Wc = rng.standard_normal((tot, hid)) * s
        self.Wo = rng.standard_normal((tot, hid)) * s
        self.hid = hid

    def forward(self, X):
        """X: (T, in_dim) -> (T, hid)."""
        T = X.shape[0]
        h = np.zeros(self.hid)
        c = np.zeros(self.hid)
        out = np.zeros((T, self.hid))
        for t in range(T):
            xh = np.concatenate([X[t], h])
            f = _sigmoid(xh @ self.Wf)
            i = _sigmoid(xh @ self.Wi)
            cc = np.tanh(xh @ self.Wc)
            o = _sigmoid(xh @ self.Wo)
            c = f * c + i * cc
            h = o * np.tanh(c)
            out[t] = h
        return out


# ---------------------------------------------------------------------------
# Hybrid U-Net + LSTM model
# ---------------------------------------------------------------------------

class HybridUNetLSTM:
    """Combines U-Net spatial features with LSTM sequential features."""

    def __init__(self, n_input_logs=4, hidden=8, seed=42):
        self.unet = MiniUNet1D(n_input_logs, mid_ch=hidden, out_ch=hidden, seed=seed)
        self.lstm = CompactLSTM(n_input_logs, hidden, seed + 100)
        # Linear projection: 2*hidden -> 1
        rng = np.random.default_rng(seed + 200)
        self.Wp = rng.standard_normal((2 * hidden, 1)) * 0.1
        self.bp = np.zeros(1)

    def predict(self, X):
        """X: (n_depths, n_input_logs) -> (n_depths,) predicted missing log."""
        f_unet = self.unet.forward(X)          # (L, hidden)
        f_lstm = self.lstm.forward(X)           # (L, hidden)
        fusion = np.concatenate([f_unet, f_lstm], axis=1)
        out = (fusion @ self.Wp + self.bp).ravel()
        return out


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-30)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 5: Hybrid U-Net + LSTM Missing-Log Demo ===\n")
    np.random.seed(42)

    n_depths = 200
    n_logs = 4  # GR, RHOB, NPHI, DTC (known)
    depth = np.linspace(2000, 2200, n_depths)
    X = np.column_stack([
        50 + 30 * np.sin(depth / 20),                  # GR
        2.2 + 0.3 * np.cos(depth / 15),                # RHOB
        0.15 + 0.10 * np.sin(depth / 25),              # NPHI
        90 + 20 * np.sin(depth / 18) + np.random.randn(n_depths) * 2  # DTC
    ])
    y_true = 180 + 40 * np.sin(depth / 22) + np.random.randn(n_depths) * 3  # DTS

    model = HybridUNetLSTM(n_input_logs=n_logs, hidden=6)
    y_pred = model.predict(X)

    print(f"RMSE (untrained) : {rmse(y_true, y_pred):.2f}")
    print(f"R^2  (untrained) : {r_squared(y_true, y_pred):.4f}")
    print("(Note: model is randomly initialized; training requires backprop.)")
    print()


if __name__ == "__main__":
    demo()
