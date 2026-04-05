#!/usr/bin/env python3
"""
Article 4: Multimodal Learning With Explicit Tensor Interaction for
           Reservoir Permeability Prediction
Authors: Yu Fang, Lizhi Xiao, Jun Zhou, Guangzhi Liao, Xiaoyu Wang
Ref: Petrophysics, Vol. 66, No. 5 (October 2025), pp. 764-784.
     DOI: 10.30632/PJV66N5-2025a4

Implements (pure-NumPy, no deep-learning framework required):
  - Feature extraction branches: LSTM-like (time-series logs),
    CNN-like (NMR T2 images), DNN (structured text features)
  - Explicit tensor interaction layer:
      binary planes  :  f_a ⊗ f_b  for each pair (a,b)
      ternary core   :  f_1 ⊗ f_2 ⊗ f_3
  - Fusion & regression head for permeability prediction
"""

import numpy as np


# ---------------------------------------------------------------------------
# Simple LSTM cell (single layer, forward only)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class SimpleLSTM:
    """Single-layer LSTM for sequence feature extraction."""

    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.1
        self.Wf = rng.standard_normal((input_size + hidden_size, hidden_size)) * s
        self.Wi = rng.standard_normal((input_size + hidden_size, hidden_size)) * s
        self.Wc = rng.standard_normal((input_size + hidden_size, hidden_size)) * s
        self.Wo = rng.standard_normal((input_size + hidden_size, hidden_size)) * s
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, X):
        """X: (seq_len, input_size). Returns last hidden state (hidden_size,)."""
        T, d = X.shape
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for t in range(T):
            xh = np.concatenate([X[t], h])
            f = _sigmoid(xh @ self.Wf + self.bf)
            i = _sigmoid(xh @ self.Wi + self.bi)
            c_hat = np.tanh(xh @ self.Wc + self.bc)
            o = _sigmoid(xh @ self.Wo + self.bo)
            c = f * c + i * c_hat
            h = o * np.tanh(c)
        return h


# ---------------------------------------------------------------------------
# Simple 1-D CNN for 2-D image (flattened T2 spectrum)
# ---------------------------------------------------------------------------

class SimpleCNN1D:
    """1-D convolution + global average pooling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((out_channels, in_channels, kernel_size)) * 0.1
        self.b = np.zeros(out_channels)
        self.ks = kernel_size

    def forward(self, X):
        """X: (length, in_channels). Returns (out_channels,)."""
        L, C = X.shape
        pad = self.ks // 2
        Xp = np.pad(X, ((pad, pad), (0, 0)), mode='constant')
        Oc = self.W.shape[0]
        out = np.zeros((L, Oc))
        for oc in range(Oc):
            for p in range(L):
                patch = Xp[p:p + self.ks, :]  # (ks, C)
                out[p, oc] = np.sum(self.W[oc] * patch.T) + self.b[oc]
        out = np.maximum(out, 0)  # ReLU
        return out.mean(axis=0)  # global average pool


# ---------------------------------------------------------------------------
# DNN for structured features
# ---------------------------------------------------------------------------

class SimpleDNN:
    """Two-layer DNN."""

    def __init__(self, in_dim, hidden_dim, out_dim, seed=2):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden_dim)) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, out_dim)) * 0.1
        self.b2 = np.zeros(out_dim)

    def forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h @ self.W2 + self.b2


# ---------------------------------------------------------------------------
# Explicit Tensor Interaction Network (ETIN)
# ---------------------------------------------------------------------------

class ETIN:
    """Explicit Tensor Interaction Network for multimodal permeability."""

    def __init__(self, seq_len=20, seq_feat=5, img_len=64, img_chan=1,
                 text_dim=4, hidden=8, seed=42):
        self.lstm = SimpleLSTM(seq_feat, hidden, seed)
        self.cnn = SimpleCNN1D(img_chan, hidden, kernel_size=3, seed=seed + 1)
        self.dnn = SimpleDNN(text_dim, hidden, hidden, seed + 2)
        self.hidden = hidden
        rng = np.random.default_rng(seed + 3)
        # Regression head weights
        fusion_dim = 3 * hidden + 3 * hidden ** 2 + hidden ** 3
        self.Wr = rng.standard_normal((fusion_dim, 1)) * 0.01
        self.br = np.zeros(1)

    def extract_features(self, seq, img, text):
        f1 = self.lstm.forward(seq)    # (H,)
        f2 = self.cnn.forward(img)     # (H,)
        f3 = self.dnn.forward(text)    # (H,)
        return f1, f2, f3

    def tensor_interaction(self, f1, f2, f3):
        # Binary interaction planes
        b12 = np.outer(f1, f2).ravel()
        b13 = np.outer(f1, f3).ravel()
        b23 = np.outer(f2, f3).ravel()
        # Ternary interaction core
        t123 = np.einsum('i,j,k->ijk', f1, f2, f3).ravel()
        return np.concatenate([f1, f2, f3, b12, b13, b23, t123])

    def predict_single(self, seq, img, text):
        f1, f2, f3 = self.extract_features(seq, img, text)
        fusion = self.tensor_interaction(f1, f2, f3)
        return float((fusion @ self.Wr + self.br).item())

    def predict(self, sequences, images, texts):
        return np.array([self.predict_single(s, im, t)
                         for s, im, t in zip(sequences, images, texts)])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("=== Article 4: ETIN Multimodal Permeability Demo ===\n")
    np.random.seed(42)
    N = 30
    seq_len, seq_feat = 20, 5
    img_len, img_chan = 64, 1
    text_dim = 4

    sequences = [np.random.randn(seq_len, seq_feat) for _ in range(N)]
    images = [np.random.randn(img_len, img_chan) for _ in range(N)]
    texts = [np.random.randn(text_dim) for _ in range(N)]

    model = ETIN(seq_len, seq_feat, img_len, img_chan, text_dim, hidden=4)
    preds = model.predict(sequences, images, texts)
    print(f"Predicted permeability range: {preds.min():.3f} – {preds.max():.3f}")
    print(f"Mean predicted permeability : {preds.mean():.3f}")
    print()


if __name__ == "__main__":
    demo()
