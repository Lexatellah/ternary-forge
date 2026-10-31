"""
Native ternary-weight MLP with straight-through estimator (STE) training.

v0.2.1 - fixes the "all-zero weights" bug:
  - adaptive per-layer threshold:   Δ = 0.7 * mean(|W|)
  - adaptive per-layer scale:       α = mean(|W| where |W| > Δ)
  - forward uses W_eff = α * sign(W)
  - backward STE uses W_eff for grad_x so gradients keep flowing
  - light L2 weight decay keeps α bounded
"""
from __future__ import annotations
import numpy as np


def ternarize(w, delta=None):
    """Return (trit matrix in {-1, 0, +1}, scale alpha)."""
    w = np.asarray(w, dtype=np.float32)
    if delta is None:
        delta = 0.7 * float(np.mean(np.abs(w)))
    mask = np.abs(w) > delta
    alpha = float(np.mean(np.abs(w[mask]))) if mask.any() else 1.0
    alpha = min(alpha, 1.0)  # safety bound
    trit = np.zeros_like(w)
    trit[mask] = np.sign(w[mask])
    return trit, alpha


class TernaryLinear:
    def __init__(self, in_features, out_features, rng, momentum=0.9):
        scale = 1.0 / np.sqrt(in_features)
        self.w_shadow = rng.uniform(-scale, scale, (out_features, in_features)).astype(np.float32)
        self.b_shadow = np.zeros(out_features, dtype=np.float32)
        self.vw = np.zeros_like(self.w_shadow)
        self.vb = np.zeros_like(self.b_shadow)
        self.momentum = momentum
        self.in_features = in_features
        self.out_features = out_features
        self._x = None
        self._w_trit = None
        self._w_eff = None
        self._alpha = 1.0

    def _quantize(self):
        self._w_trit, self._alpha = ternarize(self.w_shadow)
        self._w_eff = self._alpha * self._w_trit

    def get_trits(self):
        if self._w_trit is None:
            self._quantize()
        return self._w_trit

    def forward(self, x):
        self._x = x
        self._quantize()
        return x @ self._w_eff.T + self.b_shadow

    def backward(self, grad_out, lr, weight_decay=1e-3):
        grad_w = grad_out.T @ self._x
        grad_b = grad_out.sum(axis=0)
        grad_x = grad_out @ self._w_eff
        self.vw = self.momentum * self.vw + grad_w
        self.vb = self.momentum * self.vb + grad_b
        self.w_shadow -= lr * (self.vw + weight_decay * self.w_shadow)
        self.b_shadow -= lr * self.vb
        np.clip(self.w_shadow, -1.0, 1.0, out=self.w_shadow)
        return grad_x

    def n_weights(self):
        return self.w_shadow.size + self.b_shadow.size


class TernaryMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.layers = [TernaryLinear(sizes[i], sizes[i+1], rng)
                       for i in range(len(sizes)-1)]

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer.forward(h)
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def train_step(self, x, y, lr):
        logits = self.forward(x)
        ls = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(ls); probs = exp / exp.sum(axis=1, keepdims=True)
        n = x.shape[0]
        loss = float(-np.log(probs[np.arange(n), y] + 1e-9).mean())
        grad = probs.copy(); grad[np.arange(n), y] -= 1.0; grad /= n
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return loss

    def n_weights(self):
        return sum(layer.n_weights() for layer in self.layers)
