"""Train a ternary-weight MLP on MNIST and compare to float32."""
import numpy as np
from ternary_forge.mnist import load_mnist
from ternary_forge.net import TernaryMLP


class FloatLinear:
    def __init__(self, in_features, out_features, rng, momentum=0.9):
        scale = 1.0 / np.sqrt(in_features)
        self.w = rng.uniform(-scale, scale, (out_features, in_features)).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.vw = np.zeros_like(self.w)
        self.vb = np.zeros_like(self.b)
        self.momentum = momentum
        self._x = None
    def forward(self, x):
        self._x = x
        return x @ self.w.T + self.b
    def backward(self, grad, lr):
        gw = grad.T @ self._x
        gb = grad.sum(axis=0)
        gx = grad @ self.w
        self.vw = self.momentum * self.vw + gw
        self.vb = self.momentum * self.vb + gb
        self.w -= lr * self.vw
        self.b -= lr * self.vb
        return gx
    def n_weights(self):
        return self.w.size + self.b.size


class FloatMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [FloatLinear(sizes[i], sizes[i+1], rng) for i in range(len(sizes)-1)]
    def forward(self, x):
        h = x
        for i, l in enumerate(self.layers):
            h = l.forward(h)
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
        for l in reversed(self.layers):
            grad = l.backward(grad, lr)
        return loss
    def n_weights(self):
        return sum(l.n_weights() for l in self.layers)


def batches(x, y, batch_size, rng):
    idx = rng.permutation(x.shape[0])
    for i in range(0, x.shape[0], batch_size):
        j = idx[i:i+batch_size]
        yield x[j], y[j]


def evaluate(model, x, y):
    return float((model.predict(x) == y).mean())


def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist("data")
    print(f"  train: {x_train.shape}  test: {x_test.shape}")

    sizes = [784, 256, 10]
    epochs, batch_size = 5, 128
    lr_tern, lr_float = 0.02, 0.05

    print("\n=== Ternary MLP (weights in {-1, 0, +1}) ===")
    ternary = TernaryMLP(sizes, seed=0)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        losses = [ternary.train_step(xb, yb, lr_tern)
                  for xb, yb in batches(x_train, y_train, batch_size, rng)]
        print(f"  epoch {ep+1}: loss={np.mean(losses):.4f}  test_acc={evaluate(ternary, x_test, y_test)*100:.2f}%")

    print("\n=== Float32 MLP (baseline) ===")
    fmlp = FloatMLP(sizes, seed=0)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        losses = [fmlp.train_step(xb, yb, lr_float)
                  for xb, yb in batches(x_train, y_train, batch_size, rng)]
        print(f"  epoch {ep+1}: loss={np.mean(losses):.4f}  test_acc={evaluate(fmlp, x_test, y_test)*100:.2f}%")

    w0 = ternary.layers[0].get_trits()
    vals, counts = np.unique(w0, return_counts=True)
    dist = {float(v): int(c) for v, c in zip(vals, counts)}
    print(f"\n  Layer-0 ternary weight distribution: {dist}")
    print(f"  Layer-0 scale alpha: {ternary.layers[0]._alpha:.4f}")

    n = ternary.n_weights()
    t_bits = n * np.log2(3)
    f_bits = n * 32
    print("\n=== Weight memory ===")
    print(f"  parameters:    {n:,}")
    print(f"  ternary:       {t_bits/8/1024:>8.1f} KiB  ({t_bits/n:.2f} bits/weight)")
    print(f"  float32:       {f_bits/8/1024:>8.1f} KiB  ({f_bits/n:.2f} bits/weight)")
    print(f"  compression:   {f_bits/t_bits:.1f}x")


if __name__ == "__main__":
    main()
