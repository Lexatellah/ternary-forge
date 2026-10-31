"""Train the ternary MNIST MLP and write docs/weights.trit."""
import os
import numpy as np
from ternary_forge import TernaryMLP
from ternary_forge.mnist import load_mnist
from ternary_forge.pack import pack_trits


def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist("data")
    print(f"  train: {x_train.shape}  test: {x_test.shape}")

    print("Training ternary MLP (784 -> 256 -> 10)...")
    model = TernaryMLP([784, 256, 10], seed=0)
    rng = np.random.default_rng(0)
    for ep in range(5):
        idx = rng.permutation(len(x_train))
        for i in range(0, len(x_train), 128):
            j = idx[i:i+128]
            model.train_step(x_train[j], y_train[j], 0.02)
        acc = (model.predict(x_test) == y_test).mean()
        print(f"  epoch {ep+1}: test_acc = {acc*100:.2f}%")

    os.makedirs("docs", exist_ok=True)
    path = "docs/weights.trit"
    total_trits = 0
    with open(path, "wb") as f:
        f.write(b"TRIT")
        f.write(bytes([1]))
        f.write(bytes([len(model.layers)]))
        for layer in model.layers:
            w = layer.get_trits()
            alpha = layer._alpha
            bias = layer.b_shadow
            n_in, n_out = layer.in_features, layer.out_features
            f.write(n_in.to_bytes(4, "little"))
            f.write(n_out.to_bytes(4, "little"))
            f.write(np.float32(alpha).tobytes())
            f.write(bias.astype(np.float32).tobytes())
            trits = w.flatten().astype(int).tolist()
            packed, n = pack_trits(trits)
            f.write(n.to_bytes(4, "little"))
            f.write(packed)
            total_trits += n
            vals, counts = np.unique(w, return_counts=True)
            dist = {int(v): int(c) for v, c in zip(vals, counts)}
            print(f"  layer {n_in}->{n_out}: alpha={alpha:.4f}  dist={dist}")

    size = os.path.getsize(path)
    print(f"\nWrote {path}: {size:,} bytes ({size/1024:.1f} KiB)")
    print(f"Total trits: {total_trits:,}")
    print(f"Bits per weight: {size*8/total_trits:.3f}")


if __name__ == "__main__":
    main()