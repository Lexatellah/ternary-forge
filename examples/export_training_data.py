"""Export a compact MNIST subset for in-browser training (4-bit packed)."""
import os
import numpy as np
from ternary_forge.mnist import load_mnist


def pack_4bit(pixels_flat):
    """784 uint8 pixels -> 392 bytes (4-bit each, 2 pixels per byte)."""
    small = (np.asarray(pixels_flat) // 16).astype(np.uint8)  # 0..15
    packed = np.zeros(392, dtype=np.uint8)
    packed = (small[0::2] << 4) | small[1::2]
    return packed


def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist("data")
    x_train = (x_train * 255).astype(np.uint8)
    x_test = (x_test * 255).astype(np.uint8)

    n_train, n_test = 1000, 200
    train_x = x_train[:n_train]
    train_y = y_train[:n_train].astype(np.uint8)
    test_x = x_test[:n_test]
    test_y = y_test[:n_test].astype(np.uint8)

    os.makedirs("docs", exist_ok=True)
    path = "docs/train_data.bin"
    with open(path, "wb") as f:
        f.write(b"TRNB")                              # magic [4]
        f.write(bytes([1]))                           # version [1]
        f.write(n_train.to_bytes(4, "little"))        # n_train [4]
        f.write(n_test.to_bytes(4, "little"))         # n_test [4]
        f.write((784).to_bytes(2, "little"))          # n_pixels [2]
        f.write(b"\x00\x00\x00")                      # padding [3]
        f.write(train_y.tobytes())                    # train labels [n_train]
        f.write(test_y.tobytes())                     # test labels [n_test]
        for i in range(n_train):
            f.write(pack_4bit(train_x[i]).tobytes())  # 392 bytes each
        for i in range(n_test):
            f.write(pack_4bit(test_x[i]).tobytes())   # 392 bytes each

    size = os.path.getsize(path)
    print(f"Wrote {path}: {size:,} bytes ({size/1024:.1f} KiB)")
    print(f"  train: {n_train} samples x 392 bytes = {n_train*392:,}")
    print(f"  test:  {n_test} samples x 392 bytes = {n_test*392:,}")


if __name__ == "__main__":
    main()