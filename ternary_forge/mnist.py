"""Minimal MNIST loader. Downloads the IDX files if not present. Numpy only."""
from __future__ import annotations
import gzip
import os
import urllib.request
import numpy as np


MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: str) -> None:
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, dest)


def _load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"Bad magic for images: {magic}")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    imgs = np.frombuffer(data, dtype=np.uint8, offset=16)
    return imgs.reshape(n, rows * cols).astype(np.float32) / 255.0


def _load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"Bad magic for labels: {magic}")
    return np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)


def load_mnist(cache_dir: str = "data"):
    """Return (x_train, y_train, x_test, y_test)."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for key, fname in FILES.items():
        local = os.path.join(cache_dir, fname)
        if not os.path.exists(local):
            for mirror in MIRRORS:
                try:
                    _download(mirror + fname, local)
                    break
                except Exception as e:
                    print(f"  mirror failed ({mirror}): {e}")
            else:
                raise RuntimeError(f"Could not download {fname} from any mirror")
        paths[key] = local
    return (
        _load_images(paths["train_images"]),
        _load_labels(paths["train_labels"]),
        _load_images(paths["test_images"]),
        _load_labels(paths["test_labels"]),
    )