"""Pack trits into bytes. 5 trits per byte, 98.9% of theoretical optimum."""
from __future__ import annotations
import numpy as np


def pack_trits(trits):
    """Pack a list of trits (-1, 0, 1) into bytes. 5 trits per byte.

    3^5 = 243 states fit into 2^8 = 256 values. Perfect for a byte.
    Returns (packed_bytes, original_length).
    """
    padded = list(trits)
    while len(padded) % 5 != 0:
        padded.append(0)
    out = bytearray()
    for i in range(0, len(padded), 5):
        chunk = padded[i:i+5]
        n = 0
        for j, t in enumerate(chunk):
            n += (t + 1) * (3 ** j)
        out.append(n)
    return bytes(out), len(trits)


def unpack_trits(data, n_trits):
    """Reverse of pack_trits. Returns exactly n_trits trits."""
    trits = []
    for byte in data:
        n = byte
        for _ in range(5):
            trits.append((n % 3) - 1)
            n //= 3
    return trits[:n_trits]


def pack_matrix(matrix):
    """Pack a numpy matrix of trits. Returns (bytes, shape)."""
    m = np.asarray(matrix)
    trits = m.flatten().astype(int).tolist()
    packed, n = pack_trits(trits)
    return packed, m.shape, n


def unpack_matrix(data, shape, n_trits):
    """Unpack bytes back into a numpy trit matrix."""
    trits = unpack_trits(data, n_trits)
    return np.array(trits, dtype=np.int8).reshape(shape)
