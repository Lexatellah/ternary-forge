import itertools
import numpy as np
from ternary_forge.pack import pack_trits, unpack_trits, pack_matrix, unpack_matrix


def test_roundtrip_random():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        trits = rng.choice([-1, 0, 1], size=1000).tolist()
        packed, n = pack_trits(trits)
        assert unpack_trits(packed, n) == trits


def test_byte_size():
    for n in [0, 1, 4, 5, 6, 10, 100, 1000]:
        packed, _ = pack_trits([1] * n)
        assert len(packed) == (n + 4) // 5


def test_every_5trit_combination_fits_in_a_byte():
    # Exhaustive: every one of the 243 possible 5-trit chunks must fit in [0, 255]
    for combo in itertools.product([-1, 0, 1], repeat=5):
        packed, n = pack_trits(list(combo))
        assert len(packed) == 1
        assert 0 <= packed[0] <= 255
        assert unpack_trits(packed, 5) == list(combo)


def test_matrix_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.choice([-1, 0, 1], size=(32, 64)).astype(np.int8)
    data, shape, n = pack_matrix(m)
    m2 = unpack_matrix(data, shape, n)
    assert np.array_equal(m, m2)
