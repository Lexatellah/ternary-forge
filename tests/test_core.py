from ternary_forge.core import BalancedTernary as BT


def test_roundtrip():
    for n in range(-200, 201):
        assert BT(n).to_int() == n


def test_negation_is_inversion():
    a = BT(7)
    assert (-a).trits == [-t for t in a.trits]
    assert (-a).to_int() == -7


def test_addition_matches_int():
    for a in range(-50, 51, 7):
        for b in range(-50, 51, 5):
            assert (BT(a) + BT(b)).to_int() == a + b


def test_subtraction_matches_int():
    for a in range(-40, 41, 3):
        for b in range(-40, 41, 4):
            assert (BT(a) - BT(b)).to_int() == a - b


def test_multiplication_matches_int():
    for a in range(-20, 21):
        for b in range(-20, 21):
            assert (BT(a) * BT(b)).to_int() == a * b


def test_parse_and_repr():
    a = BT(7)          # 7 = 9 - 3 + 1 -> "+-+"
    assert repr(a) == "+-+"
    assert BT("+-+").to_int() == 7
    assert BT("-+-").to_int() == -7


def test_neural_layer_shapes():
    from ternary_forge.nn import TernaryNeuralLayer
    layer = TernaryNeuralLayer(3, 2, seed=42)
    out = layer.forward([BT(1), BT(-1), BT(0)])
    assert len(out) == 2
    assert all(o.to_int() in (-1, 0, 1) for o in out)