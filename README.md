# ternary-forge

**Native balanced-ternary arithmetic, a ternary neural network, a ternary VM, and a trit-packing file format — in pure Python/NumPy.**

Binary is not a law of nature. It is an industrial accident. This repository is a fork in the road.

## Results

| What | Number | Where |
|---|---|---|
| Ternary MLP on MNIST | **93.20%** (float baseline: 92.28%) | `examples/train_mnist.py` |
| Weight storage | **1.58 bits/weight** (float32: 32) | `examples/train_mnist.py` |
| Trit packing | **5 trits/byte**, 99.06% of Shannon optimum | `examples/pack_demo.py` |
| VM | 8-register balanced-ternary ISA, native `NEG` and `SHL3` | `examples/run_vm.py` |

## Install

    pip install -e '.[dev]'

## Run the proof

    python -m pytest -q                  # 11 tests
    python examples/train_mnist.py       # ternary vs float MNIST
    python examples/run_vm.py            # ternary assembly programs
    python examples/pack_demo.py         # 40 KB .trit file, roundtrip

## The argument

See **[docs/BINARY_ACCIDENT.md](docs/BINARY_ACCIDENT.md)** for the citable case: Setun (1958), the 5500FP (2026), Ternary Weight Networks (2016), and the three results above as evidence.

## What this is not

- Not a wrapper around PyTorch. The MLP is hand-written NumPy.
- Not a binary emulator pretending to be ternary. The VM’s `NEG` is one instruction that flips trits.
- Not a toy. The packing format is exhaustive-tested over all 243 five-trit states.

## License

MIT
