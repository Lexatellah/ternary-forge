"""One command. The whole case. Reproduce the paper's claims."""
import sys
import time
import numpy as np
from ternary_forge import BalancedTernary as BT, TernaryMLP
from ternary_forge.mnist import load_mnist
from ternary_forge.vm import TernaryVM, assemble
from ternary_forge.pack import pack_trits, unpack_trits


BAR = "=" * 66


def header(title):
    print(f"\n{BAR}\n  {title}\n{BAR}")


def claim(text):
    print(f"  CLAIM:  {text}")


def actual(text):
    print(f"  ACTUAL: {text}")


def ok():
    print("  RESULT: PASS")
    print()


def fail(msg):
    print(f"  RESULT: FAIL - {msg}")
    sys.exit(1)


def main():
    t0 = time.time()
    print()
    print(BAR)
    print("  TERNARY FORGE -- Reproduce the Paper")
    print("  The Binary Accident, verified in one command")
    print(BAR)

    # ---------- [1/4] Arithmetic ----------
    header("[1/4] Balanced ternary arithmetic")
    claim("7 = '+ - +' and -4 = '- -'; negation is pure trit inversion")
    a, b = BT(7), BT(-4)
    actual(f"7 = {repr(a)},  -4 = {repr(b)}")
    if repr(a) != "+-+" or repr(b) != "--":
        fail("arithmetic representation")
    neg_a = -a
    if neg_a.trits != [-t for t in a.trits]:
        fail("negation is not pure inversion")
    actual(f"-7 = {repr(neg_a)}  (every trit flipped, no sign bit)")
    ok()

    # ---------- [2/4] Ternary MLP on MNIST ----------
    header("[2/4] Ternary MLP on MNIST (paper: 5 epochs, 784-256-10)")
    claim("Test accuracy >= 92% with weights restricted to {-1, 0, +1}")
    print("  Loading MNIST (downloads to data/ on first run)...")
    x_train, y_train, x_test, y_test = load_mnist("data")
    actual(f"data: train={x_train.shape}, test={x_test.shape}")

    model = TernaryMLP([784, 256, 10], seed=0)
    rng = np.random.default_rng(0)
    epochs, bs, lr = 5, 128, 0.02
    for ep in range(epochs):
        idx = rng.permutation(len(x_train))
        for i in range(0, len(x_train), bs):
            j = idx[i:i + bs]
            model.train_step(x_train[j], y_train[j], lr)
        acc = (model.predict(x_test) == y_test).mean()
        print(f"    epoch {ep + 1}: test_acc = {acc * 100:.2f}%")

    final_acc = (model.predict(x_test) == y_test).mean()
    actual(f"final test accuracy: {final_acc * 100:.2f}%")
    if final_acc < 0.92:
        fail(f"MNIST accuracy {final_acc * 100:.2f}% below 92%")

    n = model.n_weights()
    t_kib = n * np.log2(3) / 8 / 1024
    f_kib = n * 32 / 8 / 1024
    actual(f"weights: {n:,} params @ 1.58 bits = {t_kib:.1f} KiB "
           f"(float32: {f_kib:.1f} KiB, {f_kib / t_kib:.1f}x larger)")
    ok()

    # ---------- [3/4] Ternary VM ----------
    header("[3/4] Ternary VM -- native NEG and SHL3")
    claim("NEG flips all trits in one instruction; SHL3 multiplies by 3")
    src = """
      LOAD r0, 100
      OUT  r0
      NEG  r0
      OUT  r0
      LOAD r0, 1
      SHL3 r0
      SHL3 r0
      OUT  r0
      HALT
    """
    vm = TernaryVM()
    vm.load(assemble(src))
    vm.run()
    for reg, val, trit in vm.output:
        print(f"    {reg} = {trit:>14s}  ({val})")
    if vm.output[1][1] != -100:
        fail("NEG(100) != -100")
    if vm.output[2][1] != 9:
        fail("SHL3 twice on 1 != 9")
    actual("NEG(100) = -100 in one instruction; two SHL3 = 9 in two")
    ok()

    # ---------- [4/4] Trit packing ----------
    header("[4/4] Trit packing -- 5 trits per byte")
    claim("3^5 = 243 < 256, so 5 trits fit in one byte = 99.06% of optimum")
    rng2 = np.random.default_rng(0)
    trits = rng2.choice([-1, 0, 1], size=200_000).tolist()
    packed, n_tr = pack_trits(trits)
    actual(f"200,000 trits -> {len(packed):,} bytes  "
           f"({len(packed) / 1024:.1f} KiB)")
    actual(f"compression vs float32 (800 KB):  {800_000 / len(packed):.1f}x")
    if unpack_trits(packed, n_tr) != trits:
        fail("packing roundtrip is not bit-exact")
    actual("roundtrip: PASS (bit-exact, verified)")
    ok()

    # ---------- Verdict ----------
    elapsed = time.time() - t0
    print(BAR)
    print(f"  VERDICT: all 4 claims verified in {elapsed:.1f}s")
    print("  The binary accident is reproducible.")
    print(BAR)


if __name__ == "__main__":
    main()
