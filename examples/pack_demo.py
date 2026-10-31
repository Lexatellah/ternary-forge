"""Pack ternary weights to bytes. Prove 20x compression with a real file."""
import os
import numpy as np
from ternary_forge.pack import pack_trits, unpack_trits


def main():
    print("=== Trit Packing: 5 trits per byte ===")
    print(f"  3^5 = {3**5} states fit in 2^8 = {2**8} values")
    print(f"  effective:  {8/5:.3f} bits/trit")
    print(f"  ideal:      {np.log2(3):.3f} bits/trit")
    print(f"  efficiency: {np.log2(3)/(8/5)*100:.2f}%")

    print("\n=== Roundtrip on 1000 random trits ===")
    rng = np.random.default_rng(42)
    trits = rng.choice([-1, 0, 1], size=1000).tolist()
    packed, n = pack_trits(trits)
    recovered = unpack_trits(packed, n)
    print(f"  input:      {len(trits)} trits")
    print(f"  packed:     {len(packed)} bytes")
    print(f"  roundtrip:  {recovered == trits}")

    print("\n=== Memory for 200,000 ternary weights ===")
    N = 200_000
    packed_bytes = (N + 4) // 5
    float_bytes = N * 4
    twobit_bytes = N // 4
    print(f"  float32:     {float_bytes/1024:>8.1f} KiB  (32.00 bits/weight)")
    print(f"  naive 2-bit: {twobit_bytes/1024:>8.1f} KiB  (2.00 bits/weight)")
    print(f"  packed:      {packed_bytes/1024:>8.1f} KiB  ({8*packed_bytes/N:.2f} bits/weight)")
    print(f"  vs float32:  {float_bytes/packed_bytes:.1f}x smaller")
    print(f"  vs 2-bit:    {twobit_bytes/packed_bytes:.2f}x smaller")

    print("\n=== Writing a real .trit file ===")
    trits_big = rng.choice([-1, 0, 1], size=N).tolist()
    packed_big, n = pack_trits(trits_big)
    path = "example.trit"
    with open(path, "wb") as f:
        f.write(b"TRIT")
        f.write(n.to_bytes(4, "little"))
        f.write(packed_big)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        n_read = int.from_bytes(f.read(4), "little")
        payload = f.read()
        recovered = unpack_trits(payload, n_read)
    print(f"  wrote:      {os.path.abspath(path)}")
    print(f"  size:       {size:,} bytes  ({size/1024:.1f} KiB)")
    print(f"  magic:      {magic}")
    print(f"  n_trits:    {n_read:,}")
    print(f"  roundtrip:  {recovered == trits_big}")
    print(f"  sample:     first 10 trits = {trits_big[:10]}")
    print(f"              first 2 bytes = {list(payload[:2])}")
    os.remove(path)


if __name__ == "__main__":
    main()
