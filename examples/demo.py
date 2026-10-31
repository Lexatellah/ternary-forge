from ternary_forge import BalancedTernary as BT, TernaryNeuralLayer

print("=== Balanced Ternary Arithmetic ===")
a, b = BT(7), BT(-4)
print(f"a         = {a}  ({a.to_int()})")
print(f"b         = {b}  ({b.to_int()})")
print(f"a + b     = {a + b}  ({(a + b).to_int()})")
print(f"a - b     = {a - b}  ({(a - b).to_int()})")
print(f"a * b     = {a * b}  ({(a * b).to_int()})")
print(f"-a        = {-a}  ({(-a).to_int()})   <- pure inversion, no sign bit")

print("\n=== Ternary Neural Layer ===")
layer = TernaryNeuralLayer(input_dim=3, output_dim=2, seed=42)
x = [BT(1), BT(-1), BT(0)]
y = layer.forward(x)
print(f"input     = {[str(t) for t in x]}")
print(f"output    = {[str(t) for t in y]}")
print("No floats. No dot products. Just trits.")