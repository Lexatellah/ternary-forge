"""A native ternary perceptron layer. Weights are trits. No floats, no dot products."""
from __future__ import annotations
import random
from typing import List
from .core import BalancedTernary


def _sign(n: int) -> int:
    return 1 if n > 0 else -1 if n < 0 else 0


class TernaryNeuralLayer:
    def __init__(self, input_dim: int, output_dim: int, seed: int | None = None):
        rng = random.Random(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights: List[List[BalancedTernary]] = [
            [BalancedTernary(rng.choice((-1, 0, 1))) for _ in range(input_dim)]
            for _ in range(output_dim)
        ]
        self.biases: List[BalancedTernary] = [
            BalancedTernary(rng.randint(-2, 2)) for _ in range(output_dim)
        ]

    def forward(self, inputs: List[BalancedTernary]) -> List[BalancedTernary]:
        if len(inputs) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} inputs, got {len(inputs)}")
        outputs: List[BalancedTernary] = []
        for row, bias in zip(self.weights, self.biases):
            acc = bias
            for w, x in zip(row, inputs):
                wt = w.to_int()
                if wt == 1:
                    acc = acc + x
                elif wt == -1:
                    acc = acc - x
            outputs.append(BalancedTernary(_sign(acc.to_int())))
        return outputs