"""
Native balanced-ternary arithmetic.

Trits are -1, 0, +1. No sign bit. No two's complement. Negation is inversion.

Addition is done trit-by-trit with a native ternary carry.
Multiplication is shift-and-add using native ternary addition.
"""
from __future__ import annotations
from typing import List, Union


class BalancedTernary:
    __slots__ = ("trits",)  # least-significant trit first

    def __init__(self, value: Union[int, str, List[int], "BalancedTernary"] = 0):
        if isinstance(value, BalancedTernary):
            self.trits = list(value.trits)
        elif isinstance(value, int):
            self.trits = self._from_int(value)
        elif isinstance(value, str):
            self.trits = self._parse(value)
        elif isinstance(value, (list, tuple)):
            self.trits = self._normalize(list(value))
        else:
            raise TypeError(f"Cannot build BalancedTernary from {type(value).__name__}")

    # ---------- constructors ----------

    @staticmethod
    def _normalize(trits: List[int]) -> List[int]:
        for t in trits:
            if t not in (-1, 0, 1):
                raise ValueError(f"Invalid trit: {t}")
        while len(trits) > 1 and trits[-1] == 0:
            trits.pop()
        return trits

    @staticmethod
    def _from_int(n: int) -> List[int]:
        if n == 0:
            return [0]
        trits: List[int] = []
        while n != 0:
            r = n % 3
            if r == 0:
                trits.append(0); n //= 3
            elif r == 1:
                trits.append(1); n = (n - 1) // 3
            else:
                trits.append(-1); n = (n + 1) // 3
        return BalancedTernary._normalize(trits)

    @staticmethod
    def _parse(s: str) -> List[int]:
        s = s.strip()
        if not s:
            raise ValueError("Empty string")
        mapping = {"+": 1, "0": 0, "-": -1}
        try:
            trits = [mapping[c] for c in reversed(s)]  # MSB first in string
        except KeyError as e:
            raise ValueError(f"Bad symbol in {s!r}: {e.args[0]}") from None
        return BalancedTernary._normalize(trits)

    # ---------- conversion ----------

    def to_int(self) -> int:
        total = 0
        for i, t in enumerate(self.trits):
            total += t * (3 ** i)
        return total

    # ---------- arithmetic ----------

    def __add__(self, other: "BalancedTernary") -> "BalancedTernary":
        if not isinstance(other, BalancedTernary):
            return NotImplemented
        a, b = self.trits, other.trits
        n = max(len(a), len(b))
        out: List[int] = []
        carry = 0
        for i in range(n):
            s = (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) + carry
            if   s ==  0: t, carry =  0,  0
            elif s ==  1: t, carry =  1,  0
            elif s == -1: t, carry = -1,  0
            elif s ==  2: t, carry = -1,  1
            elif s == -2: t, carry =  1, -1
            elif s ==  3: t, carry =  0,  1
            elif s == -3: t, carry =  0, -1
            out.append(t)
        if carry:
            out.append(carry)
        return BalancedTernary(out)

    def __neg__(self) -> "BalancedTernary":
        # Negation is pure inversion. No sign bit. This is the point.
        return BalancedTernary([-t for t in self.trits])

    def __sub__(self, other: "BalancedTernary") -> "BalancedTernary":
        return self + (-other)

    def __mul__(self, other: "BalancedTernary") -> "BalancedTernary":
        if not isinstance(other, BalancedTernary):
            return NotImplemented
        result = BalancedTernary(0)
        for i, t in enumerate(other.trits):
            if t == 0:
                continue
            partial = BalancedTernary([0] * i + list(self.trits))
            if t == -1:
                partial = -partial
            result = result + partial
        return result

    # ---------- comparison & display ----------

    def __eq__(self, other) -> bool:
        if isinstance(other, BalancedTernary):
            return self.trits == other.trits
        if isinstance(other, int):
            return self.to_int() == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self.trits))

    def __repr__(self) -> str:
        symbols = {1: "+", 0: "0", -1: "-"}
        return "".join(symbols[t] for t in reversed(self.trits))

    def __str__(self) -> str:
        return repr(self)