"""A minimal ternary virtual machine. All arithmetic in balanced ternary."""
from __future__ import annotations
from typing import List, Dict, Tuple
from .core import BalancedTernary


class TernaryVM:
    """8-register ternary VM. Every op is native BalancedTernary."""

    def __init__(self):
        self.regs = [BalancedTernary(0) for _ in range(8)]
        self.pc = 0
        self.zero_flag = False
        self.halted = False
        self.program = []
        self.output = []

    def load(self, program):
        self.program = program
        self.pc = 0
        self.halted = False
        self.zero_flag = False
        self.output = []

    def step(self):
        if self.halted or self.pc >= len(self.program):
            self.halted = True
            return False
        instr = self.program[self.pc]
        op = instr[0]
        self.pc += 1

        if op == "LOAD":
            self.regs[instr[1]] = BalancedTernary(instr[2])
        elif op == "MOV":
            self.regs[instr[1]] = BalancedTernary(self.regs[instr[2]])
        elif op == "ADD":
            self.regs[instr[1]] = self.regs[instr[2]] + self.regs[instr[3]]
        elif op == "SUB":
            self.regs[instr[1]] = self.regs[instr[2]] - self.regs[instr[3]]
        elif op == "MUL":
            self.regs[instr[1]] = self.regs[instr[2]] * self.regs[instr[3]]
        elif op == "NEG":
            self.regs[instr[1]] = -self.regs[instr[1]]
        elif op == "SHL3":
            self.regs[instr[1]] = BalancedTernary([0] + self.regs[instr[1]].trits)
        elif op == "SHR3":
            r = self.regs[instr[1]].trits
            self.regs[instr[1]] = BalancedTernary(r[1:]) if len(r) > 1 else BalancedTernary(0)
        elif op == "INC":
            self.regs[instr[1]] = self.regs[instr[1]] + BalancedTernary(1)
        elif op == "DEC":
            self.regs[instr[1]] = self.regs[instr[1]] - BalancedTernary(1)
            self.zero_flag = (self.regs[instr[1]].to_int() == 0)
        elif op == "CMP":
            self.zero_flag = (self.regs[instr[1]] == self.regs[instr[2]])
        elif op == "JMP":
            self.pc = instr[1]
        elif op == "JZ":
            if self.zero_flag:
                self.pc = instr[1]
        elif op == "JNZ":
            if not self.zero_flag:
                self.pc = instr[1]
        elif op == "OUT":
            r = self.regs[instr[1]]
            self.output.append((f"r{instr[1]}", r.to_int(), repr(r)))
        elif op == "HALT":
            self.halted = True
        elif op == "NOP":
            pass
        else:
            raise ValueError(f"Unknown opcode: {op}")
        return not self.halted

    def run(self, max_steps=1_000_000):
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"Step limit {max_steps} exceeded")
        return self.output


def assemble(source):
    """Assemble textual ternary assembly into a tuple list."""
    lines = source.splitlines()
    labels = {}
    clean = []
    for raw in lines:
        line = raw.split(";")[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.endswith(":"):
            labels[stripped[:-1].strip()] = len(clean)
            continue
        clean.append(stripped)

    program = []
    for line in clean:
        parts = line.replace(",", " ").split()
        op = parts[0].upper()
        args = []
        for a in parts[1:]:
            if a.startswith("r") and a[1:].isdigit():
                args.append(int(a[1:]))
            elif a in labels:
                args.append(labels[a])
            else:
                try:
                    args.append(int(a))
                except ValueError:
                    args.append(a)
        program.append((op, *args))
    return program
