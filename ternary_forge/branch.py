"""BR3: a ternary three-way branch. One instruction, three targets."""
from .vm import TernaryVM


class BranchingVM(TernaryVM):
    """TernaryVM with one extra instruction: BR3 reg, neg_t, zero_t, pos_t."""

    def step(self):
        if self.halted or self.pc >= len(self.program):
            self.halted = True
            return False
        instr = self.program[self.pc]
        if instr[0] == "BR3":
            _, reg, neg_t, zero_t, pos_t = instr
            v = self.regs[reg].to_int()
            self.pc = neg_t if v < 0 else zero_t if v == 0 else pos_t
            return True
        return super().step()
