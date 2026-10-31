from ternary_forge.branch import BranchingVM
from ternary_forge.vm import assemble


SRC = """
  LOAD r0, INPUT
  BR3  r0, neg, zero, pos
neg:
  LOAD r1, -1
  JMP  done
zero:
  LOAD r1, 0
  JMP  done
pos:
  LOAD r1, 1
  JMP  done
done:
  OUT  r1
  HALT
"""


def _run(value):
    vm = BranchingVM()
    vm.load(assemble(SRC.replace("INPUT", str(value))))
    vm.run()
    return vm.output[0][1]


def test_br3_all_three_paths():
    assert _run(-7) == -1
    assert _run(0) == 0
    assert _run(42) == 1


def test_br3_negative_boundary():
    assert _run(-1) == -1
    assert _run(1) == 1


def test_br3_only_one_instruction_used_for_selection():
    # The program hits BR3 exactly once, regardless of which branch is taken.
    # Count instructions in the assembled program between BR3 and HALT.
    program = assemble(SRC.replace("INPUT", "5"))
    ops = [instr[0] for instr in program]
    assert ops.count("BR3") == 1
