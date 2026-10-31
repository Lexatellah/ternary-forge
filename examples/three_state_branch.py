"""BR3: three-way branch in one instruction. Binary cannot do this natively."""
from ternary_forge.branch import BranchingVM
from ternary_forge.vm import assemble


PROGRAM = """
  LOAD r0, INPUT_VAL
  BR3  r0, neg, zero, pos
neg:
  LOAD r1, -1
  OUT  r1
  JMP  done
zero:
  LOAD r1, 0
  OUT  r1
  JMP  done
pos:
  LOAD r1, 1
  OUT  r1
  JMP  done
done:
  HALT
"""


def run_for(value):
    src = PROGRAM.replace("INPUT_VAL", str(value))
    vm = BranchingVM()
    vm.load(assemble(src))
    vm.run()
    return vm.output[0][1]


def main():
    print("=" * 64)
    print("  BR3: a ternary three-way branch")
    print("=" * 64)
    print()
    print("  Ternary program:")
    print("    LOAD r0, X")
    print("    BR3  r0, neg, zero, pos     ; <-- ONE instruction")
    print("    ...")
    print()
    print("  Binary equivalent for the same three-way choice:")
    print("    CMP r0, 0")
    print("    JL  neg")
    print("    JE  zero")
    print("    JMP pos                     ; <-- FOUR instructions")
    print()
    print("  --- running three inputs ---")
    for val, expected in [(-7, -1), (0, 0), (42, 1)]:
        result = run_for(val)
        status = "PASS" if result == expected else "FAIL"
        print(f"    BR3({val:>4}) -> {result:>2}   [{status}]")
    print()
    print("  --- what this proves ---")
    print("    * Three states, one trit, zero waste.")
    print("    * Binary needs 2 bits for 3 states (one state is invalid).")
    print("    * Binary needs 4 instructions to select three branches.")
    print("    * Ternary needs 1.")
    print()
    print("  The state IS the selector. That is the point.")


if __name__ == "__main__":
    main()
