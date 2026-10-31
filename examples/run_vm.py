"""Run three ternary assembly programs through the VM."""
from ternary_forge.vm import TernaryVM, assemble


PROGRAMS = {
"free_negation": """
; NEG is pure trit inversion. No sign bit. No two's complement. No carry.
  LOAD r0, 100       ; 100 = ++-0+ in balanced ternary
  OUT  r0
  NEG  r0            ; every trit flips sign, nothing else
  OUT  r0
  HALT
""",
"power_of_three": """
; SHL3 shifts the trit vector left by one. That's x3, not x2.
  LOAD r0, 1         ; value = 1
  LOAD r1, 10        ; counter = 10
loop:
  SHL3 r0            ; r0 *= 3
  DEC  r1
  JNZ  loop
  OUT  r0            ; 3^10 = 59049
  HALT
""",
"fibonacci": """
; A real algorithm running natively on trits.
  LOAD r0, 0
  LOAD r1, 1
  LOAD r2, 15
loop:
  OUT  r0
  ADD  r3, r0, r1
  MOV  r0, r1
  MOV  r1, r3
  DEC  r2
  JNZ  loop
  HALT
""",
}


def run_program(name, source):
    print(f"\n=== {name} ===")
    for line in source.strip().splitlines():
        print("  " + line)
    program = assemble(source)
    vm = TernaryVM()
    vm.load(program)
    vm.run()
    print("  --- output ---")
    for reg, val, trit in vm.output:
        print(f"  {reg} = {trit:>16s}  ({val})")


if __name__ == "__main__":
    for name in ("free_negation", "power_of_three", "fibonacci"):
        run_program(name, PROGRAMS[name])
