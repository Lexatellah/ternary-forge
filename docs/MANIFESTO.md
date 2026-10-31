# The Binary Accident

1. Binary was chosen for industrial reasons, not mathematical ones.
2. Balanced ternary is more efficient per digit and symmetric under negation.
3. Setun proved it works at scale in 1958. The 5500FP proved it works in 2026.
4. The AI world is groping for alternatives to float-heavy dense compute.
5. Ternary neural layers are native integer hardware — they are what a ternary
   chip would actually run.

Roadmap:
- v0.1  native arithmetic + ternary perceptron   (this release)
- v0.2  ternary MNIST trainer with straight-through estimator
- v0.3  ternary virtual machine (5500FP-style ISA subset)
- v1.0  open ternary AI accelerator specification