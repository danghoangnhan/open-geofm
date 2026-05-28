"""Algorithm 1 metric-condition sampler. CPU-bound.

Blueprint §2 Phase 2 (the novel part). Implements the paper's verbatim pseudocode:
    P_new = (P \\ M_del) ∪ M_add, with |M_del| = |M_add| = n.
"""
