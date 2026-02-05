"""
GodNode – Vector-Space Paraconsistent Logic Engine (Final Version)
Author: Venkata Ramana Kurumalla
Version: v1.0-final
License: MIT

Summary:
- Vector-native propositional logic
- Paraconsistent (no explosion)
- Non-monotonic inference
- Relevance-preserving rule application
- Fractional binding (no sign saturation)
- Approximate unbinding
- Stable 4096-D geometry
"""

import numpy as np
from dataclasses import dataclass
from typing import List

# ============================================================
# 1. GLOBAL CONFIGURATION
# ============================================================

HDC_DIM = 4096
SEED = 42
CONTRADICTION_THRESHOLD = 0.85
MAX_INFERENCE_STEPS = 6

rng = np.random.default_rng(SEED)

# ============================================================
# 2. CORE VECTOR OPERATIONS
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def random_vec() -> np.ndarray:
    return normalize(rng.standard_normal(HDC_DIM))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fractional binding preserving magnitude information"""
    return normalize(a * b)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    return normalize(np.sum(vectors, axis=0))


def negate(v: np.ndarray) -> np.ndarray:
    return -v

# ============================================================
# 3. SYMBOL REGISTRY
# ============================================================

SYMBOLS = {}


def vec(name: str) -> np.ndarray:
    if name not in SYMBOLS:
        SYMBOLS[name] = random_vec()
    return SYMBOLS[name]

# Atomic propositions
P = vec("P")
Q = vec("Q")
R = vec("R")

# Logical operators
IMPLIES = vec("IMPLIES")
AND = vec("AND")

# ============================================================
# 4. PARACONSISTENT CONTRADICTION METRIC
# ============================================================

def contradiction_strength(state: np.ndarray, proposal: np.ndarray) -> float:
    """
    Measures near-incoherence rather than strict negation.
    Enables non-monotonic belief revision.
    """
    return max(
        cosine(state, negate(proposal)),
        cosine(proposal, negate(state))
    )

# ============================================================
# 5. LOGIC RULE DEFINITION
# ============================================================

@dataclass
class LogicRule:
    name: str
    precondition: np.ndarray
    transform: np.ndarray
    threshold: float = 0.8

    def applicable(self, state: np.ndarray) -> bool:
        return cosine(state, self.precondition) >= self.threshold

# ============================================================
# 6. PROPOSITIONAL CALCULUS (VECTOR SEMANTICS)
# ============================================================

# Modus Ponens: (P → Q) ∧ P ⟹ Q
modus_ponens = LogicRule(
    name="Modus Ponens",
    precondition=bundle([
        bind(P, IMPLIES),
        P
    ]),
    transform=Q,
    threshold=0.85
)

# Modus Tollens: (P → Q) ∧ ¬Q ⟹ ¬P
modus_tollens = LogicRule(
    name="Modus Tollens",
    precondition=bundle([
        bind(P, IMPLIES),
        negate(Q)
    ]),
    transform=negate(P),
    threshold=0.85
)

# Conjunction Introduction: P ∧ Q
and_intro = LogicRule(
    name="And Introduction",
    precondition=bundle([P, Q]),
    transform=bundle([
        bind(P, AND),
        bind(Q, AND)
    ]),
    threshold=0.8
)

# Conjunction Elimination (Left): (P ∧ Q) ⟹ P
and_elim_left = LogicRule(
    name="And Elimination Left",
    precondition=bind(P, AND),
    transform=P,
    threshold=0.8
)

RULES = [
    modus_ponens,
    modus_tollens,
    and_intro,
    and_elim_left
]

# ============================================================
# 7. NON-MONOTONIC INFERENCE ENGINE
# ============================================================

def infer(state: np.ndarray, rules: List[LogicRule]) -> np.ndarray:
    current = state.copy()

    for _ in range(MAX_INFERENCE_STEPS):
        updated = False

        for rule in rules:
            if rule.applicable(current):
                proposal = rule.transform
                c = contradiction_strength(current, proposal)

                if c < CONTRADICTION_THRESHOLD:
                    current = bundle([current, proposal])
                    updated = True

        if not updated:
            break

    return normalize(current)

# ============================================================
# 8. APPROXIMATE UNBINDING
# ============================================================

def unbind(bound: np.ndarray, role: np.ndarray) -> np.ndarray:
    """Approximate inverse of fractional binding"""
    return normalize(bound * role)

# ============================================================
# 9. DEMONSTRATION / SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("GodNode – Paraconsistent Vector Logic Engine (Final)\n")

    # Knowledge base: P, (P → Q), ¬Q
    state = bundle([
        P,
        bind(P, IMPLIES),
        negate(Q)
    ])

    print("Initial state:")
    print("cos(state, P)   =", cosine(state, P))
    print("cos(state, Q)   =", cosine(state, Q))
    print("cos(state, ¬Q)  =", cosine(state, negate(Q)))

    final_state = infer(state, RULES)

    print("\nAfter inference:")
    print("cos(final, P)   =", cosine(final_state, P))
    print("cos(final, ¬P)  =", cosine(final_state, negate(P)))
    print("cos(final, Q)   =", cosine(final_state, Q))

    print("\nExplosion check:")
    print("contradiction(Q) =",
          contradiction_strength(final_state, Q))
