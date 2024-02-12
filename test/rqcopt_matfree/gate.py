import numpy as np


def apply_gate(V, L: int, i: int, j: int, psi):
    """
    Apply the two-qubit gate V to (i, j)-th qubit of state psi.
    """
    assert 0 <= i < L
    assert 0 <= j < L
    assert i != j
    V = np.reshape(V, (2, 2, 2, 2))
    if i > j:
        return apply_gate(np.transpose(V, (1, 0, 3, 2)), L, j, i, psi)
    psi = np.reshape(psi, (2**i, 2, 2**(j - i - 1), 2, 2**(L - j - 1)))
    # apply V
    psi = np.einsum(V, (1, 5, 2, 4), psi, (0, 2, 3, 4, 6), (0, 1, 3, 5, 6))
    return np.reshape(psi, -1)
