import numpy as np


def apply_gate(psi, U, idx, nqubits):
    """
    Apply the (single or multi)-qubit gate `U` to the qubits `idx`.
    """
    assert np.size(psi) == 2**nqubits
    # convert a single integer to a tuple storing this integer
    if isinstance(idx, int): idx = (idx,)
    assert U.shape == (2**len(idx), 2**len(idx))
    if len(idx) == 1:
        # single-qubit gate
        i = idx[0]
        assert 0 <= i < nqubits
        psi = np.reshape(psi, (2**i, 2, 2**(nqubits-i-1)))
        psi = np.einsum(U, (1, 2), psi, (0, 2, 3), (0, 1, 3))
        psi = np.reshape(psi, -1)
        return psi
    elif len(idx) == 2:
        # two-qubit gate
        i, j = idx  # unpack indices
        assert 0 <= i < j < nqubits
        U = np.reshape(U, (2, 2, 2, 2))
        psi = np.reshape(psi, (2**i, 2, 2**(j-i-1), 2, 2**(nqubits-j-1)))
        psi = np.einsum(U, (1, 5, 2, 4), psi, (0, 2, 3, 4, 6), (0, 1, 3, 5, 6))
        psi = np.reshape(psi, -1)
        return psi
    elif len(idx) == 3:
        # three-qubit gate
        i, j, k = idx  # unpack indices
        assert 0 <= i < j < k < nqubits
        U = np.reshape(U, (2, 2, 2, 2, 2, 2))
        psi = np.reshape(psi, (2**i, 2, 2**(j-i-1), 2, 2**(k-j-1), 2, 2**(nqubits-k-1)))
        psi = np.einsum(U, (1, 4, 8, 2, 5, 7), psi, (0, 2, 3, 5, 6, 7, 9), (0, 1, 3, 4, 6, 8, 9))
        psi = np.reshape(psi, -1)
        return psi
    elif len(idx) == 4:
        # four-qubit gate
        i, j, k, l = idx  # unpack indices
        assert 0 <= i < j < k < l < nqubits
        U = np.reshape(U, (2, 2, 2, 2, 2, 2, 2, 2))
        psi = np.reshape(psi, (2**i, 2, 2**(j-i-1), 2, 2**(k-j-1), 2, 2**(l-k-1), 2, 2**(nqubits-l-1)))
        psi = np.einsum(U, (1, 4, 8, 11, 2, 5, 7, 10), psi, (0, 2, 3, 5, 6, 7, 9, 10, 12), (0, 1, 3, 4, 6, 8, 9, 11, 12))
        psi = np.reshape(psi, -1)
        return psi
    else:
        raise RuntimeError("currently only up to 4-qubit gates supported")
