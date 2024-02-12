import numpy as np
from .quantum_circuit_matfree import apply_quantum_circuit


def _f_circuit_unitary_target_matfree(gates, wires, nqubits: int, ufunc):
    """
    Evaluate target function -Re tr[U^{\dagger} C],
    using the provided matrix-free application of U to a state.
    """
    f = 0
    # implement trace via summation over unit vectors
    for b in range(2**nqubits):
        psi = np.zeros(2**nqubits)
        psi[b] = 1
        Upsi = ufunc(psi)
        Vpsi = apply_quantum_circuit(gates, wires, nqubits, psi)
        f -= np.vdot(Vpsi, Upsi).real
    return f
