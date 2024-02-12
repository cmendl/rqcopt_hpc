from .gate import apply_gate


def apply_quantum_circuit(gates, wires, nqubits: int, psi):
    """
    Apply a quantum circuit consisting of two-qubit gates to state psi.
    """
    psi_out = psi
    for i in range(len(gates)):
        psi_out = apply_gate(gates[i], nqubits, wires[i][0], wires[i][1], psi_out)
    return psi_out
