import numpy as np


def circuit_unitary(nqubits, gates, wires):
    """
    Construct the overall unitary matrix representation of a quantum circuit.
    """
    v = np.identity(2**nqubits)
    for gate, wire in zip(gates, wires):
        v = circuit_gate(gate, nqubits, wire[0], wire[1]) @ v
    return v


def circuit_gate(gate, nqubits: int, i: int, j: int):
    """
    Construct the circuit matrix representation of a two-qubit quantum gate acting on qubits (i, j).
    """
    assert 0 <= i < nqubits
    assert 0 <= j < nqubits
    assert i != j
    if i > j:
        gate = gate.reshape((2, 2, 2, 2)).transpose((1, 0, 3, 2)).reshape((4, 4))
        return circuit_gate(gate, nqubits, j, i)
    u = np.kron(gate, np.identity(2**(nqubits - 2)))
    perm = list(range(2, i + 2)) + [0] + list(range(i + 2, j + 1)) + [1] + list(range(j + 1, nqubits))
    return permute_operation(u, perm)


def permute_operation(u: np.ndarray, perm):
    """
    Find the representation of a matrix after permuting lattice sites.
    """
    nsites = len(perm)
    assert u.shape == (2**nsites, 2**nsites)
    perm = list(perm)
    u = np.reshape(u, (2*nsites) * (2,))
    u = np.transpose(u, perm + [nsites + p for p in perm])
    u = np.reshape(u, (2**nsites, 2**nsites))
    return u
