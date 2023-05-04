import numpy as np
from apply_gate import apply_gate


def main():

    nqubits = 9

    # random number generator
    rng = np.random.default_rng(42)

    # general random 2x2 matrix (does not need to be unitary for this test)
    U = rng.standard_normal((2, 2))
    U.tofile("test_apply_gate_U.dat")

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("test_apply_gate_V.dat")

    # random input statevector
    psi = rng.standard_normal(2**nqubits)
    psi /= np.linalg.norm(psi)
    psi.tofile("test_apply_gate_psi.dat")

    psi1 = apply_gate(psi, U, 3, nqubits)
    psi1.tofile("test_apply_gate_psi1.dat")

    psi2 = apply_gate(psi, V, (2, 5), nqubits)
    psi2.tofile("test_apply_gate_psi2.dat")


if __name__ == "__main__":
    main()
