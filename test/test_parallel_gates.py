import numpy as np
import rqcopt_matfree as oc


def apply_gate_data():

    L = 9

    # random number generator
    rng = np.random.default_rng(42)

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("test_apply_gate_V.dat")

    # random input statevector
    psi = rng.standard_normal(2**L)
    psi /= np.linalg.norm(psi)
    psi.tofile("test_apply_gate_psi.dat")

    # general i < j
    chi1 = oc.apply_gate(V, L, 2, 5, psi)
    chi1.tofile("test_apply_gate_chi1.dat")
    # j < i
    chi2 = oc.apply_gate(V, L, 4, 1, psi)
    chi2.tofile("test_apply_gate_chi2.dat")
    # j == i + 1
    chi3 = oc.apply_gate(V, L, 3, 4, psi)
    chi3.tofile("test_apply_gate_chi3.dat")


def main():
    apply_gate_data()


if __name__ == "__main__":
    main()
