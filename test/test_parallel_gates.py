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


def apply_parallel_gates_data():

    # random number generator
    rng = np.random.default_rng(43)

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("test_apply_parallel_gates_V.dat")

    for i, L in enumerate([6, 8, 10]):
        # random input statevector
        psi = rng.standard_normal(2**L)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"test_apply_parallel_gates_psi{i}.dat")
        # random permutation
        perm = rng.permutation(L)
        perm.tofile(f"test_apply_parallel_gates_perm{i}.dat")
        # apply parallel gates
        chi = oc.apply_parallel_gates(V, L, psi, perm)
        chi.tofile(f"test_apply_parallel_gates_chi{i}.dat")


def main():
    apply_gate_data()
    apply_parallel_gates_data()


if __name__ == "__main__":
    main()
