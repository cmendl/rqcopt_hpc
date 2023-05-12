import numpy as np
from scipy.stats import ortho_group
import rqcopt_matfree as oc


def apply_gate_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 9

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("data/test_apply_gate_V.dat")

    # random input statevector
    psi = rng.standard_normal(2**L)
    psi /= np.linalg.norm(psi)
    psi.tofile("data/test_apply_gate_psi.dat")

    # general i < j
    chi1 = oc.apply_gate(V, L, 2, 5, psi)
    chi1.tofile("data/test_apply_gate_chi1.dat")
    # j < i
    chi2 = oc.apply_gate(V, L, 4, 1, psi)
    chi2.tofile("data/test_apply_gate_chi2.dat")
    # j == i + 1
    chi3 = oc.apply_gate(V, L, 3, 4, psi)
    chi3.tofile("data/test_apply_gate_chi3.dat")


def apply_parallel_gates_data():

    # random number generator
    rng = np.random.default_rng(43)

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("data/test_apply_parallel_gates_V.dat")

    for i, L in enumerate([6, 8, 10]):
        # random input statevector
        psi = rng.standard_normal(2**L)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"data/test_apply_parallel_gates_psi{i}.dat")
        # random permutation
        perm = rng.permutation(L)
        perm.tofile(f"data/test_apply_parallel_gates_perm{i}.dat")
        # apply parallel gates
        chi = oc.apply_parallel_gates(V, L, psi, perm)
        chi.tofile(f"data/test_apply_parallel_gates_chi{i}.dat")


def apply_parallel_gates_directed_grad_data():

    # random number generator
    rng = np.random.default_rng(44)

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("data/test_apply_parallel_gates_directed_grad_V.dat")
    # general random 4x4 gradient direction
    Z = rng.standard_normal((4, 4))
    Z.tofile("data/test_apply_parallel_gates_directed_grad_Z.dat")

    for i, L in enumerate([6, 8, 10]):
        # random input statevector
        psi = rng.standard_normal(2**L)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"data/test_apply_parallel_gates_directed_grad_psi{i}.dat")
        # random permutation
        perm = rng.permutation(L)
        perm.tofile(f"data/test_apply_parallel_gates_directed_grad_perm{i}.dat")
        # apply parallel gates
        chi = oc.apply_parallel_gates_directed_grad(V, L, Z, psi, perm)
        chi.tofile(f"data/test_apply_parallel_gates_directed_grad_chi{i}.dat")


def _Ufunc(x):
    n = len(x)
    return np.array([  0.7 * x[((i + 5) *  83) % n]
                     - 0.2 * x[i]
                     + 1.3 * x[((i + 1) * 181) % n]
                     + 0.4 * x[((i + 7) * 197) % n] for i in range(n)])


def parallel_gates_grad_matfree_data():

    # random number generator
    rng = np.random.default_rng(45)

    # general random 4x4 matrix (does not need to be unitary for this test)
    V = rng.standard_normal((4, 4))
    V.tofile("data/test_parallel_gates_grad_matfree_V.dat")

    # random permutation
    perm = rng.permutation(8)
    perm.tofile("data/test_parallel_gates_grad_matfree_perm.dat")

    for L in [2, 8]:
        for i in range(2):
            dV = oc.parallel_gates_grad_matfree(V, L, _Ufunc, None if i == 0 else ([1, 0] if L == 2 else perm))
            dV.tofile(f"data/test_parallel_gates_grad_matfree_dV{i}L{L}.dat")


def parallel_gates_hess_matfree_data():

    # random number generator
    rng = np.random.default_rng(46)

    # system size
    L = 8

    # random unitary
    V = ortho_group.rvs(4, random_state=rng)
    V.tofile("data/test_parallel_gates_hess_matfree_V.dat")

    # random permutation
    perm = rng.permutation(L)
    perm.tofile("data/test_parallel_gates_hess_matfree_perm.dat")

    # gradient direction
    rZ = 0.5 * rng.standard_normal((4, 4))
    for i, Z in enumerate([rZ, oc.project_unitary_tangent(V, rZ)]):
        Z.tofile(f"data/test_parallel_gates_hess_matfree_Z{i}.dat")
        for uproj in [False, True]:
            dV = oc.parallel_gates_hess_matfree(V, L, Z, _Ufunc, perm, unitary_proj=uproj)
            ul = "proj" if uproj else ""
            dV.tofile(f"data/test_parallel_gates_hess_matfree_dV{i}{ul}.dat")


def main():
    apply_gate_data()
    apply_parallel_gates_data()
    apply_parallel_gates_directed_grad_data()
    parallel_gates_grad_matfree_data()
    parallel_gates_hess_matfree_data()


if __name__ == "__main__":
    main()
