import numpy as np
import rqcopt_matfree as oc


def apply_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 8

    # random input statevector
    psi = rng.standard_normal(2**L)
    psi /= np.linalg.norm(psi)
    psi.tofile("data/test_apply_brickwall_unitary_psi.dat")

    max_nlayers = 4

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = [0.5 * rng.standard_normal((4, 4)) for _ in range(max_nlayers)]
    for i in range(max_nlayers):
        Vlist[i].tofile(f"data/test_apply_brickwall_unitary_V{i}.dat")

    # random permutations
    perms = [rng.permutation(L) for _ in range(max_nlayers)]
    for i in range(max_nlayers):
        perms[i].tofile(f"data/test_apply_brickwall_unitary_perm{i}.dat")

    for i, nlayers in enumerate([3, 4]):
        chi = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
        chi.tofile(f"data/test_apply_brickwall_unitary_chi{i}.dat")


def apply_adjoint_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 8

    # random input statevector
    psi = rng.standard_normal(2**L)
    psi /= np.linalg.norm(psi)
    psi.tofile("data/test_apply_adjoint_brickwall_unitary_psi.dat")

    max_nlayers = 4

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = [0.5 * rng.standard_normal((4, 4)) for _ in range(max_nlayers)]
    for i in range(max_nlayers):
        Vlist[i].tofile(f"data/test_apply_adjoint_brickwall_unitary_V{i}.dat")

    # random permutations
    perms = [rng.permutation(L) for _ in range(max_nlayers)]
    for i in range(max_nlayers):
        perms[i].tofile(f"data/test_apply_adjoint_brickwall_unitary_perm{i}.dat")

    for i, nlayers in enumerate([3, 4]):
        chi = oc.apply_adjoint_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
        chi.tofile(f"data/test_apply_adjoint_brickwall_unitary_chi{i}.dat")


def _Ufunc(x):
    n = len(x)
    return np.array([- 1.1 * x[((i + 3) * 113) % n]
                     - 0.7 * x[((i + 9) * 173) % n]
                     + 0.5 * x[i]
                     + 0.3 * x[((i + 4) * 199) % n] for i in range(n)])


def brickwall_unitary_grad_matfree_data():

    # random number generator
    rng = np.random.default_rng(44)

    # system size
    L = 8

    max_nlayers = 5

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = np.stack([0.5 * rng.standard_normal((4, 4)) for _ in range(max_nlayers)])
    for i in range(max_nlayers):
        Vlist[i].tofile(f"data/test_brickwall_unitary_grad_matfree_V{i}.dat")

    # random permutations
    perms = [rng.permutation(L) for _ in range(max_nlayers)]
    for i in range(max_nlayers):
        perms[i].tofile(f"data/test_brickwall_unitary_grad_matfree_perm{i}.dat")

    for i, nlayers in enumerate([1, 4, 5]):
        dVlist = oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, _Ufunc, perms[:nlayers])
        dVlist.tofile(f"data/test_brickwall_unitary_grad_matfree_dVlist{i}.dat")


def main():
    apply_brickwall_unitary_data()
    apply_adjoint_brickwall_unitary_data()
    brickwall_unitary_grad_matfree_data()


if __name__ == "__main__":
    main()
