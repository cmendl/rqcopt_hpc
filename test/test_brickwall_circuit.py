import numpy as np
import rqcopt_matfree as oc


def apply_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 8

    for ctype in ["real", "cplx"]:
        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"data/test_apply_brickwall_unitary_{ctype}_psi.dat")

        max_nlayers = 4

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = [0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            Vlist[i].tofile(f"data/test_apply_brickwall_unitary_{ctype}_V{i}.dat")

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            perms[i].tofile(f"data/test_apply_brickwall_unitary_{ctype}_perm{i}.dat")

        for i, nlayers in enumerate([3, 4]):
            chi = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            chi.tofile(f"data/test_apply_brickwall_unitary_{ctype}_chi{i}.dat")


def apply_adjoint_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 8

    max_nlayers = 4

    for ctype in ["real", "cplx"]:
        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"data/test_apply_adjoint_brickwall_unitary_{ctype}_psi.dat")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = [0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            Vlist[i].tofile(f"data/test_apply_adjoint_brickwall_unitary_{ctype}_V{i}.dat")

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            perms[i].tofile(f"data/test_apply_adjoint_brickwall_unitary_{ctype}_perm{i}.dat")

        for i, nlayers in enumerate([3, 4]):
            chi = oc.apply_adjoint_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            chi.tofile(f"data/test_apply_adjoint_brickwall_unitary_{ctype}_chi{i}.dat")


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

    for ctype in ["real", "cplx"]:
        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)])
        for i in range(max_nlayers):
            Vlist[i].tofile(f"data/test_brickwall_unitary_grad_matfree_{ctype}_V{i}.dat")

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            perms[i].tofile(f"data/test_brickwall_unitary_grad_matfree_{ctype}_perm{i}.dat")

        for i, nlayers in enumerate([1, 4, 5]):
            dVlist = oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, _Ufunc, perms[:nlayers])
            dVlist.tofile(f"data/test_brickwall_unitary_grad_matfree_{ctype}_dVlist{i}.dat")


def brickwall_unitary_gradient_vector_matfree_data():

    # random number generator
    rng = np.random.default_rng(45)

    # system size
    L = 6

    # number of layers
    nlayers = 3

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = np.stack([0.5 * oc.crandn((4, 4), rng) for _ in range(nlayers)])
    for i in range(nlayers):
        Vlist[i].tofile(f"data/test_brickwall_unitary_gradient_vector_matfree_V{i}.dat")

    # random permutations
    perms = [rng.permutation(L) for _ in range(nlayers)]
    for i in range(nlayers):
        perms[i].tofile(f"data/test_brickwall_unitary_gradient_vector_matfree_perm{i}.dat")

    grad = oc.brickwall_unitary_gradient_vector_matfree(Vlist, L, _Ufunc, perms)
    grad.tofile("data/test_brickwall_unitary_gradient_vector_matfree_grad.dat")


def brickwall_unitary_hess_matfree_data():

    # random number generator
    rng = np.random.default_rng(44)

    # system size
    L = 6

    nlayers = 4

    for ctype in ["real", "cplx"]:
        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(nlayers)])
        for i in range(nlayers):
            Vlist[i].tofile(f"data/test_brickwall_unitary_hess_matfree_{ctype}_V{i}.dat")

        # random permutations
        perms = [rng.permutation(L) for _ in range(nlayers)]
        for i in range(nlayers):
            perms[i].tofile(f"data/test_brickwall_unitary_hess_matfree_{ctype}_perm{i}.dat")

        # gradient direction
        rZ = 0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng)
        rZ.tofile(f"data/test_brickwall_unitary_hess_matfree_{ctype}_rZ.dat")

        for k in range(nlayers):
            for i, Z in enumerate([rZ, oc.project_unitary_tangent(Vlist[k], rZ)]):
                for uproj in [False, True]:
                    dVlist = oc.brickwall_unitary_hess_matfree(Vlist, L, Z, k, _Ufunc, perms, unitary_proj=uproj)
                    ul = "proj" if uproj else ""
                    dVlist.tofile(f"data/test_brickwall_unitary_hess_matfree_{ctype}_dVlist{k}{i}{ul}.dat")


def main():
    apply_brickwall_unitary_data()
    apply_adjoint_brickwall_unitary_data()
    brickwall_unitary_grad_matfree_data()
    brickwall_unitary_gradient_vector_matfree_data()
    brickwall_unitary_hess_matfree_data()


if __name__ == "__main__":
    main()
