import numpy as np
from scipy.stats import unitary_group
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 8

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_brickwall_unitary_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        max_nlayers = 4

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = [0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([3, 4]):
            chi = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"chi{i}"] = interleave_complex(chi, ctype)

        file.close()


def apply_adjoint_brickwall_unitary_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 8

    max_nlayers = 4

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_adjoint_brickwall_unitary_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = [0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([3, 4]):
            chi = oc.apply_adjoint_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"chi{i}"] = interleave_complex(chi, ctype)

        file.close()


def brickwall_unitary_backward_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 6

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_brickwall_unitary_backward_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        max_nlayers = 4

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = [1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([3, 4]):
            psi_out = oc.apply_brickwall_unitary(Vlist[:nlayers], L, psi, perms[:nlayers])
            file[f"psi_out{i}"] = interleave_complex(psi_out, ctype)

        # fictitious upstream derivatives
        dpsi_out = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        file["dpsi_out"] = interleave_complex(dpsi_out, ctype)

        file.close()


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
        file = h5py.File(f"data/test_brickwall_unitary_grad_matfree_{ctype}.hdf5", "w")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)])
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        for i, nlayers in enumerate([1, 4, 5]):
            dVlist = oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, _Ufunc, perms[:nlayers])
            file[f"dVlist{i}"] = interleave_complex(dVlist, ctype)

        file.close()


def brickwall_unitary_gradient_vector_matfree_data():

    # random number generator
    rng = np.random.default_rng(45)

    # system size
    L = 6

    # number of layers
    nlayers = 3

    ctype = "cplx"

    file = h5py.File(f"data/test_brickwall_unitary_gradient_vector_matfree_{ctype}.hdf5", "w")

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = np.stack([0.5 * oc.crandn((4, 4), rng) for _ in range(nlayers)])
    for i in range(nlayers):
        file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

    # random permutations
    perms = [rng.permutation(L) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"perm{i}"] = perms[i]

    grad = oc.brickwall_unitary_gradient_vector_matfree(Vlist, L, _Ufunc, perms)
    file["grad"] = grad

    file.close()


def brickwall_unitary_hess_matfree_data():

    # random number generator
    rng = np.random.default_rng(46)

    # system size
    L = 6

    nlayers = 4

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_brickwall_unitary_hess_matfree_{ctype}.hdf5", "w")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(nlayers)])
        for i in range(nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(nlayers)]
        for i in range(nlayers):
            file[f"perm{i}"] = perms[i]

        # gradient direction
        rZ = 0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng)
        file["rZ"] = interleave_complex(rZ, ctype)

        for k in range(nlayers):
            for i, Z in enumerate([rZ, oc.project_unitary_tangent(Vlist[k], rZ)]):
                for uproj in [False, True]:
                    dVlist = oc.brickwall_unitary_hess_matfree(Vlist, L, Z, k, _Ufunc, perms, unitary_proj=uproj)
                    ul = "proj" if uproj else ""
                    file[f"dVlist{k}{i}{ul}"] = interleave_complex(dVlist, ctype)

        file.close()


def brickwall_unitary_hessian_matrix_matfree_data():

    # random number generator
    rng = np.random.default_rng(47)

    # system size
    L = 6

    # number of layers
    nlayers = 5

    ctype = "cplx"

    file = h5py.File(f"data/test_brickwall_unitary_hessian_matrix_matfree_{ctype}.hdf5", "w")

    # random unitaries (unitary property required for Hessian matrix to be symmetric)
    Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

    # random permutations
    perms = [rng.permutation(L) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"perm{i}"] = perms[i]

    H = oc.brickwall_unitary_hessian_matrix_matfree(Vlist, L, _Ufunc, perms)
    file["H"] = H

    file.close()


def main():
    apply_brickwall_unitary_data()
    apply_adjoint_brickwall_unitary_data()
    brickwall_unitary_backward_data()
    brickwall_unitary_grad_matfree_data()
    brickwall_unitary_gradient_vector_matfree_data()
    brickwall_unitary_hess_matfree_data()
    brickwall_unitary_hessian_matrix_matfree_data()


if __name__ == "__main__":
    main()
