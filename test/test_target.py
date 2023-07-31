import numpy as np
from scipy.stats import unitary_group
import h5py
from io_util import interleave_complex
import rqcopt_matfree as oc


def _ufunc_real(x):
    n = len(x)
    return np.array([- 1.1 * x[((i + 3) * 113) % n]
                     - 0.7 * x[((i + 9) * 173) % n]
                     + 0.5 * x[i]
                     + 0.3 * x[((i + 4) * 199) % n] for i in range(n)])

def _ufunc_cplx(x):
    n = len(x)
    return np.array([  (-1.1 + 0.8j) * x[((i + 3) * 113) % n]
                     + ( 0.4 - 0.7j) * x[((i + 9) * 173) % n]
                     + ( 0.5 + 0.1j) * x[i]
                     + (-0.3 + 0.2j) * x[((i + 4) * 199) % n] for i in range(n)])


def unitary_target_data():

    # random number generator
    rng = np.random.default_rng(41)

    # system size
    L = 8

    max_nlayers = 5

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_unitary_target_{ctype}.hdf5", "w")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)])
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        ufunc = _ufunc_real if ctype == "real" else _ufunc_cplx

        for i, nlayers in enumerate([4, 5]):
            # target function value
            f = oc.brickwall_opt_matfree._f_unitary_target_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"f{i}"] = f

        file.close()


def unitary_target_and_gradient_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 8

    max_nlayers = 5

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_unitary_target_and_gradient_{ctype}.hdf5", "w")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)])
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        ufunc = _ufunc_real if ctype == "real" else _ufunc_cplx

        for i, nlayers in enumerate([4, 5]):
            # target function value
            f = oc.brickwall_opt_matfree._f_unitary_target_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"f{i}"] = f
            # gate gradients
            dVlist = -oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"dVlist{i}"] = interleave_complex(dVlist.conj(), ctype)   # complex conjugation due to different convention

        file.close()


def unitary_target_and_gradient_vector_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 6

    # number of layers
    nlayers = 3

    ctype = "cplx"

    file = h5py.File(f"data/test_unitary_target_and_gradient_vector_{ctype}.hdf5", "w")

    # general random 4x4 matrices (do not need to be unitary for this test)
    Vlist = np.stack([0.5 * oc.crandn((4, 4), rng) for _ in range(nlayers)])
    for i in range(nlayers):
        file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

    # random permutations
    perms = [rng.permutation(L) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"perm{i}"] = perms[i]

    ufunc = _ufunc_real if ctype == "real" else _ufunc_cplx

    # target function value
    f = oc.brickwall_opt_matfree._f_unitary_target_matfree(Vlist, L, ufunc, perms)
    file["f"] = f
    # gate gradients as real vector
    grad = -oc.brickwall_unitary_gradient_vector_matfree(Vlist, L, ufunc, perms)
    file["grad"] = grad

    file.close()


def _brickwall_unitary_plain_hessian_matrix_matfree(Vlist, L, Ufunc, perms):
    """
    Construct the Hessian matrix of Re tr[U† W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16), dtype=Vlist[0].dtype)
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = Z.reshape((4, 4))
            dVZj = oc.brickwall_unitary_hess_matfree(Vlist, L, Z, j, Ufunc, perms, unitary_proj=False)
            for i in range(n):
                H[i, :, j, k] = dVZj[i].reshape(-1)
    return H.reshape((n * 16, n * 16))


def unitary_target_gradient_hessian_data():

    # random number generator
    rng = np.random.default_rng(45)

    # system size
    L = 6

    max_nlayers = 5

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_unitary_target_gradient_hessian_{ctype}.hdf5", "w")

        # general random 4x4 matrices (do not need to be unitary for this test)
        Vlist = np.stack([1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)])
        for i in range(max_nlayers):
            file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

        # random permutations
        perms = [rng.permutation(L) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"perm{i}"] = perms[i]

        ufunc = _ufunc_real if ctype == "real" else _ufunc_cplx

        for i, nlayers in enumerate([4, 5]):
            # target function value
            f = oc.brickwall_opt_matfree._f_unitary_target_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"f{i}"] = f
            # gate gradients
            dVlist = -oc.brickwall_unitary_grad_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"dVlist{i}"] = interleave_complex(dVlist.conj(), ctype)   # complex conjugation due to different convention
            # Hessian matrix
            hess = -_brickwall_unitary_plain_hessian_matrix_matfree(Vlist[:nlayers], L, ufunc, perms[:nlayers])
            file[f"hess{i}"] = interleave_complex(hess.conj(), ctype)       # complex conjugation due to different convention

        file.close()


def unitary_target_gradient_vector_hessian_matrix_data():

    # random number generator
    rng = np.random.default_rng(47)

    # system size
    L = 6

    # number of layers
    nlayers = 5

    ctype = "cplx"

    file = h5py.File(f"data/test_unitary_target_gradient_vector_hessian_matrix_{ctype}.hdf5", "w")

    # random unitaries (unitary property required for Hessian matrix to be symmetric)
    Vlist = [unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"V{i}"] = interleave_complex(Vlist[i], ctype)

    # random permutations
    perms = [rng.permutation(L) for _ in range(nlayers)]
    for i in range(nlayers):
        file[f"perm{i}"] = perms[i]

    ufunc = _ufunc_real if ctype == "real" else _ufunc_cplx

    # target function value
    f = oc.brickwall_opt_matfree._f_unitary_target_matfree(Vlist, L, ufunc, perms)
    file["f"] = f

    # gate gradients as real vector
    grad = -oc.brickwall_unitary_gradient_vector_matfree(Vlist, L, ufunc, perms)
    file["grad"] = grad

    # Hessian matrix
    H = -oc.brickwall_unitary_hessian_matrix_matfree(Vlist, L, ufunc, perms)
    file["H"] = H

    file.close()


def main():
    unitary_target_data()
    unitary_target_and_gradient_data()
    unitary_target_and_gradient_vector_data()
    unitary_target_gradient_hessian_data()
    unitary_target_gradient_vector_hessian_matrix_data()


if __name__ == "__main__":
    main()
