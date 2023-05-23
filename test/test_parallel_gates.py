import numpy as np
from scipy.stats import ortho_group, unitary_group
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_gate_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 9

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_gate_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # general i < j
        chi1 = oc.apply_gate(V, L, 2, 5, psi)
        # j < i
        chi2 = oc.apply_gate(V, L, 4, 1, psi)
        # j == i + 1
        chi3 = oc.apply_gate(V, L, 3, 4, psi)

        file["chi1"] = interleave_complex(chi1, ctype)
        file["chi2"] = interleave_complex(chi2, ctype)
        file["chi3"] = interleave_complex(chi3, ctype)

        file.close()


def apply_parallel_gates_data():

    # random number generator
    rng = np.random.default_rng(43)

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_parallel_gates_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        for i, L in enumerate([6, 8, 10]):
            # random input statevector
            psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
            psi /= np.linalg.norm(psi)
            file[f"psi{i}"] = interleave_complex(psi, ctype)

            # random permutation
            perm = rng.permutation(L)
            file[f"perm{i}"] = perm

            # apply parallel gates
            chi = oc.apply_parallel_gates(V, L, psi, perm)
            file[f"chi{i}"] = interleave_complex(chi, ctype)

        file.close()


def apply_parallel_gates_directed_grad_data():

    # random number generator
    rng = np.random.default_rng(44)

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_parallel_gates_directed_grad_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # general random 4x4 gradient direction
        Z = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["Z"] = interleave_complex(Z, ctype)

        for i, L in enumerate([6, 8, 10]):
            # random input statevector
            psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
            psi /= np.linalg.norm(psi)
            file[f"psi{i}"] = interleave_complex(psi, ctype)

            # random permutation
            perm = rng.permutation(L)
            file[f"perm{i}"] = perm

            # apply parallel gates
            chi = oc.apply_parallel_gates_directed_grad(V, L, Z, psi, perm)
            file[f"chi{i}"]  = interleave_complex(chi, ctype)

        file.close()


def _Ufunc(x):
    n = len(x)
    return np.array([  0.7 * x[((i + 5) *  83) % n]
                     - 0.2 * x[i]
                     + 1.3 * x[((i + 1) * 181) % n]
                     + 0.4 * x[((i + 7) * 197) % n] for i in range(n)])


def parallel_gates_grad_matfree_data():

    # random number generator
    rng = np.random.default_rng(45)

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_parallel_gates_grad_matfree_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # random permutation
        perm = rng.permutation(8)
        file["perm"] = perm

        for L in [2, 8]:
            for i in range(2):
                dV = oc.parallel_gates_grad_matfree(V, L, _Ufunc, None if i == 0 else ([1, 0] if L == 2 else perm))
                file[f"dV{i}L{L}"] = interleave_complex(dV, ctype)

        file.close()


def parallel_gates_hess_matfree_data():

    # random number generator
    rng = np.random.default_rng(46)

    # system size
    L = 8

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_parallel_gates_hess_matfree_{ctype}.hdf5", "w")

        # random unitary
        V = ortho_group.rvs(4, random_state=rng) if ctype == "real" else unitary_group.rvs(4, random_state=rng)
        file["V"] = interleave_complex(V, ctype)

        # random permutation
        perm = rng.permutation(L)
        file["perm"] = perm

        # gradient direction
        rZ = 0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng)
        for i, Z in enumerate([rZ, oc.project_unitary_tangent(V, rZ)]):
            file[f"Z{i}"] = interleave_complex(Z, ctype)
            for uproj in [False, True]:
                dV = oc.parallel_gates_hess_matfree(V, L, Z, _Ufunc, perm, unitary_proj=uproj)
                ul = "proj" if uproj else ""
                file[f"dV{i}{ul}"] = interleave_complex(dV, ctype)

        file.close()


def main():
    apply_gate_data()
    apply_parallel_gates_data()
    apply_parallel_gates_directed_grad_data()
    parallel_gates_grad_matfree_data()
    parallel_gates_hess_matfree_data()


if __name__ == "__main__":
    main()
