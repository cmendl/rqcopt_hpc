import numpy as np
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


def brickwall_unitary_backward_hessian_data():

    # random number generator
    rng = np.random.default_rng(47)

    # system size
    L = 6

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_brickwall_unitary_backward_hessian_{ctype}.hdf5", "w")

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

        # gradient direction of quantum gates
        Zlist = [1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(max_nlayers)]
        for i in range(max_nlayers):
            file[f"Z{i}"] = interleave_complex(Zlist[i], ctype)

        file.close()


def main():
    apply_brickwall_unitary_data()
    brickwall_unitary_backward_data()
    brickwall_unitary_backward_hessian_data()


if __name__ == "__main__":
    main()
