import numpy as np
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def transpose_statevector_data():

    L = 9

    # random number generator
    rng = np.random.default_rng(42)

    for ctype in ["real", "cplx"]:
        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)

        # random permutation
        perm = rng.permutation(L)

        chi = psi.reshape(L * (2,)).transpose(perm).reshape(-1)

        # save to disk
        with h5py.File(f"data/test_transpose_statevector_{ctype}.hdf5", "w") as file:
            file["psi"]  = interleave_complex(psi, ctype)
            file["perm"] = perm
            file["chi"]  = interleave_complex(chi, ctype)


def main():
    transpose_statevector_data()


if __name__ == "__main__":
    main()
