import numpy as np
import rqcopt_matfree as oc


def transpose_statevector_data():

    L = 9

    # random number generator
    rng = np.random.default_rng(42)

    for ctype in ["real", "cplx"]:
        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        psi.tofile(f"data/test_transpose_statevector_{ctype}_psi.dat")

        # random permutation
        perm = rng.permutation(L)
        perm.tofile(f"data/test_transpose_statevector_{ctype}_perm.dat")

        chi = psi.reshape(L * (2,)).transpose(perm).reshape(-1)
        chi.tofile(f"data/test_transpose_statevector_{ctype}_chi.dat")


def main():
    transpose_statevector_data()


if __name__ == "__main__":
    main()
