import numpy as np


def transpose_statevector_data():

    L = 9

    # random number generator
    rng = np.random.default_rng(42)

    # random input statevector
    psi = rng.standard_normal(2**L)
    psi /= np.linalg.norm(psi)
    psi.tofile("data/test_transpose_statevector_psi.dat")

    # random permutation
    perm = rng.permutation(L)
    perm.tofile("data/test_transpose_statevector_perm.dat")

    chi = psi.reshape(L * (2,)).transpose(perm).reshape(-1)
    chi.tofile("data/test_transpose_statevector_chi.dat")


def main():
    transpose_statevector_data()


if __name__ == "__main__":
    main()
