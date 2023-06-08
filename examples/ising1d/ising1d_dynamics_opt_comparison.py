import numpy as np
import h5py
from io_util import decode_complex


def main():

    # side length of lattice
    L = 6
    # number of layers
    nlayers = 3

    # reference data
    with h5py.File(f"ising1d_dynamics_opt_n{nlayers}_ref.hdf5", "r") as file:
        f_iter_ref = file["f_iter"][:]
        Vlist_ref  = decode_complex(file["Vlist"][:], "cplx")
        assert file.attrs["L"] == L
    # data based on C implementation
    with h5py.File(f"ising1d_dynamics_opt_n{nlayers}.hdf5", "r") as file:
        f_iter = file["f_iter"][:]
        Vlist  = decode_complex(file["Vlist"][:], "cplx")
        assert file.attrs["L"] == L

    # check unitary property
    print("all unitary:", all(np.allclose(V.conj().T @ V, np.identity(4), atol=1e-14, rtol=1e-14) for V in Vlist))

    print("target function:")
    print(f_iter)
    print("target function difference:")
    print(f_iter - f_iter_ref)

    print("deviation of optimized gates:")
    print([np.linalg.norm(V - Vref) for V, Vref in zip(Vlist, Vlist_ref)])


if __name__ == "__main__":
    main()
