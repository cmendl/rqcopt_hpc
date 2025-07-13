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
        assert file.attrs["nqubits"] == L
    # data based on C implementation and sampling
    with h5py.File(f"ising1d_dynamics_opt_sampling_n{nlayers}.hdf5", "r") as file:
        f_iter_sampling = file["f_iter"][:]
        Vlist_sampling  = decode_complex(file["Vlist"][:], "cplx")
        assert file.attrs["nqubits"] == L

    # check unitary property
    print("all unitary:", all(np.allclose(V.conj().T @ V, np.identity(4), atol=1e-14, rtol=1e-14) for V in Vlist))
    print("all unitary (sampling):", all(np.allclose(V.conj().T @ V, np.identity(4), atol=1e-14, rtol=1e-14) for V in Vlist_sampling))

    print("target function:")
    print(f_iter)
    print("target function difference:")
    print(f_iter - f_iter_ref)

    print("target function (sampling):")
    print(f_iter_sampling)
    print("target function (sampling) difference:")
    print(f_iter_sampling - f_iter_ref)

    print("deviation of optimized gates:")
    print([np.linalg.norm(V - Vref) for V, Vref in zip(Vlist, Vlist_ref)])

    print("deviation of optimized gates (sampling):")
    print([np.linalg.norm(V - Vref) for V, Vref in zip(Vlist_sampling, Vlist_ref)])


if __name__ == "__main__":
    main()
