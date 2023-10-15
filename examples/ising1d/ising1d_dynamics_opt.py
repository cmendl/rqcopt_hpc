import sys
import time
import numpy as np
import scipy
import qib
import h5py
# compiled Python module; ensure that it is on the Python search path
import rqcopt_hpc as oc_hpc
sys.path.append("../../test/")
import rqcopt_matfree as oc
from io_util import decode_complex
import matplotlib.pyplot as plt


def construct_ising_local_term(J, g):
    """
    Construct local interaction term of Ising Hamiltonian on a one-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))


def ising1d_dynamics_opt(nlayers: int, bootstrap: bool, coeffs_start=None, niter=20):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the time evolution governed by an Ising Hamiltonian.
    """
    # side length of lattice
    L = 6
    # Hamiltonian parameters
    J = 1
    g = 0.75

    # construct Hamiltonian
    latt = qib.lattice.IntegerLattice((L,), pbc=True)
    field = qib.field.Field(qib.field.ParticleType.QUBIT, latt)
    H = qib.IsingHamiltonian(field, J, 0., g).as_matrix()

    # time
    t = 1.0

    # reference global unitary
    expiH = scipy.linalg.expm(-1j*H.todense()*t)

    # unitaries used as starting point for optimization
    if bootstrap:
        # load optimized unitaries for nlayers - 2 from disk
        with h5py.File(f"ising1d_dynamics_opt_n{nlayers-2}_ref.hdf5", "r") as f:
            # parameters must agree
            assert f.attrs["L"] == L
            assert f.attrs["J"] == J
            assert f.attrs["g"] == g
            assert f.attrs["t"] == t
            Vlist_start = decode_complex(f["Vlist"][:], "cplx")
            assert Vlist_start.shape[0] == nlayers - 2
        # pad identity matrices
        id4 = np.identity(4).reshape((1, 4, 4))
        Vlist_start = np.concatenate((id4, Vlist_start, id4), axis=0)
        assert Vlist_start.shape[0] == nlayers
        # convention for permutations used in C code
        perms = [np.arange(L) if i % 2 == 1 else np.roll(range(L), 1) for i in range(len(Vlist_start))]
    else:
        # local Hamiltonian term
        hloc = construct_ising_local_term(J, g)
        assert len(coeffs_start) == nlayers
        Vlist_start = [scipy.linalg.expm(-1j*c*t*hloc) for c in coeffs_start]
        # convention for permutations used in C code
        perms = [np.arange(L) if i % 2 == 0 else np.roll(range(L), 1) for i in range(len(Vlist_start))]
    # perform optimization
    t_start = time.perf_counter()
    Vlist, f_iter = oc_hpc.optimize_brickwall_circuit(L, expiH, Vlist_start, perms, niter)
    print(f"Completed optimization in {time.perf_counter() - t_start:g} seconds")

    # visualize optimization progress
    # rescaled and shifted target function
    print(f"f_iter before: {f_iter[0]}")
    print(f"f_iter after {niter} iterations: {f_iter[-1]}")
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**L)
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(\mathrm{Vlist})/2^L$")
    plt.title(f"optimization target function for a quantum circuit with {len(Vlist)} layers")
    plt.show()

    # reference data
    with h5py.File(f"ising1d_dynamics_opt_n{nlayers}_ref.hdf5", "r") as file:
        f_iter_ref = file["f_iter"][:]
        Vlist_ref  = decode_complex(file["Vlist"][:], "cplx")
        assert file.attrs["L"] == L
    # compare
    print("target function difference:")
    print(f_iter - f_iter_ref)
    print("deviation of optimized gates:")
    print([np.linalg.norm(V - Vref) for V, Vref in zip(Vlist, Vlist_ref)])


def main():

    # 3 layers
    # use a single Strang splitting step as starting point for optimization
    strang = oc.SplittingMethod.suzuki(2, 1)
    ising1d_dynamics_opt(3, False, strang.coeffs, niter=10)

    # # 5 layers
    # # use two Strang splitting steps as starting point for optimization
    # strang = oc.SplittingMethod.suzuki(2, 1)
    # _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    # # divide by 2 since we are taking two steps
    # coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
    # print("coeffs_start_n5:", coeffs_start_n5)
    # ising1d_dynamics_opt(5, False, coeffs_start_n5, niter=16)

    # # 7 layers
    # ising1d_dynamics_opt(7, True, niter=200)


if __name__ == "__main__":
    main()
