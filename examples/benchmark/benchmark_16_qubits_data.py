import numpy as np
from scipy.stats import ortho_group, unitary_group
import h5py
from io_util import interleave_complex


def benchmark_16_qubits_data():

    # random number generator
    rng = np.random.default_rng(528)

    # system size
    nqubits = 16
    # number of layers
    nlayers = 8

    ctype = "cplx"

    # random unitary gates
    vlist = [ortho_group.rvs(4, random_state=rng) if ctype == "real" else unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    # permutations (wires the gates act on)
    perms = [range(nqubits) if i % 2 == 0 else np.roll(range(nqubits), 1) for i in range(nlayers)]

    # random input statevector
    psi = rng.standard_normal(2**nqubits) if ctype == "real" else crandn(2**nqubits, rng)
    psi /= np.linalg.norm(psi)
    # fictitious upstream gradient vector
    dpsi_out = rng.standard_normal(2**nqubits) if ctype == "real" else crandn(2**nqubits, rng)

    # save data to disk
    with h5py.File(f"benchmark_{nqubits}_qubits_data.hdf5", "w") as file:
        file["vlist"] = interleave_complex(np.stack(vlist), "cplx")
        for i in range(nlayers):
            file[f"perm{i}"] = perms[i]
        file["psi"] = interleave_complex(psi, ctype)
        file["dpsi_out"] = interleave_complex(dpsi_out, ctype)


def crandn(size, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None: rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


if __name__ == "__main__":
    benchmark_16_qubits_data()
