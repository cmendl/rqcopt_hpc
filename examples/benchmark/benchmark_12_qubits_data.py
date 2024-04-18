import numpy as np
from scipy.stats import ortho_group, unitary_group
import h5py
from io_util import interleave_complex


def benchmark_12_qubits_data():

    # random number generator
    rng = np.random.default_rng(387)

    # system size
    nqubits = 12
    # number of layers
    nlayers = 8

    ctype = "cplx"

    # random unitary gates
    vlist = [ortho_group.rvs(4, random_state=rng) if ctype == "real" else unitary_group.rvs(4, random_state=rng) for _ in range(nlayers)]
    # permutations (wires the gates act on)
    perms = [range(nqubits) if i % 2 == 0 else np.roll(range(nqubits), 1) for i in range(nlayers)]

    # save data to disk
    with h5py.File(f"benchmark_{nqubits}_qubits_data.hdf5", "w") as file:
        file["vlist"] = interleave_complex(np.stack(vlist), "cplx")
        for i in range(nlayers):
            file[f"perm{i}"] = perms[i]


if __name__ == "__main__":
    benchmark_12_qubits_data()
