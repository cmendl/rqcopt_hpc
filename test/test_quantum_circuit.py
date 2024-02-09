import numpy as np
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_quantum_circuit_data():

    # random number generator
    rng = np.random.default_rng(285)

    # system size
    nqubits = 8
    # number of gates
    ngates  = 4

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_quantum_circuit_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**nqubits) if ctype == "real" else oc.crandn(2**nqubits, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # general random 4x4 matrices (do not need to be unitary for this test)
        gates = [0.5 * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(ngates)]
        for i in range(ngates):
            file[f"G{i}"] = interleave_complex(gates[i], ctype)

        # random wires which the gates act on
        wires = np.array([rng.choice(nqubits, 2, replace=False) for _ in range(ngates)])
        file["wires"] = wires

        psi_out = psi
        for i in range(ngates):
            psi_out = oc.apply_gate(gates[i], nqubits, wires[i][0], wires[i][1], psi_out)
        file["psi_out"] = interleave_complex(psi_out, ctype)

        file.close()


def quantum_circuit_backward_data():

    # random number generator
    rng = np.random.default_rng(913)

    # system size
    nqubits = 6
    # number of gates
    ngates  = 5

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_quantum_circuit_backward_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**nqubits) if ctype == "real" else oc.crandn(2**nqubits, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # general random 4x4 matrices (do not need to be unitary for this test)
        gates = [1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng) for _ in range(ngates)]
        for i in range(ngates):
            file[f"G{i}"] = interleave_complex(gates[i], ctype)

        # random wires which the gates act on
        wires = np.array([rng.choice(nqubits, 2, replace=False) for _ in range(ngates)])
        file["wires"] = wires

        psi_out = psi
        for i in range(ngates):
            psi_out = oc.apply_gate(gates[i], nqubits, wires[i][0], wires[i][1], psi_out)
        file["psi_out"] = interleave_complex(psi_out, ctype)

        # fictitious upstream derivatives
        dphi = rng.standard_normal(2**nqubits) if ctype == "real" else oc.crandn(2**nqubits, rng)
        file["dphi"] = interleave_complex(dphi, ctype)

        file.close()


def main():
    apply_quantum_circuit_data()
    quantum_circuit_backward_data()


if __name__ == "__main__":
    main()
