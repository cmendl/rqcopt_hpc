import numpy as np
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def apply_gate_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 9

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_gate_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # general i < j
        chi1 = oc.apply_gate(V, L, 2, 5, psi)
        # j < i
        chi2 = oc.apply_gate(V, L, 4, 1, psi)
        # j == i + 1
        chi3 = oc.apply_gate(V, L, 3, 4, psi)

        file["chi1"] = interleave_complex(chi1, ctype)
        file["chi2"] = interleave_complex(chi2, ctype)
        file["chi3"] = interleave_complex(chi3, ctype)

        file.close()


def apply_gate_backward_data():

    # random number generator
    rng = np.random.default_rng(43)

    # system size
    L = 9

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_gate_backward_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # fictitious upstream derivatives
        dpsi_out = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        dpsi_out /= np.linalg.norm(dpsi_out)
        file["dpsi_out"] = interleave_complex(dpsi_out, ctype)

        file.close()


def apply_gate_to_array_data():

    # random number generator
    rng = np.random.default_rng(42)

    # system size
    L = 6
    # number of states
    nstates = 5

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_gate_to_array_{ctype}.hdf5", "w")

        # general random 4x4 matrix (does not need to be unitary for this test)
        V = 1/np.sqrt(2) * rng.standard_normal((4, 4)) if ctype == "real" else 0.5 * oc.crandn((4, 4), rng)
        file["V"] = interleave_complex(V, ctype)

        # random input statevectors
        psi_list = []
        chi1_list = []
        chi2_list = []
        chi3_list = []
        for _ in range(nstates):
            psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
            psi /= np.linalg.norm(psi)
            psi_list.append(psi)

            # general i < j
            chi1 = oc.apply_gate(V, L, 2, 5, psi)
            # j < i
            chi2 = oc.apply_gate(V, L, 4, 1, psi)
            # j == i + 1
            chi3 = oc.apply_gate(V, L, 3, 4, psi)

            chi1_list.append(chi1)
            chi2_list.append(chi2)
            chi3_list.append(chi3)

        file["psi"]  = interleave_complex(np.asarray(psi_list).T, ctype)
        file["chi1"] = interleave_complex(np.asarray(chi1_list).T, ctype)
        file["chi2"] = interleave_complex(np.asarray(chi2_list).T, ctype)
        file["chi3"] = interleave_complex(np.asarray(chi3_list).T, ctype)

        file.close()


def apply_gate_placeholder_data():

    # random number generator
    rng = np.random.default_rng(44)

    # system size
    L = 7

    for ctype in ["real", "cplx"]:
        file = h5py.File(f"data/test_apply_gate_placeholder_{ctype}.hdf5", "w")

        # random input statevector
        psi = rng.standard_normal(2**L) if ctype == "real" else oc.crandn(2**L, rng)
        psi /= np.linalg.norm(psi)
        file["psi"] = interleave_complex(psi, ctype)

        # i < j
        i = 2
        j = 5
        # output statevector array
        psi_out1 = np.kron(psi, np.identity(4).reshape(-1))
        psi_out1 = psi_out1.reshape((2**i, 2, 2**(j - i - 1), 2, 2**(L - 1 - j), 2, 2, 2, 2))
        psi_out1 = psi_out1.transpose((0, 5, 2, 6, 4, 7, 8, 1, 3))
        psi_out1 = psi_out1.reshape(-1)
        file["psi_out1"] = interleave_complex(psi_out1, ctype)

        # i > j
        i = 5
        j = 1
        # output statevector array
        psi_out2 = np.kron(psi, np.identity(4).reshape(-1))
        psi_out2 = psi_out2.reshape((2**j, 2, 2**(i - j - 1), 2, 2**(L - 1 - i), 2, 2, 2, 2))
        psi_out2 = psi_out2.transpose((0, 5, 2, 6, 4, 8, 7, 3, 1))
        psi_out2 = psi_out2.reshape(-1)
        file["psi_out2"] = interleave_complex(psi_out2, ctype)

        file.close()


def main():
    apply_gate_data()
    apply_gate_backward_data()
    apply_gate_to_array_data()
    apply_gate_placeholder_data()


if __name__ == "__main__":
    main()
