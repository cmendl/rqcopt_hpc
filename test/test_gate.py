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
