import numpy as np
from scipy.stats import unitary_group, ortho_group
import h5py
import rqcopt_matfree as oc
from io_util import interleave_complex


def real_to_unitary_tangent_data():

    # random number generator
    rng = np.random.default_rng(41)

    ctype = "cplx"

    r = 0.5 * rng.standard_normal((4, 4))
    v = unitary_group.rvs(4, random_state=rng)

    with h5py.File(f"data/test_real_to_unitary_tangent_{ctype}.hdf5", "w") as file:
        file["r"] = r
        file["v"] = interleave_complex(v, ctype)


def real_to_ortho_tangent_data():

    # random number generator
    rng = np.random.default_rng(41)

    ctype = "real"

    r = 0.5 * rng.standard_normal(6)
    v = ortho_group.rvs(4, random_state=rng)

    with h5py.File(f"data/test_real_to_ortho_tangent_{ctype}.hdf5", "w") as file:
        file["r"] = r
        file["v"] = interleave_complex(v, ctype)


def multiply_data():

    # random number generator
    rng = np.random.default_rng(42)

    for ctype in ["real", "cplx"]:
        a = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        b = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        c = a @ b
        # save to disk
        with h5py.File(f"data/test_multiply_{ctype}.hdf5", "w") as file:
            file["a"] = interleave_complex(a, ctype)
            file["b"] = interleave_complex(b, ctype)
            file["c"] = interleave_complex(c, ctype)


def project_unitary_tangent_data():

    # random number generator
    rng = np.random.default_rng(43)

    for ctype in ["real", "cplx"]:
        u = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        z = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)

        p = oc.project_unitary_tangent(u, z)

        # save to disk
        with h5py.File(f"data/test_project_unitary_tangent_{ctype}.hdf5", "w") as file:
            file["u"] = interleave_complex(u, ctype)
            file["z"] = interleave_complex(z, ctype)
            file["p"] = interleave_complex(p, ctype)


def polar_factor_data():

    # random number generator
    rng = np.random.default_rng(44)

    for ctype in ["real", "cplx"]:
        a = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        u, _ = oc.polar_decomp(a)
        # save to disk
        with h5py.File(f"data/test_polar_factor_{ctype}.hdf5", "w") as file:
            file["a"] = interleave_complex(a, ctype)
            file["u"] = interleave_complex(u, ctype)


def main():
    real_to_unitary_tangent_data()
    real_to_ortho_tangent_data()
    multiply_data()
    project_unitary_tangent_data()
    polar_factor_data()


if __name__ == "__main__":
    main()
