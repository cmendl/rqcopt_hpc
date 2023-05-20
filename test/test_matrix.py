import numpy as np
import rqcopt_matfree as oc


def multiply_data():

    # random number generator
    rng = np.random.default_rng(42)

    for ctype in ["real", "cplx"]:
        a = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        b = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        c = a @ b
        # save to disk
        a.tofile(f"data/test_multiply_{ctype}_a.dat")
        b.tofile(f"data/test_multiply_{ctype}_b.dat")
        c.tofile(f"data/test_multiply_{ctype}_c.dat")


def project_unitary_tangent_data():

    # random number generator
    rng = np.random.default_rng(43)

    for ctype in ["real", "cplx"]:
        u = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)
        z = rng.standard_normal((4, 4)) if ctype == "real" else oc.crandn((4, 4), rng)

        p = oc.project_unitary_tangent(u, z)

        # save to disk
        u.tofile(f"data/test_project_unitary_tangent_{ctype}_u.dat")
        z.tofile(f"data/test_project_unitary_tangent_{ctype}_z.dat")
        p.tofile(f"data/test_project_unitary_tangent_{ctype}_p.dat")


def main():
    multiply_data()
    project_unitary_tangent_data()


if __name__ == "__main__":
    main()
