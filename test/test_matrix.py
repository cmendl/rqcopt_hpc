import numpy as np
import rqcopt_matfree as oc


def multiply_data():

    # random number generator
    rng = np.random.default_rng(42)

    a = rng.standard_normal((4, 4))
    b = rng.standard_normal((4, 4))
    c = a @ b
    # save to disk
    a.tofile("data/test_multiply_a.dat")
    b.tofile("data/test_multiply_b.dat")
    c.tofile("data/test_multiply_c.dat")


def project_unitary_tangent_data():

    # random number generator
    rng = np.random.default_rng(43)

    u = rng.standard_normal((4, 4))
    z = rng.standard_normal((4, 4))

    p = oc.project_unitary_tangent(u, z)

    # save to disk
    u.tofile("data/test_project_unitary_tangent_u.dat")
    z.tofile("data/test_project_unitary_tangent_z.dat")
    p.tofile("data/test_project_unitary_tangent_p.dat")


def main():
    multiply_data()
    project_unitary_tangent_data()


if __name__ == "__main__":
    main()
