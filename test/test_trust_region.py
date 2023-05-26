import numpy as np
import h5py
import rqcopt_matfree as oc


def truncated_cg_data():

    # random number generator
    rng = np.random.default_rng(42)

    n = 12

    radius = 1.5

    grad = 0.1 * rng.standard_normal(n)
    hess = oc.symm(rng.standard_normal((n, n)))

    z, on_boundary = oc.truncated_cg(grad, hess, radius)

    # save to disk
    with h5py.File("data/test_truncated_cg.hdf5", "w") as file:
        file["grad"] = grad
        file["hess"] = hess
        file["z"]    = z
        file["on_boundary"] = int(on_boundary)


def main():
    truncated_cg_data()


if __name__ == "__main__":
    main()
