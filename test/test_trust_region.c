#include <math.h>
#include "config.h"
#include "trust_region.h"
#include "io_util.h"


char* test_truncated_cg()
{
	hid_t file = H5Fopen("../test/data/test_truncated_cg.hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_truncated_cg failed";
	}

	int n = 12;
	double radius = 1.5;

	double* grad = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad) < 0) {
		return "reading gradient vector from disk failed";
	}

	double* hess = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(double));
	if (read_hdf5_dataset(file, "hess", H5T_NATIVE_DOUBLE, hess) < 0) {
		return "reading Hessian matrix from disk failed";
	}

	// reference data
	double* zref = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	if (read_hdf5_dataset(file, "z", H5T_NATIVE_DOUBLE, zref) < 0) {
		return "reading reference tCG solution vector from disk failed";
	}
	int on_boundary_ref;
	if (read_hdf5_dataset(file, "on_boundary", H5T_NATIVE_INT, &on_boundary_ref) < 0) {
		return "reading reference tCG on boundary indicator from disk failed";
	}

	struct truncated_cg_params params;
	set_truncated_cg_default_params(n, &params);

	double* z = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	bool on_boundary = truncated_cg(grad, hess, n, radius, &params, z);

	// compare with reference
	double d = 0;
	for (int i = 0; i < n; i++)
	{
		d = fmax(d, fabs(z[i] - zref[i]));
	}
	if (d > 1e-14) {
		return "computed tCG solution vector does not match reference";
	}
	if (on_boundary != on_boundary_ref) {
		return "computed tCG on boundary indicator does not match reference";
	}

	aligned_free(z);
	aligned_free(zref);
	aligned_free(hess);
	aligned_free(grad);

	H5Fclose(file);

	return 0;
}
