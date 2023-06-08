#include <stdio.h>
#include <cblas.h>
#include "brickwall_opt.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT


static int Ufunc(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* U = (numeric*)fdata;

	// apply U
	numeric alpha = 1;
	numeric beta = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, U, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
}


int main()
{
	const int L = 6;
	const int nlayers = 3;

	const intqs n = (intqs)1 << L;

	// read initial data from disk
	char filename[1024];
	sprintf(filename, "../examples/ising1d/ising1d_dynamics_opt_n%i_init.hdf5", nlayers);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}
	// target unitary
	numeric* expiH = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(numeric));
	if (expiH == NULL) {
		fprintf(stderr, "memory allocation for target unitary failed\n");
		return -1;
	}
	if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
		fprintf(stderr, "reading 'expiH' from disk failed\n");
		return -1;
	}
	// initial to-be optimized quantum gates
	struct mat4x4* Vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (read_hdf5_dataset(file, "Vlist_start", H5T_NATIVE_DOUBLE, Vlist_start) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	}
	// permutations
	int perms[3][8];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed\n");
			return -1;
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };
	H5Fclose(file);

	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(nlayers * 16, &params);

	// number of iterations
	const int niter = 10;

	double* f_iter = aligned_alloc(MEM_DATA_ALIGN, niter * sizeof(double));

	struct mat4x4* Vlist_opt = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

	// perform optimization
	optimize_brickwall_circuit_matfree(L, Ufunc, expiH, Vlist_start, nlayers, pperms, &params, niter, f_iter, Vlist_opt);

	for (int i = 0; i < niter; i++)
	{
		printf("f_iter[%i] = %.12f\n", i, f_iter[i]);
	}

	// save results to disk
	sprintf(filename, "../examples/ising1d/ising1d_dynamics_opt_n%i.hdf5", nlayers);
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}
	hsize_t vdims[4] = { nlayers, 4, 4, 2 };
	if (write_hdf5_dataset(file, "Vlist", 4, vdims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, Vlist_opt) < 0) {
		fprintf(stderr, "writing 'Vlist_opt' to disk failed\n");
		return -1;
	}
	hsize_t fdims[1] = { niter };
	if (write_hdf5_dataset(file, "f_iter", 1, fdims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, f_iter) < 0) {
		fprintf(stderr, "writing 'f_iter' to disk failed\n");
		return -1;
	}
	// store parameters
	if (write_hdf5_scalar_attribute(file, "L", H5T_STD_I32LE, H5T_NATIVE_INT, &L)) {
		fprintf(stderr, "writing 'L' to disk failed\n");
		return -1;
	}
	H5Fclose(file);

	aligned_free(Vlist_opt);
	aligned_free(f_iter);
	aligned_free(Vlist_start);
	aligned_free(expiH);

	return 0;
}


#else


int main()
{
	printf("Cannot perform optimization. Please re-build with support for complex numbers enabled.\n");
	return 0;
}


#endif
