#include <stdio.h>
#include <cblas.h>
#include "circuit_opt.h"
#include "util.h"
#include "timing.h"


#ifdef COMPLEX_CIRCUIT


static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* u = (numeric*)udata;

	// apply U
	numeric alpha = 1;
	numeric beta  = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, u, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
}


int main()
{
	const int nqubits = 5;
	const intqs n = (intqs)1 << nqubits;

	// read initial data from disk
	char filename[1024];
	sprintf(filename, "../examples/molecular/molecular_dynamics_opt_init.hdf5");
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}
	// coefficients, as reference
	double* tkin = aligned_malloc(n * n * sizeof(double));
	if (tkin == NULL) {
		fprintf(stderr, "memory allocation for kinetic coefficients failed\n");
		return -1;
	}
	if (read_hdf5_dataset(file, "tkin", H5T_NATIVE_DOUBLE, tkin) < 0) {
		fprintf(stderr, "reading 'tkin' from disk failed\n");
		return -1;
	}
	double* vint = aligned_malloc(n * n * sizeof(double));
	if (vint == NULL) {
		fprintf(stderr, "memory allocation for interaction coefficients failed\n");
		return -1;
	}
	if (read_hdf5_dataset(file, "vint", H5T_NATIVE_DOUBLE, vint) < 0) {
		fprintf(stderr, "reading 'vint' from disk failed\n");
		return -1;
	}
	// target unitary
	numeric* expiH = aligned_malloc(n * n * sizeof(numeric));
	if (expiH == NULL) {
		fprintf(stderr, "memory allocation for target unitary failed\n");
		return -1;
	}
	if (read_hdf5_dataset(file, "expiH", H5T_NATIVE_DOUBLE, expiH) < 0) {
		fprintf(stderr, "reading 'expiH' from disk failed\n");
		return -1;
	}
	// initial to-be optimized quantum gates
	hsize_t gates_start_dims[4];  // last dimension due to real and imaginary parts
	if (get_hdf5_dataset_dims(file, "gates_start", gates_start_dims)) {
		fprintf(stderr, "cannot obtain dimensions of initial two-qubit quantum gates");
		return -1;
	}
	int ngates = gates_start_dims[0];
	printf("number of quantum gates: %i\n", ngates);
	struct mat4x4* gates_start = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (read_hdf5_dataset(file, "gates_start", H5T_NATIVE_DOUBLE, gates_start) < 0) {
		fprintf(stderr, "reading initial two-qubit quantum gates from disk failed\n");
		return -1;
	}
	// corresponding quantum wires
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		fprintf(stderr, "reading wire indices from disk failed\n");
		return -1;
	}
	// evolution time
	double t;
	if (read_hdf5_attribute(file, "t", H5T_NATIVE_DOUBLE, &t) < 0) {
		fprintf(stderr, "reading evolution time from disk failed\n");
		return -1;
	}
	printf("t: %g\n", t);

	H5Fclose(file);

	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(ngates * 16, &params);

	// number of iterations
	const int niter = 10;

	double* f_iter = aligned_malloc((niter + 1) * sizeof(double));

	struct mat4x4* gates_opt = aligned_malloc(ngates * sizeof(struct mat4x4));

	uint64_t start_tick = get_ticks();

	// perform optimization
	optimize_quantum_circuit(ufunc, expiH, gates_start, ngates, nqubits, wires, &params, niter, f_iter, gates_opt);

	uint64_t total_ticks = get_ticks() - start_tick;
	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();
	printf("wall time: %g\n", total_ticks / ticks_per_sec);

	for (int i = 0; i < niter + 1; i++)
	{
		printf("f_iter[%i] = %.12f\n", i, f_iter[i]);
	}

	// save results to disk
	sprintf(filename, "../examples/molecular/molecular_dynamics_opt.hdf5");
	file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fcreate' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}
	hsize_t tkin_dims[2] = { nqubits, nqubits };
	if (write_hdf5_dataset(file, "tkin", 2, tkin_dims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, tkin) < 0) {
		fprintf(stderr, "writing 'tkin' to disk failed\n");
		return -1;
	}
	hsize_t vint_dims[2] = { nqubits, nqubits };
	if (write_hdf5_dataset(file, "vint", 2, vint_dims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, vint) < 0) {
		fprintf(stderr, "writing 'vint' to disk failed\n");
		return -1;
	}
	hsize_t wires_dims[2] = { ngates, 2 };
	if (write_hdf5_dataset(file, "wires", 2, wires_dims, H5T_STD_I32LE, H5T_NATIVE_INT, wires) < 0) {
		fprintf(stderr, "writing 'wires' to disk failed\n");
		return -1;
	}
	hsize_t gates_dims[4] = { ngates, 4, 4, 2 };
	if (write_hdf5_dataset(file, "gates", 4, gates_dims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, gates_opt) < 0) {
		fprintf(stderr, "writing 'gates_opt' to disk failed\n");
		return -1;
	}
	hsize_t fdims[1] = { niter + 1 };
	if (write_hdf5_dataset(file, "f_iter", 1, fdims, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, f_iter) < 0) {
		fprintf(stderr, "writing 'f_iter' to disk failed\n");
		return -1;
	}
	// store parameters
	if (write_hdf5_scalar_attribute(file, "nqubits", H5T_STD_I32LE, H5T_NATIVE_INT, &nqubits)) {
		fprintf(stderr, "writing 'nqubits' to disk failed\n");
		return -1;
	}
	if (write_hdf5_scalar_attribute(file, "t", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, &t)) {
		fprintf(stderr, "writing 't' to disk failed\n");
		return -1;
	}

	H5Fclose(file);

	aligned_free(gates_opt);
	aligned_free(f_iter);
	aligned_free(wires);
	aligned_free(gates_start);
	aligned_free(expiH);
	aligned_free(vint);
	aligned_free(tkin);

	return 0;
}


#else


int main()
{
	printf("Please re-build with support for complex numbers enabled.\n");
	return 0;
}


#endif
