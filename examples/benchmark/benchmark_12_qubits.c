// Performance benchmarking target function value and gradient computation for 12 qubits and 8 circuit layers.

#include <stdio.h>
#include <assert.h>
#include "brickwall_opt.h"
#include "util.h"
#include "timing.h"


static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	// suppress "unused parameter" warning
	(void)udata;

	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		#ifndef COMPLEX_CIRCUIT
		psi_out->data[i] = -1.1 * psi->data[((i + 3) * 113) % n] - 0.7 * psi->data[((i + 9) * 173) % n] + 0.5 * psi->data[i] + 0.3 * psi->data[((i + 4) * 199) % n];
		#else
		psi_out->data[i] = (-1.1 + 0.8*I) * psi->data[((i + 3) * 113) % n] + (0.4 - 0.7*I) * psi->data[((i + 9) * 173) % n] + (0.5 + 0.1*I) * psi->data[i] + (-0.3 + 0.2*I) * psi->data[((i + 4) * 199) % n];
		#endif
	}

	return 0;
}


#ifdef COMPLEX_CIRCUIT


int main()
{
	const int nqubits = 12;
	const int nlayers = 8;

	char filename[1024];
	sprintf(filename, "../examples/benchmark/benchmark_%i_qubits_data.hdf5", nqubits);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

	// quantum gates
	struct mat4x4* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, vlist) < 0) {
		fprintf(stderr, "reading two-qubit quantum gates from disk failed\n");
		return -1;
	}

	// permutations
	int** perms = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(int*));
	for (int i = 0; i < nlayers; i++)
	{
		perms[i] = aligned_alloc(MEM_DATA_ALIGN, nqubits * sizeof(int));

		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed");
			return -1;
		}
	}

	H5Fclose(file);

	const int m = nlayers * num_tangent_params;

	numeric f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));

	uint64_t start_tick = get_ticks();

	if (brickwall_unitary_target_and_projected_gradient(ufunc, NULL, vlist, nlayers, nqubits, (const int**)perms, &f, grad) < 0) {
		fprintf(stderr, "'brickwall_unitary_target_and_projected_gradient' failed internally");
		return -2;
	}

	uint64_t total_ticks = get_ticks() - start_tick;
	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();
	printf("benchmark completed, wall time: %g\n", total_ticks / ticks_per_sec);

	// clean up
	aligned_free(grad);
	for (int i = 0; i < nlayers; i++) {
		aligned_free(perms[i]);
	}
	aligned_free(perms);
	aligned_free(vlist);

	return 0;
}


#else


int main()
{
	printf("Cannot run benchmark. Please re-build with support for complex numbers enabled.\n");
	return 0;
}


#endif
