// Performance benchmarking gradient and Hessian computation for a single statevector application, for 16 qubits and 8 circuit layers.

#include <stdio.h>
#include "brickwall_circuit.h"
#include "util.h"
#include "timing.h"


#ifdef COMPLEX_CIRCUIT


int main()
{
	const int nqubits = 16;
	const int nlayers = 8;

	char filename[1024];
	sprintf(filename, "../examples/benchmark/benchmark_%i_qubits_data.hdf5", nqubits);
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		fprintf(stderr, "'H5Fopen' for '%s' failed, return value: %" PRId64 "\n", filename, file);
		return -1;
	}

	struct statevector psi, psi_out, dpsi_out, dpsi;
	if (allocate_statevector(nqubits, &psi)      < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
	if (allocate_statevector(nqubits, &psi_out)  < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
	if (allocate_statevector(nqubits, &dpsi_out) < 0) { fprintf(stderr, "memory allocation failed");  return -1; }
	if (allocate_statevector(nqubits, &dpsi)     < 0) { fprintf(stderr, "memory allocation failed");  return -1; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		fprintf(stderr, "reading input statevector data from disk failed");
		return -1;
	}

	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		fprintf(stderr, "reading upstream gradient data from disk failed");
		return -1;
	}

	// quantum gates
	struct mat4x4* vlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (read_hdf5_dataset(file, "vlist", H5T_NATIVE_DOUBLE, vlist) < 0) {
		fprintf(stderr, "reading two-qubit quantum gates from disk failed\n");
		return -1;
	}

	// permutations
	int** perms = aligned_malloc(nlayers * sizeof(int*));
	for (int i = 0; i < nlayers; i++)
	{
		perms[i] = aligned_malloc(nqubits * sizeof(int));

		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			fprintf(stderr, "reading permutation data from disk failed");
			return -1;
		}
	}

	H5Fclose(file);

	struct quantum_circuit_cache cache;
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	const int m = nlayers * 16;

	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	numeric* hess = aligned_malloc(m * m * sizeof(numeric));

	uint64_t start_tick = get_ticks();

	// brickwall unitary forward pass
	if (brickwall_unitary_forward(vlist, nlayers, (const int**)perms, &psi, &cache, &psi_out) < 0) {
		fprintf(stderr, "'brickwall_unitary_forward' failed internally");
		return -2;
	}

	// brickwall unitary backward pass and Hessian computation
	if (brickwall_unitary_backward_hessian(vlist, nlayers, (const int**)perms, &cache, &dpsi_out, &dpsi, dvlist, hess) < 0) {
		fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
		return -2;
	}

	uint64_t total_ticks = get_ticks() - start_tick;
	// get the tick resolution
	const double ticks_per_sec = (double)get_tick_resolution();
	printf("benchmark completed, wall time: %g\n", total_ticks / ticks_per_sec);

	// clean up
	aligned_free(hess);
	aligned_free(dvlist);
	free_quantum_circuit_cache(&cache);
	for (int i = 0; i < nlayers; i++) {
		aligned_free(perms[i]);
	}
	aligned_free(perms);
	aligned_free(vlist);
	free_statevector(&dpsi);
	free_statevector(&dpsi_out);
	free_statevector(&psi_out);
	free_statevector(&psi);

	return 0;
}


#else


int main()
{
	printf("Cannot run benchmark. Please re-build with support for complex numbers enabled.\n");
	return 0;
}


#endif
