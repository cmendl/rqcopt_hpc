#include "config.h"
#include "statevector.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_transpose_statevector()
{
	int L = 9;

	struct statevector psi, chi, chiref;
	if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

	hid_t file = H5Fopen("../test/data/test_transpose_statevector" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_transpose_statevector failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi", H5T_NATIVE_DOUBLE, chiref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	int* perm = aligned_malloc(L * sizeof(int));
	if (read_hdf5_dataset(file, "perm", H5T_NATIVE_INT, perm) < 0) {
		return "reading permutation data from disk failed";
	}

	transpose_statevector(&psi, perm, &chi);

	// compare with reference
	if (uniform_distance((long)1 << L, chi.data, chiref.data) > 1e-12) {
		return "transposed quantum state does not match reference";
	}

	H5Fclose(file);

	aligned_free(perm);
	free_statevector(&chiref);
	free_statevector(&chi);
	free_statevector(&psi);

	return 0;
}
