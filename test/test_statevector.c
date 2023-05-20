#include <stdio.h>
#include "config.h"
#include "statevector.h"
#include "util.h"
#include "file_io.h"


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

	if (read_data("../test/data/test_transpose_statevector" CDATA_LABEL "_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_data("../test/data/test_transpose_statevector" CDATA_LABEL "_chi.dat", chiref.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading output statevector data from disk failed";
	}

	int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
	if (read_data("../test/data/test_transpose_statevector" CDATA_LABEL "_perm.dat", perm, sizeof(int), L) < 0) {
		return "reading permutation data from disk failed";
	}

	transpose_statevector(&psi, perm, &chi);

	// compare with reference
	if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
		return "transposed quantum state does not match reference";
	}

	aligned_free(perm);
	free_statevector(&chiref);
	free_statevector(&chi);
	free_statevector(&psi);

	return 0;
}
