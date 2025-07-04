#include "config.h"
#include "statevector.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


//________________________________________________________________________________________________________________________
///
/// \brief Square function x -> x^2.
///
static inline double square(const double x)
{
	return x*x;
}


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


char* test_haar_random_statevector()
{
	const int nqubits = 3;

	struct statevector psi;
	if (allocate_statevector(nqubits, &psi) < 0) {
		return "memory allocation failed";
	}

	struct rng_state rng;
	seed_rng_state(42, &rng);

	const long nsamples = 1731;
	double c = 0;
	for (long n = 0; n < nsamples; n++)
	{
		haar_random_statevector(&psi, &rng);
		c += square(_abs(psi.data[0]));
	}
	c /= nsamples;

	// theoretical expectation value
	const double c_ref = 1. / ((intqs)1 << nqubits);

	if (fabs(c - c_ref) / fabs(c_ref) > 0.05) {
		return "empirical average deviates from the expectation value";
	}

	free_statevector(&psi);

	return 0;
}
