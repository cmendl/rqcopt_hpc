#include "gate.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_gate()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_gate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate failed";
	}

	struct statevector psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3ref) < 0) { return "memory allocation failed"; }

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi1", H5T_NATIVE_DOUBLE, chi1ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi2", H5T_NATIVE_DOUBLE, chi2ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi3", H5T_NATIVE_DOUBLE, chi3ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// apply the gate
	apply_gate(&V, 2, 5, &psi, &chi1);
	apply_gate(&V, 4, 1, &psi, &chi2);
	apply_gate(&V, 3, 4, &psi, &chi3);

	// compare with reference
	if (uniform_distance((size_t)1 << L, chi1.data, chi1ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((size_t)1 << L, chi2.data, chi2ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((size_t)1 << L, chi3.data, chi3ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }

	free_statevector(&chi3ref);
	free_statevector(&chi3);
	free_statevector(&chi2ref);
	free_statevector(&chi2);
	free_statevector(&chi1ref);
	free_statevector(&chi1);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}
