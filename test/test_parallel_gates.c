#include <assert.h>
#include "parallel_gates.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_parallel_gates()
{
	hid_t file = H5Fopen("../test/data/test_apply_parallel_gates" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_parallel_gates failed";
	}

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	const int Llist[3] = { 6, 8, 10 };
	for (int i = 0; i < 3; i++)
	{
		int L = Llist[i];

		struct statevector psi, chi, chiref;
		if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
		if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
		if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

		char varname[32];

		sprintf(varname, "psi%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psi.data) < 0) { return "reading input statevector data from disk failed"; }
		sprintf(varname, "chi%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, chiref.data) < 0) { return "reading output statevector data from disk failed"; }

		int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perm) < 0) { return "reading permutation data from disk failed"; }

		if (apply_parallel_gates(&V, &psi, perm, &chi) < 0) {
			return "'apply_parallel_gates' failed internally";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
			return "quantum state after applying parallel gates does not match reference";
		}

		aligned_free(perm);
		free_statevector(&chiref);
		free_statevector(&chi);
		free_statevector(&psi);
	}

	H5Fclose(file);

	return 0;
}


char* test_apply_parallel_gates_directed_grad()
{
	hid_t file = H5Fopen("../test/data/test_apply_parallel_gates_directed_grad" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_parallel_gates_directed_grad failed";
	}

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	struct mat4x4 Z;
	if (read_hdf5_dataset(file, "Z", H5T_NATIVE_DOUBLE, Z.data) < 0) {
		return "reading gradient direction data from disk failed";
	}

	const int Llist[3] = { 6, 8, 10 };
	for (int i = 0; i < 3; i++)
	{
		int L = Llist[i];

		struct statevector psi, chi, chiref;
		if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
		if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
		if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

		char varname[32];

		sprintf(varname, "psi%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psi.data) < 0) { return "reading input statevector data from disk failed"; }
		sprintf(varname, "chi%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, chiref.data) < 0) { return "reading output statevector data from disk failed"; }

		int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perm) < 0) { return "reading permutation data from disk failed"; }

		if (apply_parallel_gates_directed_grad(&V, &Z, &psi, perm, &chi) < 0) {
			return "'apply_parallel_gates_directed_grad' failed internally";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
			return "quantum state after applying parallel gates in gradient direction does not match reference";
		}

		aligned_free(perm);
		free_statevector(&chiref);
		free_statevector(&chi);
		free_statevector(&psi);
	}

	H5Fclose(file);

	return 0;
}


static int Ufunc(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		psi_out->data[i] = 0.7 * psi->data[((i + 5) * 83) % n] - 0.2 * psi->data[i] + 1.3 * psi->data[((i + 1) * 181) % n] + 0.4 * psi->data[((i + 7) * 197) % n];
	}

	return 0;
}


char* test_parallel_gates_grad_matfree()
{
	hid_t file = H5Fopen("../test/data/test_parallel_gates_grad_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_parallel_gates_grad_matfree failed";
	}

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	int idpm[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	int flpm[2] = { 1, 0 };
	int perm[8];
	if (read_hdf5_dataset(file, "perm", H5T_NATIVE_INT, perm) < 0) {
		return "reading permutation data from disk failed";
	}
	const int* perms[2][2] = { { idpm, flpm }, { idpm, perm } };

	const int Llist[3] = { 2, 8 };
	for (int j = 0; j < 2; j++)
	{
		for (int i = 0; i < 2; i++)
		{
			struct mat4x4 dV;
			if (parallel_gates_grad_matfree(&V, Llist[j], Ufunc, NULL, perms[j][i], &dV) < 0) {
				return "'parallel_gates_grad_matfree' failed internally";
			}

			struct mat4x4 dVref;
			char varname[32];
			sprintf(varname, "dV%iL%i", i, Llist[j]);
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVref.data) < 0) {
				return "reading reference gradient data from disk failed";
			}

			// compare with reference
			if (relative_distance(16, dV.data, dVref.data, 1e-8) > 1e-12) {
				return "computed gradient for parallel gates does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}


char* test_parallel_gates_hess_matfree()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_parallel_gates_hess_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_parallel_gates_hess_matfree failed";
	}

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	int perm[8];
	if (read_hdf5_dataset(file, "perm", H5T_NATIVE_INT, perm) < 0) {
		return "reading permutation data from disk failed";
	}

	for (int i = 0; i < 2; i++)
	{
		char varname[32];
		struct mat4x4 Z;
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Z.data) < 0) {
			return "reading gradient direction data from disk failed";
		}

		for (int uproj = 0; uproj < 2; uproj++)
		{
			struct mat4x4 dV;
			if (parallel_gates_hess_matfree(&V, L, &Z, Ufunc, NULL, perm, uproj, &dV) < 0) {
				return "'parallel_gates_hess_matfree' failed internally";
			}

			struct mat4x4 dVref;
			sprintf(varname, "dV%i%s", i, uproj == 1 ? "proj" : "");
			if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVref.data) < 0) {
				return "reading reference Hessian data from disk failed";
			}

			// compare with reference
			if (relative_distance(16, dV.data, dVref.data, 1e-8) > 1e-12) {
				return "computed Hessian for parallel gates does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}
