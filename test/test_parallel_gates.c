#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "matrix.h"
#include "statevector.h"
#include "parallel_gates.h"
#include "util.h"
#include "file_io.h"


char* test_apply_gate()
{
	int L = 9;

	struct statevector psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3ref) < 0) { return "memory allocation failed"; }

	struct mat4x4 V;
	if (read_data("../test/data/test_apply_gate_V.dat", V.data, sizeof(numeric), 16) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_data("../test/data/test_apply_gate_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_data("../test/data/test_apply_gate_chi1.dat", chi1ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_data("../test/data/test_apply_gate_chi2.dat", chi2ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_data("../test/data/test_apply_gate_chi3.dat", chi3ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
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

	return 0;
}


char* test_apply_parallel_gates()
{
	struct mat4x4 V;
	if (read_data("../test/data/test_apply_parallel_gates_V.dat", V.data, sizeof(numeric), 16) < 0) {
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

		char filename[1024];

		sprintf(filename, "../test/data/test_apply_parallel_gates_psi%i.dat", i);
		if (read_data(filename, psi.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading input statevector data from disk failed"; }
		sprintf(filename, "../test/data/test_apply_parallel_gates_chi%i.dat", i);
		if (read_data(filename, chiref.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading output statevector data from disk failed"; }

		int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
		sprintf(filename, "../test/data/test_apply_parallel_gates_perm%i.dat", i);
		if (read_data(filename, perm, sizeof(int), L) < 0) { return "reading permutation data from disk failed"; }

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

	return 0;
}


char* test_apply_parallel_gates_directed_grad()
{
	struct mat4x4 V;
	if (read_data("../test/data/test_apply_parallel_gates_directed_grad_V.dat", V.data, sizeof(numeric), 16) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	struct mat4x4 Z;
	if (read_data("../test/data/test_apply_parallel_gates_directed_grad_Z.dat", Z.data, sizeof(numeric), 16) < 0) {
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

		char filename[1024];

		sprintf(filename, "../test/data/test_apply_parallel_gates_directed_grad_psi%i.dat", i);
		if (read_data(filename, psi.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading input statevector data from disk failed"; }
		sprintf(filename, "../test/data/test_apply_parallel_gates_directed_grad_chi%i.dat", i);
		if (read_data(filename, chiref.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading output statevector data from disk failed"; }

		int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
		sprintf(filename, "../test/data/test_apply_parallel_gates_directed_grad_perm%i.dat", i);
		if (read_data(filename, perm, sizeof(int), L) < 0) { return "reading permutation data from disk failed"; }

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
	struct mat4x4 V;
	if (read_data("../test/data/test_parallel_gates_grad_matfree_V.dat", V.data, sizeof(numeric), 16) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}
	int idpm[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	int flpm[2] = { 1, 0 };
	int perm[8];
	if (read_data("../test/data/test_parallel_gates_grad_matfree_perm.dat", perm, sizeof(int), 8) < 0) {
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
			char filename[1024];
			sprintf(filename, "../test/data/test_parallel_gates_grad_matfree_dV%iL%i.dat", i, Llist[j]);
			if (read_data(filename, dVref.data, sizeof(numeric), 16) < 0) {
				return "reading reference gradient data from disk failed";
			}

			// compare with reference
			if (relative_distance(16, dV.data, dVref.data, 1e-8) > 1e-12) {
				return "computed gradient for parallel gates does not match reference";
			}
		}
	}

	return 0;
}


char* test_parallel_gates_hess_matfree()
{
	int L = 8;

	struct mat4x4 V;
	if (read_data("../test/data/test_parallel_gates_hess_matfree_V.dat", V.data, sizeof(numeric), 16) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	int perm[8];
	if (read_data("../test/data/test_parallel_gates_hess_matfree_perm.dat", perm, sizeof(int), 8) < 0) {
		return "reading permutation data from disk failed";
	}

	for (int i = 0; i < 2; i++)
	{
		char filename[1024];
		struct mat4x4 Z;
		sprintf(filename, "../test/data/test_parallel_gates_hess_matfree_Z%i.dat", i);
		if (read_data(filename, Z.data, sizeof(numeric), 16) < 0) {
			return "reading gradient direction data from disk failed";
		}

		for (int uproj = 0; uproj < 2; uproj++)
		{
			struct mat4x4 dV;
			if (parallel_gates_hess_matfree(&V, L, &Z, Ufunc, NULL, perm, uproj, &dV) < 0) {
				return "'parallel_gates_hess_matfree' failed internally";
			}

			struct mat4x4 dVref;
			sprintf(filename, "../test/data/test_parallel_gates_hess_matfree_dV%i%s.dat", i, uproj == 1 ? "proj" : "");
			if (read_data(filename, dVref.data, sizeof(numeric), 16) < 0) {
				return "reading reference Hessian data from disk failed";
			}

			// compare with reference
			if (relative_distance(16, dV.data, dVref.data, 1e-8) > 1e-12) {
				return "computed Hessian for parallel gates does not match reference";
			}
		}
	}

	return 0;
}
