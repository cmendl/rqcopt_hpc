#include <stdio.h>
#include "config.h"
#include "gate.h"
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

	struct two_qubit_gate V;
	if (read_data("../../../test/test_apply_gate_V.dat", V.data, sizeof(numeric), 16) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_data("../../../test/test_apply_gate_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_data("../../../test/test_apply_gate_chi1.dat", chi1ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_data("../../../test/test_apply_gate_chi2.dat", chi2ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_data("../../../test/test_apply_gate_chi3.dat", chi3ref.data, sizeof(numeric), (size_t)1 << L) < 0) {
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
	struct two_qubit_gate V;
	if (read_data("../../../test/test_apply_parallel_gates_V.dat", V.data, sizeof(numeric), 16) < 0) {
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

		sprintf_s(filename, 1024, "../../../test/test_apply_parallel_gates_psi%i.dat", i);
		if (read_data(filename, psi.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading input statevector data from disk failed"; }
		sprintf_s(filename, 1024, "../../../test/test_apply_parallel_gates_chi%i.dat", i);
		if (read_data(filename, chiref.data, sizeof(numeric), (size_t)1 << L) < 0) { return "reading output statevector data from disk failed"; }

		int* perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
		sprintf_s(filename, 1024, "../../../test/test_apply_parallel_gates_perm%i.dat", i);
		if (read_data(filename, perm, sizeof(int), L) < 0) { return "reading permutation data from disk failed"; }

		if (apply_parallel_gates(&V, &psi, perm, &chi) < 0) {
			return "'apply_parallel_gates' failed internally";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
			return "quantum state after applying parallel gates does not match reference";
		}

		free_statevector(&chiref);
		free_statevector(&chi);
		free_statevector(&psi);
	}

	return 0;
}
