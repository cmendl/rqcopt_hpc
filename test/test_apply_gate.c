#include <stdio.h>
#include "config.h"
#include "gate.h"
#include "statevector.h"
#include "apply_gate.h"
#include "util.h"
#include "file_io.h"
#include "minunit.h"


int tests_run = 0;


static char* test_apply_gate()
{
	int nqubits = 9;

	struct statevector psi;
	if (allocate_statevector(nqubits, &psi) < 0) {
		return "memory allocation failed";
	}
	struct statevector psi1, psi1ref;
	if (allocate_statevector(nqubits, &psi1) < 0) {
		return "memory allocation failed";
	}
	if (allocate_statevector(nqubits, &psi1ref) < 0) {
		return "memory allocation failed";
	}
	struct statevector psi2, psi2ref;
	if (allocate_statevector(nqubits, &psi2) < 0) {
		return "memory allocation failed";
	}
	if (allocate_statevector(nqubits, &psi2ref) < 0) {
		return "memory allocation failed";
	}

	if (read_data("../../../test/test_apply_gate_psi.dat", psi.data, sizeof(numeric), (size_t)1 << nqubits) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_data("../../../test/test_apply_gate_psi1.dat", psi1ref.data, sizeof(numeric), (size_t)1 << nqubits) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_data("../../../test/test_apply_gate_psi2.dat", psi2ref.data, sizeof(numeric), (size_t)1 << nqubits) < 0) {
		return "reading output statevector data from disk failed";
	}

	struct single_qubit_gate U;
	if (read_data("../../../test/test_apply_gate_U.dat", U.data, sizeof(numeric), 4) < 0)
	{
		return "reading single-qubit quantum gate entries from disk failed";
	}
	struct two_qubit_gate V;
	if (read_data("../../../test/test_apply_gate_V.dat", V.data, sizeof(numeric), 16) < 0)
	{
		return "reading two-qubit quantum gate entries from disk failed";
	}

	// apply the gates
	apply_single_qubit_gate(&U, 3, &psi, &psi1);
	apply_two_qubit_gate(&V, 2, 5, &psi, &psi2);

	// compare with reference
	mu_assert(relative_distance((size_t)1 << nqubits, psi1.data, psi1ref.data, 1e-8) < 1e-12, "output quantum state after single-qubit gate does not match reference");
	mu_assert(relative_distance((size_t)1 << nqubits, psi2.data, psi2ref.data, 1e-8) < 1e-12, "output quantum state after two-qubit gate does not match reference");

	free_statevector(&psi2ref);
	free_statevector(&psi2);
	free_statevector(&psi1ref);
	free_statevector(&psi1);
	free_statevector(&psi);

	return 0;
}


static char* all_tests()
{
	mu_run_test(test_apply_gate);
	return 0;
}


int main()
{
	char* result = all_tests();
	if (result != 0)
	{
		printf("%s\n", result);
	}
	else
	{
		printf("ALL TESTS PASSED\n");
	}
	printf("Tests run: %d\n", tests_run);

	return result != 0;
}
