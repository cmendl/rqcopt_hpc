#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "matrix.h"
#include "statevector.h"
#include "brickwall_circuit.h"
#include "util.h"
#include "file_io.h"


char* test_apply_brickwall_unitary()
{
	int L = 8;

	struct statevector psi, chi, chiref;
	if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

	if (read_data("../../../test/data/test_apply_brickwall_unitary_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_brickwall_unitary_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_brickwall_unitary_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		// apply the brickwall unitary
		apply_brickwall_unitary(Vlist, nlayers, &psi, pperms, &chi);

		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_brickwall_unitary_chi%i.dat", nlayers - 3);
		if (read_data(filename, chiref.data, sizeof(numeric), (size_t)1 << L) < 0) {
			return "reading output statevector data from disk failed";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
			return "quantum state after applying gate does not match reference";
		}
	}

	free_statevector(&chiref);
	free_statevector(&chi);
	free_statevector(&psi);

	return 0;
}


char* test_apply_adjoint_brickwall_unitary()
{
	int L = 8;

	struct statevector psi, chi, chiref;
	if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

	if (read_data("../../../test/data/test_apply_adjoint_brickwall_unitary_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_adjoint_brickwall_unitary_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_adjoint_brickwall_unitary_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		// apply the adjoint_brickwall unitary
		apply_adjoint_brickwall_unitary(Vlist, nlayers, &psi, pperms, &chi);

		char filename[1024];
		sprintf_s(filename, 1024, "../../../test/data/test_apply_adjoint_brickwall_unitary_chi%i.dat", nlayers - 3);
		if (read_data(filename, chiref.data, sizeof(numeric), (size_t)1 << L) < 0) {
			return "reading output statevector data from disk failed";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chiref.data) > 1e-12) {
			return "quantum state after applying gate does not match reference";
		}
	}

	free_statevector(&chiref);
	free_statevector(&chi);
	free_statevector(&psi);

	return 0;
}
