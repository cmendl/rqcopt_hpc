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

	if (read_data("../test/data/test_apply_brickwall_unitary_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_brickwall_unitary_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_brickwall_unitary_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		// apply the brickwall unitary
		if (apply_brickwall_unitary(Vlist, nlayers, &psi, pperms, &chi) < 0) {
			return "'apply_brickwall_unitary' failed internally";
		}

		char filename[1024];
		sprintf(filename, "../test/data/test_apply_brickwall_unitary_chi%i.dat", nlayers - 3);
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

	if (read_data("../test/data/test_apply_adjoint_brickwall_unitary_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		// apply the adjoint_brickwall unitary
		if (apply_adjoint_brickwall_unitary(Vlist, nlayers, &psi, pperms, &chi) < 0) {
			return "'apply_adjoint_brickwall_unitary' failed internally";
		}

		char filename[1024];
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary_chi%i.dat", nlayers - 3);
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


static int Ufunc(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		psi_out->data[i] = -1.1 * psi->data[((i + 3) * 113) % n] - 0.7 * psi->data[((i + 9) * 173) % n] + 0.5 * psi->data[i] + 0.3 * psi->data[((i + 4) * 199) % n];
	}

	return 0;
}


char* test_brickwall_unitary_grad_matfree()
{
	int L = 8;

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 1, 4, 5 };
	for (int i = 0; i < 3; i++)
	{
		struct mat4x4 dVlist[5];
		if (brickwall_unitary_grad_matfree(Vlist, nlayers[i], L, Ufunc, NULL, pperms, dVlist) < 0) {
			return "'brickwall_unitary_grad_matfree' failed internally";
		}

		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree_dVlist%i.dat", i);
		struct mat4x4 dVlist_ref[5];
		if (read_data(filename, dVlist_ref, sizeof(numeric), nlayers[i] * 16) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_ref[j].data) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}
	}

	return 0;
}
