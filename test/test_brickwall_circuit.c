#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "matrix.h"
#include "statevector.h"
#include "brickwall_circuit.h"
#include "util.h"
#include "file_io.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_brickwall_unitary()
{
	int L = 8;

	struct statevector psi, chi, chiref;
	if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

	if (read_data("../test/data/test_apply_brickwall_unitary" CDATA_LABEL "_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_brickwall_unitary" CDATA_LABEL "_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_brickwall_unitary" CDATA_LABEL "_perm%i.dat", i);
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
		sprintf(filename, "../test/data/test_apply_brickwall_unitary" CDATA_LABEL "_chi%i.dat", nlayers - 3);
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

	if (read_data("../test/data/test_apply_adjoint_brickwall_unitary" CDATA_LABEL "_psi.dat", psi.data, sizeof(numeric), (size_t)1 << L) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary" CDATA_LABEL "_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary" CDATA_LABEL "_perm%i.dat", i);
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
		sprintf(filename, "../test/data/test_apply_adjoint_brickwall_unitary" CDATA_LABEL "_chi%i.dat", nlayers - 3);
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
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree" CDATA_LABEL "_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree" CDATA_LABEL "_perm%i.dat", i);
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
		sprintf(filename, "../test/data/test_brickwall_unitary_grad_matfree" CDATA_LABEL "_dVlist%i.dat", i);
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


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_gradient_vector_matfree()
{
	int L = 6;
	int nlayers = 3;

	struct mat4x4 Vlist[3];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_gradient_vector_matfree_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[3][6];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_gradient_vector_matfree_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };

	double grad[3 * 16];
	if (brickwall_unitary_gradient_vector_matfree(Vlist, nlayers, L, Ufunc, NULL, pperms, grad) < 0) {
		return "'brickwall_unitary_gradient_vector_matfree' failed internally";
	}

	double grad_ref[3 * 16];
	if (read_data("../test/data/test_brickwall_unitary_gradient_vector_matfree_grad.dat", grad_ref, sizeof(double), nlayers * 16) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	double d = 0;
	for (int i = 0; i < nlayers * 16; i++)
	{
		d = fmax(d, fabs(grad[i] - grad_ref[i]));
	}
	if (d > 1e-14) {
		return "computed brickwall circuit gradient vector does not match reference";
	}

	return 0;
}

#endif


char* test_brickwall_unitary_hess_matfree()
{
	int L = 6;
	int nlayers = 4;

	struct mat4x4 Vlist[4];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_hess_matfree" CDATA_LABEL "_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][6];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_hess_matfree" CDATA_LABEL "_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	// gradient direction
	struct mat4x4 rZ;
	if (read_data("../test/data/test_brickwall_unitary_hess_matfree" CDATA_LABEL "_rZ.dat", rZ.data, sizeof(numeric), 16) < 0) {
		return "reading gradient direction data from disk failed";
	}

	for (int k = 0; k < nlayers; k++)
	{
		for (int i = 0; i < 2; i++)
		{
			struct mat4x4 Z;
			if (i == 0) {
				memcpy(Z.data, rZ.data, sizeof(Z.data));
			}
			else {
				project_unitary_tangent(&Vlist[k], &rZ, &Z);
			}

			for (int uproj = 0; uproj < 2; uproj++)
			{
				struct mat4x4 dVlist[4];
				if (brickwall_unitary_hess_matfree(Vlist, nlayers, L, &Z, k, Ufunc, NULL, pperms, uproj, dVlist) < 0) {
					return "'brickwall_unitary_hess_matfree' failed internally";
				}

				char filename[1024];
				sprintf(filename, "../test/data/test_brickwall_unitary_hess_matfree" CDATA_LABEL "_dVlist%i%i%s.dat", k, i, uproj == 1 ? "proj" : "");
				struct mat4x4 dVlist_ref[4];
				if (read_data(filename, dVlist_ref, sizeof(numeric), nlayers * 16) < 0) {
					return "reading reference Hessian data from disk failed";
				}

				// compare with reference
				for (int j = 0; j < nlayers; j++) {
					if (uniform_distance(16, dVlist[j].data, dVlist_ref[j].data) > 1e-12) {
						return "computed brickwall circuit Hessian does not match reference";
					}
				}
			}
		}
	}

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_hessian_matrix_matfree()
{
	int L = 6;
	int nlayers = 5;

	struct mat4x4 Vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_hessian_matrix_matfree_V%i.dat", i);
		if (read_data(filename, Vlist[i].data, sizeof(numeric), 16) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char filename[1024];
		sprintf(filename, "../test/data/test_brickwall_unitary_hessian_matrix_matfree_perm%i.dat", i);
		if (read_data(filename, perms[i], sizeof(int), L) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	const int m = nlayers * 16;
	double* H = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H == NULL) {
		return "memory allocation for Hessian matrix failed";
	}
	if (brickwall_unitary_hessian_matrix_matfree(Vlist, nlayers, L, Ufunc, NULL, pperms, H) < 0) {
		return "'brickwall_unitary_hessian_matrix_matfree' failed internally";
	}

	// check symmetry
	double es = 0;
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			es = fmax(es, fabs(H[i * m + j] - H[j * m + i]));
		}
	}
	if (es > 1e-13) {
		return "computed brickwall circuit Hessian matrix is not symmetric";
	}

	double* H_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H_ref == NULL) {
		return "memory allocation for reference Hessian matrix failed";
	}
	if (read_data("../test/data/test_brickwall_unitary_hessian_matrix_matfree_H.dat", H_ref, sizeof(double), m * m) < 0) {
		return "reading reference Hessian matrix from disk failed";
	}

	// compare with reference
	double d = 0;
	for (int i = 0; i < m * m; i++)
	{
		d = fmax(d, fabs(H[i] - H_ref[i]));
	}
	if (d > 1e-14) {
		return "computed brickwall circuit Hessian matrix does not match reference";
	}

	aligned_free(H_ref);
	aligned_free(H);

	return 0;
}

#endif
