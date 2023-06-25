#include <memory.h>
#include <assert.h>
#include "config.h"
#include "matrix.h"
#include "statevector.h"
#include "brickwall_circuit.h"
#include "numerical_gradient.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_brickwall_unitary()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_apply_brickwall_unitary" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_brickwall_unitary failed";
	}

	struct statevector psi, chi, chi_ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi_ref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
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

		char varname[32];
		sprintf(varname, "chi%i", nlayers - 3);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, chi_ref.data) < 0) {
			return "reading output statevector data from disk failed";
		}

		// compare with reference
		if (uniform_distance((size_t)1 << L, chi.data, chi_ref.data) > 1e-12) {
			return "quantum state after applying gate does not match reference";
		}
	}

	free_statevector(&chi_ref);
	free_statevector(&chi);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


char* test_apply_adjoint_brickwall_unitary()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_apply_adjoint_brickwall_unitary" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_adjoint_brickwall_unitary failed";
	}

	struct statevector psi, chi, chiref;
	if (allocate_statevector(L, &psi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chiref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
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

		char varname[32];
		sprintf(varname, "chi%i", nlayers - 3);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, chiref.data) < 0) {
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

	H5Fclose(file);

	return 0;
}


struct brickwall_unitary_forward_psi_params
{
	int nqubits;
	int nlayers;
	const struct mat4x4* Vlist;
	const int** perms;
};

// wrapper of brickwall_unitary_forward as a function of 'psi'
static void brickwall_unitary_forward_psi(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_forward_psi_params* params = p;

	struct statevector psi;
	allocate_statevector(params->nqubits, &psi);
	memcpy(psi.data, x, ((size_t)1 << params->nqubits) * sizeof(numeric));

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	struct brickwall_unitary_cache cache;
	allocate_brickwall_unitary_cache(params->nlayers, params->nqubits, &cache);

	brickwall_unitary_forward(params->Vlist, params->nlayers, params->perms, &psi, &cache, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_brickwall_unitary_cache(&cache);
	free_statevector(&psi_out);
	free_statevector(&psi);
}

struct brickwall_unitary_forward_gates_params
{
	int nqubits;
	int nlayers;
	const struct statevector* psi;
	const int** perms;
};

// wrapper of brickwall_unitary_forward as a function of the gates
static void brickwall_unitary_forward_gates(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_forward_gates_params* params = p;

	struct mat4x4* Vlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < params->nlayers; i++) {
		memcpy(Vlist[i].data, &x[i * 16], sizeof(Vlist[i].data));
	}

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	struct brickwall_unitary_cache cache;
	allocate_brickwall_unitary_cache(params->nlayers, params->nqubits, &cache);

	brickwall_unitary_forward(Vlist, params->nlayers, params->perms, params->psi, &cache, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_brickwall_unitary_cache(&cache);
	free_statevector(&psi_out);
	aligned_free(Vlist);
}

char* test_brickwall_unitary_backward()
{
	int L = 6;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_backward" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_backward failed";
	}

	struct statevector psi, psi_out, psi_out_ref, dpsi_out, dpsi;
	if (allocate_statevector(L, &psi)         < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &psi_out_ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi_out)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &dpsi)        < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading upstream gradient data from disk failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][8];
	for (int i = 0; i < 4; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	for (int nlayers = 3; nlayers <= 4; nlayers++)
	{
		struct brickwall_unitary_cache cache;
		if (allocate_brickwall_unitary_cache(nlayers, L, &cache) < 0) {
			return "'allocate_brickwall_unitary_cache' failed";
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(Vlist, nlayers, pperms, &psi, &cache, &psi_out) < 0) {
			return "'brickwall_unitary_forward' failed internally";
		}

		char varname[32];
		sprintf(varname, "psi_out%i", nlayers - 3);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
			return "reading output statevector data from disk failed";
		}
		// compare output state of forward pass with reference
		if (uniform_distance((size_t)1 << L, psi_out.data, psi_out_ref.data) > 1e-12) {
			return "quantum state after applying brick wall quantum circuit does not match reference";
		}

		// brickwall unitary backward pass
		struct mat4x4 dVlist[4];
		if (brickwall_unitary_backward(Vlist, nlayers, pperms, &cache, &dpsi_out, &dpsi, dVlist) < 0) {
			return "'brickwall_unitary_backward' failed internally";
		}

		const double h = 1e-5;

		// numerical gradient with respect to 'psi'
		struct brickwall_unitary_forward_psi_params params_psi = {
			.nqubits = L,
			.nlayers = nlayers,
			.Vlist = Vlist,
			.perms = pperms,
		};
		struct statevector dpsi_num;
		if (allocate_statevector(L, &dpsi_num) < 0) { return "memory allocation failed"; }
		numerical_gradient(brickwall_unitary_forward_psi, &params_psi, (size_t)1 << L, psi.data, (size_t)1 << L, dpsi_out.data, h, dpsi_num.data);
		// compare
		if (uniform_distance((size_t)1 << L, dpsi.data, dpsi_num.data) > 1e-8) {
			return "gradient with respect to 'psi' computed by 'brickwall_unitary_backward' does not match finite difference approximation";
		}

		// numerical gradient with respect to gates
		struct brickwall_unitary_forward_gates_params params_gates = {
			.nqubits = L,
			.nlayers = nlayers,
			.psi = &psi,
			.perms = pperms,
		};
		struct mat4x4 dVlist_num[4];
		numerical_gradient(brickwall_unitary_forward_gates, &params_gates, nlayers * 16, (numeric*)Vlist, (size_t)1 << L, dpsi_out.data, h, (numeric*)dVlist_num);
		// compare
		if (uniform_distance(nlayers * 16, (numeric*)dVlist, (numeric*)dVlist_num) > 1e-8) {
			return "gradient with respect to gates computed by 'brickwall_unitary_backward' does not match finite difference approximation";
		}

		free_statevector(&dpsi_num);
		free_brickwall_unitary_cache(&cache);
	}

	free_statevector(&dpsi);
	free_statevector(&dpsi_out);
	free_statevector(&psi_out_ref);
	free_statevector(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

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

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_grad_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_grad_matfree failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
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

		char varname[32];
		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dVlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_ref[j].data) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_gradient_vector_matfree()
{
	int L = 6;
	int nlayers = 3;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_gradient_vector_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_gradient_vector_matfree failed";
	}

	struct mat4x4 Vlist[3];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[3][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };

	double grad[3 * 16];
	if (brickwall_unitary_gradient_vector_matfree(Vlist, nlayers, L, Ufunc, NULL, pperms, grad) < 0) {
		return "'brickwall_unitary_gradient_vector_matfree' failed internally";
	}

	double grad_ref[3 * 16];
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
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

	H5Fclose(file);

	return 0;
}

#endif


char* test_brickwall_unitary_hess_matfree()
{
	int L = 6;
	int nlayers = 4;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_hess_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_hess_matfree failed";
	}

	struct mat4x4 Vlist[4];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[4][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	// gradient direction
	struct mat4x4 rZ;
	if (read_hdf5_dataset(file, "rZ", H5T_NATIVE_DOUBLE, rZ.data) < 0) {
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

				char varname[32];
				sprintf(varname, "dVlist%i%i%s", k, i, uproj == 1 ? "proj" : "");
				struct mat4x4 dVlist_ref[4];
				if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVlist_ref) < 0) {
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

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_hessian_matrix_matfree()
{
	int L = 6;
	int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_hessian_matrix_matfree" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_hessian_matrix_matfree failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
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
	if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H_ref) < 0) {
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

	H5Fclose(file);

	return 0;
}

#endif
