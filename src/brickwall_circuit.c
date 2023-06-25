#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "brickwall_circuit.h"
#include "gate.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers, const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	// temporary statevector
	struct statevector chi = { 0 };
	if (nlayers > 1) {
		if (allocate_statevector(psi->nqubits, &chi) < 0) {
			return -1;
		}
	}

	for (int i = 0; i < nlayers; i++)
	{
		int p = (nlayers - i) % 2;
		const struct statevector* psi0 = (i == 0 ? psi : (p == 0 ? psi_out : &chi));
		struct statevector* psi1 = (p == 0 ? &chi : psi_out);
		int ret = apply_parallel_gates(&Vlist[i], psi0, perms[i], psi1);
		if (ret < 0) {
			if (nlayers > 1) {
				free_statevector(&chi);
			}
			return ret;
		}
	}

	if (nlayers > 1) {
		free_statevector(&chi);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the adjoint unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_adjoint_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers, const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	// temporary statevector
	struct statevector chi = { 0 };
	if (nlayers > 1) {
		if (allocate_statevector(psi->nqubits, &chi) < 0) {
			return -1;
		}
	}

	for (int i = nlayers - 1; i >= 0; i--)
	{
		int p = i % 2;
		const struct statevector* psi0 = (i == nlayers - 1 ? psi : (p == 0 ? &chi : psi_out));
		struct statevector* psi1 = (p == 0 ? psi_out : &chi);
		struct mat4x4 Vh;
		adjoint(&Vlist[i], &Vh);
		int ret = apply_parallel_gates(&Vh, psi0, perms[i], psi1);
		if (ret < 0) {
			if (nlayers > 1) {
				free_statevector(&chi);
			}
			return ret;
		}
	}

	if (nlayers > 1) {
		free_statevector(&chi);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate temporary cache required by backward pass of a brick wall quantum circuit.
///
int allocate_brickwall_unitary_cache(const int nlayers, const int nqubits, struct brickwall_unitary_cache* cache)
{
	assert(nlayers >= 1);
	assert(nqubits % 2 == 0);

	cache->nlayers = nlayers;
	cache->nqubits = nqubits;

	const int nstates = nlayers * (nqubits / 2);
	cache->psi_list = aligned_alloc(MEM_DATA_ALIGN, nstates * sizeof(struct statevector));
	if (cache->psi_list == NULL) {
		fprintf(stderr, "allocating memory for statevector array failed\n");
		return -1;
	}

	for (int i = 0; i < nstates; i++)
	{
		int ret = allocate_statevector(nqubits, &cache->psi_list[i]);
		if (ret < 0) {
			return ret;
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Free temporary cache required by backward pass of a brick wall quantum circuit.
///
void free_brickwall_unitary_cache(struct brickwall_unitary_cache* cache)
{
	const int nstates = cache->nlayers * (cache->nqubits / 2);

	for (int i = 0; i < nstates; i++)
	{
		free_statevector(&cache->psi_list[i]);
	}

	aligned_free(cache->psi_list);
	cache->psi_list = NULL;

	cache->nqubits = 0;
	cache->nlayers = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int brickwall_unitary_forward(const struct mat4x4 Vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct brickwall_unitary_cache* cache, struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);
	assert(cache->nlayers == nlayers);
	assert(cache->nqubits == psi->nqubits);
	assert(psi->nqubits % 2 == 0);

	// store initial statevector in cache as well
	memcpy(cache->psi_list[0].data, psi->data, ((size_t)1 << psi->nqubits) * sizeof(numeric));

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(int));
	if (inv_perm == NULL) {
		fprintf(stderr, "allocating permutation vector failed\n");
		return -1;
	}

	const int nstates = nlayers * (psi->nqubits / 2);

	int k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		inverse_permutation(psi->nqubits, perms[i], inv_perm);

		for (int j = 0; j < psi->nqubits; j += 2)
		{
			apply_gate(&Vlist[i], inv_perm[j], inv_perm[j + 1], &cache->psi_list[k],
				(k + 1 < nstates ? &cache->psi_list[k + 1] : psi_out));
			k++;
		}
	}

	aligned_free(inv_perm);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int brickwall_unitary_backward(const struct mat4x4 Vlist[], int nlayers, const int* perms[], const struct brickwall_unitary_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dVlist[])
{
	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(cache->nlayers == nlayers);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->nqubits % 2 == 0);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(dpsi->nqubits, &tmp) < 0) {
		return -1;
	}

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, dpsi->nqubits * sizeof(int));
	if (inv_perm == NULL) {
		fprintf(stderr, "allocating permutation vector failed\n");
		return -1;
	}

	const int nstates = nlayers * (dpsi->nqubits / 2);

	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		inverse_permutation(dpsi->nqubits, perms[i], inv_perm);

		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			const struct statevector* dpsi1 = ((k == nstates - 1) ? dpsi_out : (k % 2 == 1 ? dpsi : &tmp));
			struct statevector* dpsi0 = (k % 2 == 1 ? &tmp : dpsi);

			struct mat4x4 dV;
			apply_gate_backward(&Vlist[i], inv_perm[j], inv_perm[j + 1], &cache->psi_list[k], dpsi1, dpsi0, &dV);
			// accumulate gradient
			add_matrix(&dVlist[i], &dV);

			k--;
		}
	}

	aligned_free(inv_perm);
	free_statevector(&tmp);

	return 0;
}


struct ufunc_parallel_gates_data
{
	unitary_func Ufunc;
	void* fdata;
	int nlayers;
	const struct mat4x4* Vlist;
	const int** perms;
	int j;
};


static int ufunc_parallel_gates(const struct statevector* restrict psi, void* fpgdata, struct statevector* restrict psi_out)
{
	struct ufunc_parallel_gates_data* data = fpgdata;
	assert(0 <= data->j && data->j < data->nlayers);

	// temporary statevector
	struct statevector chi = { 0 };
	if (data->nlayers > 1) {
		if (allocate_statevector(psi->nqubits, &chi) < 0) {
			return -1;
		}
	}

	int p = 0;
	if (data->j > 0) { p++; }
	if (data->nlayers - data->j - 1 > 0) { p++; }
	struct statevector* psi0 = (p % 2 == 0 ? &chi : psi_out);
	struct statevector* psi1 = (p % 2 == 1 ? &chi : psi_out);

	if (data->j > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist, data->j, psi, data->perms, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	// apply U
	{
		int ret = data->Ufunc(data->j > 0 ? psi0 : psi, data->fdata, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->nlayers - data->j - 1 > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist + (data->j + 1), data->nlayers - data->j - 1, psi0, data->perms + (data->j + 1), psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->nlayers > 1) {
		free_statevector(&chi);
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the gradient of Re tr[U^{\dagger} W] with respect to Vlist,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_grad_matfree(const struct mat4x4 Vlist[], int nlayers, int L, unitary_func Ufunc, void* fdata, const int* perms[], struct mat4x4 Glist[])
{
	struct ufunc_parallel_gates_data data = {
		.Ufunc = Ufunc,
		.fdata = fdata,
		.nlayers = nlayers,
		.Vlist = Vlist,
		.perms = perms,
		.j = -1,
	};

	for (int j = 0; j < nlayers; j++)
	{
		data.j = j;

		int ret = parallel_gates_grad_matfree(&Vlist[j], L, ufunc_parallel_gates, &data, perms[j], &Glist[j]);
		if (ret < 0) {
			return ret;
		}
	}

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Represent the gradient of Re tr[U^{\dagger} W] as real vector,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_gradient_vector_matfree(const struct mat4x4 Vlist[], int nlayers, int L, unitary_func Ufunc, void* fdata, const int* perms[], double* grad_vec)
{
	struct mat4x4* Glist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (Glist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	int ret = brickwall_unitary_grad_matfree(Vlist, nlayers, L, Ufunc, fdata, perms, Glist);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix
	// and then convert to a vector
	for (int j = 0; j < nlayers; j++)
	{
		struct mat4x4 W;
		adjoint(&Vlist[j], &W);
		struct mat4x4 T;
		multiply_matrices(&W, &Glist[j], &T);
		antisymm(&T, &W);
		antisymm_to_real(&W, &grad_vec[j * 16]);
	}

	aligned_free(Glist);

	return 0;
}

#endif


struct ufunc_parallel_gates_grad_data
{
	unitary_func Ufunc;
	void* fdata;
	int nlayers;
	const struct mat4x4* Vlist;
	const struct mat4x4* Z;
	const int** perms;
	int j, k;
};


static int ufunc_parallel_gates_grad_1(const struct statevector* restrict psi, void* fpggdata, struct statevector* restrict psi_out)
{
	struct ufunc_parallel_gates_grad_data* data = fpggdata;
	assert(0 <= data->j && data->j < data->k && data->k < data->nlayers);

	// temporary statevector
	struct statevector chi = { 0 };
	if (allocate_statevector(psi->nqubits, &chi) < 0) {
		return -1;
	}

	int p = 0;
	if (data->j > 0) { p++; }
	if (data->nlayers - data->k - 1 > 0) { p++; }
	if (data->k - data->j - 1 > 0) { p++; }
	struct statevector* psi0 = (p % 2 == 1 ? &chi : psi_out);
	struct statevector* psi1 = (p % 2 == 0 ? &chi : psi_out);

	if (data->j > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist, data->j, psi, data->perms, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	// apply U
	{
		int ret = data->Ufunc(data->j > 0 ? psi0 : psi, data->fdata, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->nlayers - data->k - 1 > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist + (data->k + 1), data->nlayers - data->k - 1, psi0, data->perms + (data->k + 1), psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	// gradient direction Z for layer 'k'
	{
		struct mat4x4 Vkh;
		adjoint(&data->Vlist[data->k], &Vkh);
		struct mat4x4 Zh;
		adjoint(data->Z, &Zh);
		int ret = apply_parallel_gates_directed_grad(&Vkh, &Zh, psi0, data->perms[data->k], psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->k - data->j - 1 > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist + (data->j + 1), data->k - data->j - 1, psi0, data->perms + (data->j + 1), psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	free_statevector(&chi);

	return 0;
}


static int ufunc_parallel_gates_grad_2(const struct statevector* restrict psi, void* fpggdata, struct statevector* restrict psi_out)
{
	struct ufunc_parallel_gates_grad_data* data = fpggdata;
	assert(0 <= data->k && data->k < data->j && data->j < data->nlayers);

	// temporary statevector
	struct statevector chi = { 0 };
	if (allocate_statevector(psi->nqubits, &chi) < 0) {
		return -1;
	}

	int p = 0;
	if (data->j - data->k - 1 > 0) { p++; }
	if (data->k > 0) { p++; }
	if (data->nlayers - data->j - 1 > 0) { p++; }
	struct statevector* psi0 = (p % 2 == 1 ? &chi : psi_out);
	struct statevector* psi1 = (p % 2 == 0 ? &chi : psi_out);
	
	if (data->j - data->k - 1 > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist + (data->k + 1), data->j - data->k - 1, psi, data->perms + (data->k + 1), psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	// gradient direction Z for layer 'k'
	{
		struct mat4x4 Vkh;
		adjoint(&data->Vlist[data->k], &Vkh);
		struct mat4x4 Zh;
		adjoint(data->Z, &Zh);
		int ret = apply_parallel_gates_directed_grad(&Vkh, &Zh, data->j - data->k - 1 > 0 ? psi0 : psi, data->perms[data->k], psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->k > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist, data->k, psi0, data->perms, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	// apply U
	{
		int ret = data->Ufunc(psi0, data->fdata, psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	if (data->nlayers - data->j - 1 > 0)
	{
		int ret = apply_adjoint_brickwall_unitary(data->Vlist + (data->j + 1), data->nlayers - data->j - 1, psi0, data->perms + (data->j + 1), psi1);
		if (ret < 0) {
			return ret;
		}
		// swap psi0 <-> psi1 pointers
		struct statevector* tmp = psi0;
		psi0 = psi1;
		psi1 = tmp;
	}

	free_statevector(&chi);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Hessian of Re tr[U^{\dagger} W] in direction Z with respect to Vlist[k],
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_hess_matfree(const struct mat4x4 Vlist[], int nlayers, int L, const struct mat4x4* Z, int k, unitary_func Ufunc, void* fdata, const int* perms[], bool unitary_proj, struct mat4x4 dVlist[])
{
	assert(0 <= k && k < nlayers);

	for (int j = 0; j < nlayers; j++)
	{
		memset(&dVlist[j], 0, sizeof(struct mat4x4));
	}

	struct ufunc_parallel_gates_grad_data gdata = {
		.Ufunc = Ufunc,
		.fdata = fdata,
		.nlayers = nlayers,
		.Vlist = Vlist,
		.Z = Z,
		.perms = perms,
		.j = -1,
		.k = k,
	};

	for (int j = 0; j < k; j++)
	{
		// j < k
		// directed gradient with respect to Vlist[k] in direction Z
		gdata.j = j;
		struct mat4x4 dVj;
		int ret = parallel_gates_grad_matfree(&Vlist[j], L, ufunc_parallel_gates_grad_1, &gdata, perms[j], &dVj);
		if (ret < 0) {
			return ret;
		}
		if (unitary_proj)
		{
			struct mat4x4 dVjproj;
			project_unitary_tangent(&Vlist[j], &dVj, &dVjproj);
			add_matrix(&dVlist[j], &dVjproj);
		}
		else
		{
			add_matrix(&dVlist[j], &dVj);
		}
	}

	// Hessian for layer k
	{
		struct ufunc_parallel_gates_data hdata = {
			.Ufunc = Ufunc,
			.fdata = fdata,
			.nlayers = nlayers,
			.Vlist = Vlist,
			.perms = perms,
			.j = k,
		};
		struct mat4x4 dVk;
		int ret = parallel_gates_hess_matfree(&Vlist[k], L, Z, ufunc_parallel_gates, &hdata, perms[k], unitary_proj, &dVk);
		if (ret < 0) {
			return ret;
		}
		add_matrix(&dVlist[k], &dVk);
	}

	for (int j = k + 1; j < nlayers; j++)
	{
		// k < j
		// directed gradient with respect to Vlist[k] in direction Z
		gdata.j = j;
		struct mat4x4 dVj;
		int ret = parallel_gates_grad_matfree(&Vlist[j], L, ufunc_parallel_gates_grad_2, &gdata, perms[j], &dVj);
		if (ret < 0) {
			return ret;
		}
		if (unitary_proj)
		{
			struct mat4x4 dVjproj;
			project_unitary_tangent(&Vlist[j], &dVj, &dVjproj);
			add_matrix(&dVlist[j], &dVjproj);
		}
		else
		{
			add_matrix(&dVlist[j], &dVj);
		}
	}

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Construct the Hessian matrix of Re tr[U^{\dagger} W] with respect to Vlist
/// defining the layers of W,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_hessian_matrix_matfree(const struct mat4x4 Vlist[], int nlayers, int L, unitary_func Ufunc, void* fdata, const int* perms[], double* H)
{
	int m = nlayers * 16;
	memset(H, 0, m * m * sizeof(double));

	struct mat4x4* dVZj = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

	for (int j = 0; j < nlayers; j++)
	{
		for (int k = 0; k < 16; k++)
		{
			// unit vector
			double r[16] = { 0 };
			r[k] = 1;
			struct mat4x4 Ek;
			real_to_antisymm(r, &Ek);
			// Z = Vlist[j] @ Ek
			struct mat4x4 Z;
			multiply_matrices(&Vlist[j], &Ek, &Z);
			int ret = brickwall_unitary_hess_matfree(Vlist, nlayers, L, &Z, j, Ufunc, fdata, perms, true, dVZj);
			if (ret < 0) {
				return ret;
			}

			for (int i = 0; i < nlayers; i++)
			{
				// Vlist[i]^{\dagger} @ dVZj[i])
				struct mat4x4 W;
				adjoint(&Vlist[i], &W);
				struct mat4x4 T;
				multiply_matrices(&W, &dVZj[i], &T);
				antisymm(&T, &W);
				double h[16];
				antisymm_to_real(&W, h);
				for (int l = 0; l < 16; l++)
				{
					H[((i * 16 + l) * nlayers + j) * 16 + k] = h[l];
				}
			}
		}
	}

	aligned_free(dVZj);

	return 0;
}

#endif
