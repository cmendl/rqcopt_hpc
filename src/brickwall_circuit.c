#include <stdio.h>
#include <assert.h>
#include "brickwall_circuit.h"


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


struct ufunc_parallel_gates_data
{
	unitary_func Ufunc;
	void* fdata;
	int nlayers;
	const struct mat4x4* Vlist;
	const int** perms;
	int L;
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
		.L = L,
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
		fprintf(stderr, "allocating temporary memory for gradient matrices failed");
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
		multiply(&W, &Glist[j], &T);
		antisymm(&T, &W);
		antisymm_to_real(&W, &grad_vec[j * 16]);
	}

	aligned_free(Glist);

	return 0;
}

#endif
