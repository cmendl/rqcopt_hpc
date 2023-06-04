#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "brickwall_opt.h"


#ifdef COMPLEX_CIRCUIT


struct f_target_data
{
	int L;
	unitary_func Ufunc;
	void* udata;
	int nlayers;
	const int** perms;
};


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function,
/// using the provided matrix-free application of U to a state.
///
static double f_target_matfree(const double* restrict x, void* fdata)
{
	struct f_target_data* data = fdata;

	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(data->L, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits in 'f_target_matfree' failed\n", data->L);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(data->L, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits in 'f_target_matfree' failed\n", data->L);
		return -1;
	}
	struct statevector Vpsi = { 0 };
	if (allocate_statevector(data->L, &Vpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits in 'f_target_matfree' failed\n", data->L);
		return -1;
	}

	double f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << data->L;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = data->Ufunc(&psi, data->udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'Ufunc' in 'f_target_matfree' failed, return value: %i\n", ret);
			return -1;
		}

		ret = apply_brickwall_unitary((const struct mat4x4*)x, data->nlayers, &psi, data->perms, &Vpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_brickwall_unitary' in 'f_target_matfree' failed, return value: %i\n", ret);
			return -1;
		}

		// f -= Re <Vpsi | Upsi>
		for (intqs a = 0; a < n; a++)
		{
			f -= creal(conj(Vpsi.data[a]) * Upsi.data[a]);
		}
	}

	free_statevector(&Vpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	return f;
}


//________________________________________________________________________________________________________________________
///
/// \brief Retraction, with tangent direction represented as anti-symmetric matrices.
///
static void retract_unitary_list(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs)
{
	const int nlayers = *((int*)rdata);
	assert(nlayers > 0);

	const struct mat4x4* Vlist = (const struct mat4x4*)x;
	struct mat4x4* Vslist = (struct mat4x4*)xs;

	for (int j = 0; j < nlayers; j++)
	{
		struct mat4x4 Z;
		real_to_antisymm(&eta[j * 16], &Z);
		// add identity matrix
		Z.data[ 0]++;
		Z.data[ 5]++;
		Z.data[10]++;
		Z.data[15]++;

		struct mat4x4 W;
		multiply_matrices(&Vlist[j], &Z, &W);

		polar_factor(&W, &Vslist[j]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the gradient corresponding to the target function.
///
static void gradfunc_matfree(const double* restrict x, void* gdata, double* restrict grad)
{
	struct f_target_data* data = gdata;

	int ret = brickwall_unitary_gradient_vector_matfree((const struct mat4x4*)x, data->nlayers, data->L, data->Ufunc, data->udata, data->perms, grad);
	if (ret < 0) {
		fprintf(stderr, "call of 'brickwall_unitary_gradient_vector_matfree' in 'gradfunc_matfree' failed, return value: %i\n", ret);
	}

	// flip the sign of the gradient vector
	for (int i = 0; i < data->nlayers * 16; i++)
	{
		grad[i] = -grad[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the Hessian corresponding to the target function.
///
static void hessfunc_matfree(const double* restrict x, void* hdata, double* restrict hess)
{
	struct f_target_data* data = hdata;

	int ret = brickwall_unitary_hessian_matrix_matfree((const struct mat4x4*)x, data->nlayers, data->L, data->Ufunc, data->udata, data->perms, hess);
	if (ret < 0) {
		fprintf(stderr, "call of 'brickwall_unitary_hessian_matrix_matfree' in 'hessfunc_matfree' failed, return value: %i\n", ret);
	}

	// flip the sign of the Hessian matrix
	const int m = data->nlayers * 16;
	for (int i = 0; i < m * m; i++)
	{
		hess[i] = -hess[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// the unitary matrix `U` using a trust-region method.
///
void optimize_brickwall_circuit_matfree(int L, unitary_func Ufunc, void* udata, const struct mat4x4 Vlist_start[], int nlayers, const int* perms[], struct rtr_params* params, double* f_iter, double* g_iter, struct mat4x4 Vlist_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.L = L,
		.Ufunc = Ufunc,
		.udata = udata,
		.nlayers = nlayers,
		.perms = perms,
	};

	// TODO: quantify error by spectral norm
	params->gfunc = NULL;
	params->gdata = NULL;

	// perform optimization
	riemannian_trust_region_optimize(f_target_matfree, &fdata, retract_unitary_list, &nlayers,
		gradfunc_matfree, &fdata, hessfunc_matfree, &fdata, nlayers * 16,
		(const double*)Vlist_start, nlayers * 16 * 2, params, f_iter, g_iter, (double*)Vlist_opt);
}


#endif
