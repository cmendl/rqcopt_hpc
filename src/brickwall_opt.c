#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "target.h"
#include "brickwall_opt.h"


#ifdef COMPLEX_CIRCUIT


struct f_target_data
{
	unitary_func ufunc;
	void* udata;
	int nlayers;
	int nqubits;
	const int** perms;
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of target function evaluation.
///
static double f(const double* x, void* fdata)
{
	struct f_target_data* data = fdata;

	double f;
	if (target(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, &f) < 0) {
		fprintf(stderr, "target function evaluation failed internally\n");
		return -1;
	}

	return f;
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of target function, gradient and Hessian evaluation.
///
static double f_deriv(const double* restrict x, void* fdata, double* restrict grad, double* restrict hess)
{
	struct f_target_data* data = fdata;

	double f;
	if (target_gradient_vector_hessian_matrix(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, &f, grad, hess) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

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
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// the unitary matrix `U` using a trust-region method.
///
void optimize_brickwall_circuit_matfree(const int L, unitary_func ufunc, void* udata, const struct mat4x4 Vlist_start[], const int nlayers, const int* perms[], struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 Vlist_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.ufunc = ufunc,
		.udata = udata,
		.nlayers = nlayers,
		.nqubits = L,
		.perms = perms,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;
	riemannian_trust_region_optimize(f, f_deriv, &fdata, retract_unitary_list, &rdata,
		nlayers * 16, (const double*)Vlist_start, nlayers * 16 * 2, params, niter, f_iter, (double*)Vlist_opt);
}


#endif
