#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "brickwall_opt.h"


struct f_target_data
{
	linear_func ufunc;
	void* udata;
	const int** perms;
	int nlayers;
	int nqubits;
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of target function evaluation.
///
static double f(const double* x, void* fdata)
{
	struct f_target_data* data = fdata;

	numeric f;
	if (brickwall_unitary_target(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, &f) < 0) {
		fprintf(stderr, "target function evaluation failed internally\n");
		return -1;
	}

	return creal(f);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of brickwall circuit target function and gradient evaluation.
///
static double f_deriv(const double* restrict x, void* fdata, double* restrict grad)
{
	struct f_target_data* data = fdata;

	numeric fval;
	if (brickwall_unitary_target_and_projected_gradient(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, &fval, grad) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

	return creal(fval);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of Hessian-vector product evaluation for brickwall circuit target function.
///
static void f_hvp(const double* restrict x, void* fdata, double* restrict vec, double* restrict hvp)
{
	const struct mat4x4* vlist = (const struct mat4x4*)x;

	const struct f_target_data* data = fdata;

	// convert input vectors to matrices in tangent space
	struct mat4x4* vdirs = aligned_malloc(data->nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < data->nlayers; i++)
	{
		real_to_tangent(&vec[i * num_tangent_params], &vlist[i], &vdirs[i]);
	}

	numeric fval;
	double* grad_vec = aligned_malloc(data->nlayers * num_tangent_params * sizeof(double));

	brickwall_unitary_target_projected_hessian_vector_product(data->ufunc, data->udata, vlist, vdirs, data->nlayers, data->perms, data->nqubits, &fval, grad_vec, hvp);

	aligned_free(grad_vec);
	aligned_free(vdirs);
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of brickwall circuit target function, gradient and Hessian evaluation.
///
static double f_deriv_hess(const double* restrict x, void* fdata, double* restrict grad, double* restrict hess)
{
	struct f_target_data* data = fdata;

	numeric f;
	if (brickwall_unitary_target_gradient_vector_hessian_matrix(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, &f, grad, hess) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

	return creal(f);
}

#endif


//________________________________________________________________________________________________________________________
///
/// \brief Retraction, with tangent direction represented as anti-symmetric matrices.
///
static void retract_unitary_list(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs)
{
	const int nlayers = *((int*)rdata);
	assert(nlayers > 0);

	const struct mat4x4* vlist  = (const struct mat4x4*)x;
	struct mat4x4* retractvlist = (struct mat4x4*)xs;

	for (int i = 0; i < nlayers; i++)
	{
		retract(&vlist[i], &eta[i * num_tangent_params], &retractvlist[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// a unitary matrix `U` using the Riemannian trust-region method.
///
void optimize_brickwall_circuit_hvp(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.ufunc   = ufunc,
		.udata   = udata,
		.perms   = perms,
		.nlayers = nlayers,
		.nqubits = nqubits,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;
	riemannian_trust_region_optimize_hvp(f, f_deriv, f_hvp, &fdata, retract_unitary_list, &rdata,
		nlayers * num_tangent_params, (const double*)vlist_start, nlayers * 16 * (sizeof(numeric)/sizeof(double)), params, niter, f_iter, (double*)vlist_opt);
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// a unitary matrix `U` using the Riemannian trust-region method.
///
void optimize_brickwall_circuit_hmat(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.ufunc   = ufunc,
		.udata   = udata,
		.perms   = perms,
		.nlayers = nlayers,
		.nqubits = nqubits,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;
	riemannian_trust_region_optimize_hmat(f, f_deriv_hess, &fdata, retract_unitary_list, &rdata,
		nlayers * 16, (const double*)vlist_start, nlayers * 16 * 2, params, niter, f_iter, (double*)vlist_opt);
}

#endif


#ifdef COMPLEX_CIRCUIT


struct f_target_data_sampling
{
	linear_func ufunc;
	void* udata;
	const int** perms;
	struct rng_state* rng;
	long nsamples;
	int nlayers;
	int nqubits;
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of target function evaluation using state vector sampling.
///
static double f_sampling(const double* x, void* fdata)
{
	struct f_target_data_sampling* data = fdata;

	numeric f;
	if (brickwall_unitary_target_sampling(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, data->nsamples, data->rng, &f) < 0) {
		fprintf(stderr, "target function evaluation failed internally\n");
		return -1;
	}

	return creal(f);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of brickwall circuit target function, gradient and Hessian evaluation using state vector sampling.
///
static double f_deriv_hess_sampling(const double* restrict x, void* fdata, double* restrict grad, double* restrict hess)
{
	struct f_target_data_sampling* data = fdata;

	numeric f;
	if (brickwall_unitary_target_gradient_vector_hessian_matrix_sampling(data->ufunc, data->udata, (const struct mat4x4*)x, data->nlayers, data->nqubits, data->perms, data->nsamples, data->rng, &f, grad, hess) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

	return creal(f);
}


//________________________________________________________________________________________________________________________
///
/// \brief Optimize the quantum gates in a brickwall layout to approximate
/// a unitary matrix `U` using the Riemannian trust-region method using state vector sampling.
///
void optimize_brickwall_circuit_hmat_sampling(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	const long nsamples, struct rng_state* rng,
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[])
{
	// target function data
	struct f_target_data_sampling fdata = {
		.ufunc    = ufunc,
		.udata    = udata,
		.perms    = perms,
		.rng      = rng,
		.nsamples = nsamples,
		.nlayers  = nlayers,
		.nqubits  = nqubits,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = nlayers;
	riemannian_trust_region_optimize_hmat(f_sampling, f_deriv_hess_sampling, &fdata, retract_unitary_list, &rdata,
		nlayers * 16, (const double*)vlist_start, nlayers * 16 * 2, params, niter, f_iter, (double*)vlist_opt);
}

#endif
