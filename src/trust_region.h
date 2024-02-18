#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Parameters for the truncated CG (tCG) method.
///
struct truncated_cg_params
{
	int maxiter;    //!< maximum number of iterations
	double abstol;  //!< absolute tolerance (for early stopping)
	double reltol;  //!< relative tolerance (for early stopping)

};


//________________________________________________________________________________________________________________________
///
/// \brief Set default parameters for the truncated CG (tCG) method.
///
static inline void set_truncated_cg_default_params(int n, struct truncated_cg_params* params)
{
	params->maxiter = 2 * n;
	params->abstol  = 1e-8;
	params->reltol  = 1e-6;
}


bool truncated_cg_old(const double* grad, const double* hess, int n, double radius, const struct truncated_cg_params* params, double* z);


typedef void (*hessian_vector_product_func)(const double* restrict x, void* fdata, double* restrict vec, double* restrict hvp);

bool truncated_cg(const double* restrict x, const double* restrict grad, const hessian_vector_product_func f_hvp, void* fdata, const int n, const double radius, const struct truncated_cg_params* params, double* restrict z);


typedef double (*target_func)(const double* x, void* fdata);

typedef double (*target_gradient_func)(const double* restrict x, void* fdata, double* restrict grad);

typedef double (*target_gradient_hessian_func)(const double* restrict x, void* fdata, double* restrict grad, double* restrict hess);

typedef void (*retract_func)(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs);


//________________________________________________________________________________________________________________________
///
/// \brief Parameters for the Riemannian trust-region (RTR) algorithm.
///
struct rtr_params
{
	struct truncated_cg_params tcg_params;  //!< parameters for the internally used truncated CG method
	double rho_trust;                       //!< threshold for accepting next candidate point
	double radius_init;                     //!< initial radius
	double maxradius;                       //!< maximum radius
	target_func g_func;                     //!< optional user-provided function, called at each iteration (unless set to NULL)
	void*       g_data;                     //!< additional arguments of 'gfunc'
	double*     g_iter;                     //!< array to store evaluations of 'gfunc', of length 'niter + 1'
};


//________________________________________________________________________________________________________________________
///
/// \brief Set default parameters for the Riemannian trust-region (RTR) algorithm.
///
static inline void set_rtr_default_params(int n, struct rtr_params* params)
{
	set_truncated_cg_default_params(n, &params->tcg_params);
	params->rho_trust   = 0.125;
	params->radius_init = 0.01;
	params->maxradius   = 0.1;
	params->g_func      = NULL;
	params->g_data      = NULL;
	params->g_iter      = NULL;
}


void riemannian_trust_region_optimize_old(target_func f, target_gradient_hessian_func f_deriv, void* fdata, retract_func retract, void* rdata,
	const int n, const double* x_init, const int m, const struct rtr_params* params, const int niter, double* f_iter, double* x_final);

void riemannian_trust_region_optimize(const target_func f, const target_gradient_func f_deriv, const hessian_vector_product_func f_hvp, void* fdata,
	retract_func retract, void* rdata,
	const int n, const double* x_init, const int m, const struct rtr_params* params, const int niter, double* f_iter, double* x_final);
