#pragma once

#include <stdbool.h>


//________________________________________________________________________________________________________________________
///
/// \brief Parameters for the truncated CG (tCG) method.
///
struct truncated_cg_params
{
	int maxiter;
	double abstol;
	double reltol;
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


bool truncated_cg(const double* grad, const double* hess, int n, double radius, const struct truncated_cg_params* params, double* z);


typedef double (*target_func)(const double* restrict x, void* fdata);

typedef void (*retract_func)(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs);

typedef void (*gradient_func)(const double* restrict x, void* gdata, double* restrict grad);
typedef void  (*hessian_func)(const double* restrict x, void* hdata, double* restrict hess);


//________________________________________________________________________________________________________________________
///
/// \brief Parameters for the Riemannian trust-region (RTR) algorithm.
///
struct rtr_params
{
	struct truncated_cg_params tcg_params;
	double rho_trust;
	double radius_init;
	double maxradius;
	int niter;
	target_func gfunc;
	void* gdata;
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
	params->niter       = 20;
	params->gfunc       = NULL;
	params->gdata       = NULL;
}


void riemannian_trust_region_optimize(target_func f, void* fdata, retract_func retract, void* rdata,
	gradient_func gradfunc, void* graddata, hessian_func hessfunc, void* hessdata, const int n,
	const double* x_init, const int m, const struct rtr_params* params, double* f_iter, double* g_iter, double* x_final);
