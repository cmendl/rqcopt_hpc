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
