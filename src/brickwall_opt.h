#pragma once

#include "brickwall_circuit.h"
#include "target.h"
#include "trust_region.h"


void optimize_brickwall_circuit_hvp(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[]);


#ifdef COMPLEX_CIRCUIT

void optimize_brickwall_circuit_hmat(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[]);

void optimize_brickwall_circuit_hmat_sampling(linear_func ufunc, void* udata,
	const struct mat4x4 vlist_start[], const int nlayers, const int nqubits, const int* perms[],
	const long nsamples, struct rng_state* rng,
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 vlist_opt[]);

#endif
