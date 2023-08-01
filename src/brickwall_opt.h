#pragma once

#include "brickwall_circuit.h"
#include "target.h"
#include "trust_region.h"


#ifdef COMPLEX_CIRCUIT

void optimize_brickwall_circuit(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist_start[], const int nlayers, const int L, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 Vlist_opt[]);

#endif
