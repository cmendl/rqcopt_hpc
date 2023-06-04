#pragma once

#include "brickwall_circuit.h"
#include "trust_region.h"


#ifdef COMPLEX_CIRCUIT

void optimize_brickwall_circuit_matfree(int L, unitary_func Ufunc, void* udata,
	const struct mat4x4 Vlist_start[], int nlayers, const int* perms[],
	struct rtr_params* params, double* f_iter, double* g_iter, struct mat4x4 Vlist_opt[]);

#endif
