#pragma once

#include "brickwall_circuit.h"
#include "trust_region.h"


typedef int (*unitary_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


#ifdef COMPLEX_CIRCUIT

void optimize_brickwall_circuit_matfree(const int L, unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist_start[], const int nlayers, const int* perms[],
	struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 Vlist_opt[]);

#endif
