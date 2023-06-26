#pragma once

#include "matrix.h"
#include "statevector.h"


typedef int (*unitary_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


int target_and_gradient(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[]);
