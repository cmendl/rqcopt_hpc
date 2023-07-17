#pragma once

#include "matrix.h"
#include "statevector.h"


typedef int (*unitary_func)(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out);


int target(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval);


int target_and_gradient(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[]);

#ifdef COMPLEX_CIRCUIT
int target_and_gradient_vector(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec);
#endif


int target_gradient_hessian(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[], numeric* hess);

#ifdef COMPLEX_CIRCUIT
int target_gradient_vector_hessian_matrix(unitary_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec, double* H);
#endif
