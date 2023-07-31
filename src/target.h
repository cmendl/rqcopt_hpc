#pragma once

#include "matrix.h"
#include "statevector.h"


typedef int (*linear_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


int unitary_target(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval);


int unitary_target_and_gradient(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[]);

#ifdef COMPLEX_CIRCUIT
int unitary_target_and_gradient_vector(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec);
#endif


int unitary_target_gradient_hessian(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[], numeric* hess);

#ifdef COMPLEX_CIRCUIT
int unitary_target_gradient_vector_hessian_matrix(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec, double* H);
#endif
