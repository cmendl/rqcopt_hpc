#pragma once

#include "matrix.h"
#include "statevector.h"
#include "parallel_gates.h"


int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);

int apply_adjoint_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);

int brickwall_unitary_grad_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	unitary_func Ufunc, void* fdata, const int* perms[], struct mat4x4 Glist[]);

#ifdef COMPLEX_CIRCUIT

int brickwall_unitary_gradient_vector_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	unitary_func Ufunc, void* fdata, const int* perms[], double* grad_vec);

#endif
