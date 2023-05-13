#pragma once

#include "matrix.h"
#include "statevector.h"


int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);

int apply_adjoint_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);
