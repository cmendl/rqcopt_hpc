#pragma once

#include "matrix.h"
#include "statevector.h"


void apply_gate(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);

void apply_gate_backward(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4* dgate);


void apply_gate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out);
