#pragma once

#include "matrix.h"
#include "statevector.h"
#include "quantum_circuit.h"


int apply_brickwall_unitary(const struct mat4x4 vlist[], const int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct statevector* restrict psi_out);

int brickwall_unitary_forward(const struct mat4x4 vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out);

int brickwall_unitary_backward(const struct mat4x4 vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dvlist[]);

int brickwall_unitary_backward_hessian(const struct mat4x4 vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dvlist[], numeric* hess);


int apply_brickwall_unitary_gate_placeholder(const struct mat4x4 vlist[], int nlayers, const int* perms[], int l,
	const struct quantum_circuit_cache* cache, struct statevector_array* restrict psi_out);
