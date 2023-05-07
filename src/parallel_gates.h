#pragma once

#include "gate.h"
#include "statevector.h"


void apply_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);

int apply_parallel_gates(const struct two_qubit_gate* V, const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);

int apply_parallel_gates_directed_grad(const struct two_qubit_gate* V, const struct two_qubit_gate* Z,
	const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);
