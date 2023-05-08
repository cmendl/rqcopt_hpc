#pragma once

#include "gate.h"
#include "statevector.h"


typedef int (*unitary_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


void apply_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);

int apply_parallel_gates(const struct two_qubit_gate* V, const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);

int apply_parallel_gates_directed_grad(const struct two_qubit_gate* V, const struct two_qubit_gate* Z,
	const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);

int parallel_gates_grad_matfree(const struct two_qubit_gate* V, int L, unitary_func Ufunc, void* fdata, const int* perm, struct two_qubit_gate* G);
