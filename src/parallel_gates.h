#pragma once

#include <stdbool.h>
#include "matrix.h"
#include "statevector.h"


typedef int (*unitary_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


void apply_gate(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);

int apply_parallel_gates(const struct mat4x4* V, const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);

int apply_parallel_gates_directed_grad(const struct mat4x4* V, const struct mat4x4* Z,
	const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out);

int parallel_gates_grad_matfree(const struct mat4x4* restrict V, int L, unitary_func Ufunc, void* fdata, const int* perm, struct mat4x4* restrict G);

int parallel_gates_hess_matfree(const struct mat4x4* restrict V, int L, const struct mat4x4* restrict Z, unitary_func Ufunc, void* fdata, const int* perm, bool unitary_proj, struct mat4x4* restrict G);
