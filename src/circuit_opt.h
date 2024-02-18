#pragma once

#include "quantum_circuit.h"
#include "target.h"
#include "trust_region.h"


void optimize_quantum_circuit(linear_func ufunc, void* udata, const struct mat4x4 gates_start[], const int ngates, const int nqubits, const int wires[], struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 gates_opt[]);
