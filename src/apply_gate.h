#pragma once

#include "gate.h"
#include "statevector.h"


void apply_single_qubit_gate(const struct single_qubit_gate* gate, int i, const struct statevector* restrict psi, struct statevector* restrict psi_out);

void apply_two_qubit_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);
