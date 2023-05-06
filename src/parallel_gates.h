#pragma once

#include "gate.h"
#include "statevector.h"


void apply_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);
