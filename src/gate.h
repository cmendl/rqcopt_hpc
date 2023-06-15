#pragma once

#include "matrix.h"
#include "statevector.h"


void apply_gate(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out);
