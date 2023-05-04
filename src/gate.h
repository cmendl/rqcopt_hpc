#pragma once

#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Single-qubit gate, storing the entries of a 2x2 matrix in row-major order.
///
struct single_qubit_gate
{
	numeric data[4];  //!< 2x2 matrix entries
};


//________________________________________________________________________________________________________________________
///
/// \brief Two-qubit gate, storing the entries of a 4x4 matrix in row-major order.
///
struct two_qubit_gate
{
	numeric data[16];  //!< 4x4 matrix entries
};
