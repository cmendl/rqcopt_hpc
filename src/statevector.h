#pragma once

#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Quantum statevector representing a multi-qubit state.
///
struct statevector
{
	numeric* data;  //!< data entries, vector of length 2^nqubits
	int nqubits;    //!< number of qubits
};


int allocate_statevector(int nqubits, struct statevector* psi);

void free_statevector(struct statevector* psi);
