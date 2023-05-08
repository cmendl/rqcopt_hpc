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


void transpose_statevector(const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_trans);
