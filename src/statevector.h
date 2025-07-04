#pragma once

#include "config.h"
#include "rng.h"


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


void normalize_statevector(struct statevector* restrict psi);


void transpose_statevector(const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_trans);


void haar_random_statevector(struct statevector* psi, struct rng_state* rng);


void add_statevectors(const struct statevector* restrict psi1, const struct statevector* restrict psi2, struct statevector* restrict psi_sum);


//________________________________________________________________________________________________________________________
///
/// \brief Array of quantum statevectors, using interleaved entries.
///
struct statevector_array
{
	numeric* data;  //!< data entries, matrix of size 2^nqubits x nstates
	int nqubits;    //!< number of qubits
	int nstates;    //!< number of statevectors
};


int allocate_statevector_array(int nqubits, int nstates, struct statevector_array* psi_list);

void free_statevector_array(struct statevector_array* psi_list);
