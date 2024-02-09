#pragma once

#include "matrix.h"
#include "statevector.h"


int apply_quantum_circuit(const struct mat4x4 gates[], const int ngates, const int wires[],
	const struct statevector* restrict psi, struct statevector* restrict psi_out);


//________________________________________________________________________________________________________________________
///
/// \brief Temporary cache required by backward pass of a quantum circuit,
/// storing the sequence of intermediate statevectors.
///
struct quantum_circuit_cache
{
	struct statevector* psi_list;
	int nqubits;
	int ngates;
};

int allocate_quantum_circuit_cache(const int nqubits, const int ngates, struct quantum_circuit_cache* cache);

void free_quantum_circuit_cache(struct quantum_circuit_cache* cache);

int quantum_circuit_forward(const struct mat4x4 gates[], int ngates, const int wires[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out);

int quantum_circuit_backward(const struct mat4x4 gates[], int ngates, const int wires[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dgates[]);
