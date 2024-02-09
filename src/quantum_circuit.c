#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "quantum_circuit.h"
#include "gate.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply a quantum circuit consisting of two-qubit gates to state psi.
///
int apply_quantum_circuit(const struct mat4x4 gates[], const int ngates, const int wires[], const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(ngates >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(psi->nqubits, &tmp) < 0) {
		fprintf(stderr, "allocating temporary statevector with %i qubits failed\n", psi->nqubits);
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		const struct statevector* psi0 = (i == 0 ? psi : ((ngates - i) % 2 == 0 ? psi_out : &tmp));
		struct statevector* psi1 = ((ngates - i) % 2 == 0 ? &tmp : psi_out);
		apply_gate(&gates[i], wires[2*i], wires[2*i + 1], psi0, psi1);
	}

	free_statevector(&tmp);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate temporary cache required by backward pass of a brick wall quantum circuit.
///
int allocate_quantum_circuit_cache(const int nqubits, const int ngates, struct quantum_circuit_cache* cache)
{
	assert(ngates >= 1);

	cache->nqubits = nqubits;
	cache->ngates  = ngates;

	cache->psi_list = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct statevector));
	if (cache->psi_list == NULL) {
		fprintf(stderr, "allocating memory for statevector array failed\n");
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		int ret = allocate_statevector(nqubits, &cache->psi_list[i]);
		if (ret < 0) {
			return ret;
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Free temporary cache required by backward pass of a brick wall quantum circuit.
///
void free_quantum_circuit_cache(struct quantum_circuit_cache* cache)
{
	for (int i = 0; i < cache->ngates; i++)
	{
		free_statevector(&cache->psi_list[i]);
	}

	aligned_free(cache->psi_list);
	cache->psi_list = NULL;

	cache->nqubits = 0;
	cache->ngates  = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a quantum circuit consisting of two-qubit gates to state psi.
///
int quantum_circuit_forward(const struct mat4x4 gates[], int ngates, const int wires[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out)
{
	assert(ngates >= 1);
	assert(psi->nqubits == psi_out->nqubits);
	assert(cache->ngates == ngates);
	assert(cache->nqubits == psi->nqubits);

	// store initial statevector in cache as well
	memcpy(cache->psi_list[0].data, psi->data, ((size_t)1 << psi->nqubits) * sizeof(numeric));

	for (int i = 0; i < ngates; i++)
	{
		apply_gate(&gates[i], wires[2*i], wires[2*i + 1], &cache->psi_list[i],
			(i + 1 < ngates ? &cache->psi_list[i + 1] : psi_out));
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying a quantum circuit consisting of two-qubit gates to state psi.
///
int quantum_circuit_backward(const struct mat4x4 gates[], int ngates, const int wires[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dgates[])
{
	assert(ngates >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == ngates);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(dpsi->nqubits, &tmp) < 0) {
		return -1;
	}

	for (int i = ngates - 1; i >= 0; i--)
	{
		const struct statevector* dpsi1 = ((i == ngates - 1) ? dpsi_out : (i % 2 == 1 ? dpsi : &tmp));
		struct statevector* dpsi0 = (i % 2 == 1 ? &tmp : dpsi);

		apply_gate_backward(&gates[i], wires[2*i], wires[2*i + 1], &cache->psi_list[i], dpsi1, dpsi0, &dgates[i]);
	}

	free_statevector(&tmp);

	return 0;
}
