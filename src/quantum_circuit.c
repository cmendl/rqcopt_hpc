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
int quantum_circuit_forward(const struct mat4x4 gates[], const int ngates, const int wires[],
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
int quantum_circuit_backward(const struct mat4x4 gates[], const int ngates, const int wires[], const struct quantum_circuit_cache* cache,
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


//________________________________________________________________________________________________________________________
///
/// \brief Quantum circuit output state, gradient and Hessian-vector product with respect to gates in direction 'gatedirs'.
///
int quantum_circuit_gates_hessian_vector_product(const struct mat4x4 gates[], const struct mat4x4 gatedirs[], const int ngates, const int wires[],
	const struct statevector* restrict psi, const struct statevector* restrict phi,
	struct statevector* restrict psi_out, struct mat4x4 dgates[], struct mat4x4 hess_gatedirs[])
{
	assert(psi->nqubits == phi->nqubits);
	assert(psi->nqubits == psi_out->nqubits);

	struct quantum_circuit_cache cache;
	if (allocate_quantum_circuit_cache(psi->nqubits, ngates, &cache) < 0) {
		fprintf(stderr, "allocating quantum circuit cache failed\n");
		return -1;
	}

	int ret = quantum_circuit_forward(gates, ngates, wires, psi, &cache, psi_out);
	if (ret < 0) {
		return ret;
	}

	struct quantum_circuit_cache cache_rev;
	if (allocate_quantum_circuit_cache(phi->nqubits, ngates, &cache_rev) < 0) {
		fprintf(stderr, "allocating quantum circuit cache failed\n");
		return -1;
	}
	struct statevector phi_out;
	allocate_statevector(phi->nqubits, &phi_out);
	// apply quantum circuit in reverse order
	memcpy(cache_rev.psi_list[ngates - 1].data, phi->data, ((size_t)1 << phi->nqubits) * sizeof(numeric));
	for (int i = ngates - 1; i >= 0; i--)
	{
		struct statevector* dpsi0 = (i > 0 ? &cache_rev.psi_list[i - 1] : &phi_out);
		apply_gate_backward(&gates[i], wires[2*i], wires[2*i + 1], &cache.psi_list[i], &cache_rev.psi_list[i], dpsi0, &dgates[i]);
	}
	free_statevector(&phi_out);

	// perform a forward-backward pass, reversing the computational steps for gradient backpropagation

	struct statevector tmp1, tmp2;
	allocate_statevector(psi->nqubits, &tmp1);
	allocate_statevector(psi->nqubits, &tmp2);

	struct statevector dphi;
	allocate_statevector(phi->nqubits, &dphi);
	// first iteration i = 0
	{
		zero_matrix(&hess_gatedirs[0]);
		apply_gate(&gatedirs[0], wires[0], wires[1], &cache.psi_list[0], &dphi);
	}
	for (int i = 1; i < ngates; i++)  // continue with i = 1
	{
		struct mat4x4 gate_rev;
		transpose(&gates[i], &gate_rev);
		struct mat4x4 dgate;
		// skip for first iteration since logically, 'dphi' is initially the zero vector
		apply_gate_backward(&gate_rev, wires[2*i], wires[2*i + 1], &cache_rev.psi_list[i], &dphi, &tmp1, &dgate);
		transpose(&dgate, &hess_gatedirs[i]);
		apply_gate(&gatedirs[i], wires[2*i], wires[2*i + 1], &cache.psi_list[i], &tmp2);
		add_statevectors(&tmp1, &tmp2, &dphi);
	}
	free_statevector(&dphi);

	struct statevector dpsi;
	allocate_statevector(psi->nqubits, &dpsi);
	// first iteration i = ngates - 1
	{
		int i = ngates - 1;
		struct mat4x4 gatedir_t;
		transpose(&gatedirs[i], &gatedir_t);
		apply_gate(&gatedir_t, wires[2*i], wires[2*i + 1], &cache_rev.psi_list[i], &dpsi);
	}
	for (int i = ngates - 2; i >= 0; i--)  // continue with i = ngates - 2
	{
		struct mat4x4 dgate;
		// skip for first iteration since logically, 'dpsi' is initially the zero vector
		apply_gate_backward(&gates[i], wires[2*i], wires[2*i + 1], &cache.psi_list[i], &dpsi, &tmp1, &dgate);
		add_matrix(&hess_gatedirs[i], &dgate);
		struct mat4x4 gatedir_t;
		transpose(&gatedirs[i], &gatedir_t);
		apply_gate(&gatedir_t, wires[2*i], wires[2*i + 1], &cache_rev.psi_list[i], &tmp2);
		add_statevectors(&tmp1, &tmp2, &dpsi);
	}
	free_statevector(&dpsi);

	free_statevector(&tmp2);
	free_statevector(&tmp1);

	free_quantum_circuit_cache(&cache_rev);
	free_quantum_circuit_cache(&cache);

	return 0;
}
