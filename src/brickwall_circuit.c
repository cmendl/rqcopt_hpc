#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "brickwall_circuit.h"
#include "gate.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_brickwall_unitary(const struct mat4x4 Vlist[], const int nlayers, const int* perms[], const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	const int nstates = nlayers * (psi->nqubits / 2);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(psi->nqubits, &tmp) < 0) {
		fprintf(stderr, "allocating temporary statevector with %i qubits failed\n", psi->nqubits);
		return -1;
	}

	int k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < psi->nqubits; j += 2)
		{
			const struct statevector* psi0 = (k == 0 ? psi : ((nstates - k) % 2 == 0 ? psi_out : &tmp));
			struct statevector* psi1 = ((nstates - k) % 2 == 0 ? &tmp : psi_out);
			apply_gate(&Vlist[i], perms[i][j], perms[i][j + 1], psi0, psi1);
			k++;
		}
	}

	free_statevector(&tmp);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int brickwall_unitary_forward(const struct mat4x4 Vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct quantum_circuit_cache* cache, struct statevector* restrict psi_out)
{
	const int nstates = nlayers * (psi->nqubits / 2);

	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);
	assert(cache->ngates == nstates);
	assert(cache->nqubits == psi->nqubits);
	assert(psi->nqubits % 2 == 0);

	// store initial statevector in cache as well
	memcpy(cache->psi_list[0].data, psi->data, ((size_t)1 << psi->nqubits) * sizeof(numeric));

	int k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < psi->nqubits; j += 2)
		{
			apply_gate(&Vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k],
				(k + 1 < nstates ? &cache->psi_list[k + 1] : psi_out));
			k++;
		}
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
int brickwall_unitary_backward(const struct mat4x4 Vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dVlist[])
{
	const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(dpsi->nqubits, &tmp) < 0) {
		return -1;
	}

	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			const struct statevector* dpsi1 = ((k == nstates - 1) ? dpsi_out : (k % 2 == 1 ? dpsi : &tmp));
			struct statevector* dpsi0 = (k % 2 == 1 ? &tmp : dpsi);

			struct mat4x4 dV;
			apply_gate_backward(&Vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], dpsi1, dpsi0, &dV);
			// accumulate gradient
			add_matrix(&dVlist[i], &dV);

			k--;
		}
	}

	free_statevector(&tmp);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass and Hessian computation for applying the unitary matrix representation of a brick wall
/// quantum circuit with periodic boundary conditions to state psi.
///
/// On input, 'hess' must point to a memory block of size (nlayers * 16)^2.
///
int brickwall_unitary_backward_hessian(const struct mat4x4 Vlist[], int nlayers, const int* perms[], const struct quantum_circuit_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dVlist[], numeric* hess)
{
	const int nstates = nlayers * (dpsi->nqubits / 2);

	assert(nlayers >= 1);
	assert(dpsi_out->nqubits == dpsi->nqubits);
	assert(dpsi_out->nqubits % 2 == 0);
	assert(cache->nqubits == dpsi->nqubits);
	assert(cache->ngates == nstates);

	// store gradient vectors in another cache
	struct quantum_circuit_cache grad_cache;
	if (allocate_quantum_circuit_cache(dpsi->nqubits, nstates, &grad_cache) < 0) {
		fprintf(stderr, "allocating a brick wall unitary cache failed\n");
		return -1;
	}

	// store initial gradient statevector in cache
	memcpy(grad_cache.psi_list[nstates - 1].data, dpsi_out->data, ((size_t)1 << dpsi_out->nqubits) * sizeof(numeric));

	// gradient
	int k = nstates - 1;
	for (int i = nlayers - 1; i >= 0; i--)
	{
		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));

		for (int j = dpsi->nqubits - 2; j >= 0; j -= 2)
		{
			struct statevector* dpsi0 = (k > 0 ? &grad_cache.psi_list[k - 1] : dpsi);

			struct mat4x4 dV;
			apply_gate_backward(&Vlist[i], perms[i][j], perms[i][j + 1], &cache->psi_list[k], &grad_cache.psi_list[k], dpsi0, &dV);
			// accumulate gradient
			add_matrix(&dVlist[i], &dV);

			k--;
		}
	}

	// Hessian
	memset(hess, 0, nlayers * 16 * nlayers * 16 * sizeof(numeric));
	// temporary statevector array
	struct statevector_array tmp[2];
	for (int i = 0; i < 2; i++) {
		if (allocate_statevector_array(dpsi->nqubits, 16, &tmp[i]) < 0) {
			fprintf(stderr, "memory allocation of a statevector array for %i qubits and 16 states failed", dpsi->nqubits);
			return -1;
		}
	}
	// temporary derivative with respect to two gates
	struct mat4x4* h = aligned_alloc(MEM_DATA_ALIGN, 16 * sizeof(struct mat4x4));
	if (h == NULL) {
		fprintf(stderr, "allocating temporary gates failed\n");
		return -1;
	}
	k = 0;
	for (int i = 0; i < nlayers; i++)
	{
		for (int r = 0; r < dpsi->nqubits; r += 2)
		{
			apply_gate_placeholder(perms[i][r], perms[i][r + 1], &cache->psi_list[k], &tmp[0]);
			k++;

			// proceed through the circuit with gate placeholder at layer i and qubit pair indexed by r
			int p = 0;
			int l = k;
			for (int j = i; j < nlayers; j++)
			{
				for (int s = 0; s < dpsi->nqubits; s += 2)
				{
					if (i == j && s <= r) {
						continue;
					}

					apply_gate_backward_array(&Vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &grad_cache.psi_list[l], h);
					// accumulate Hessian entries
					for (int x = 0; x < 16; x++) {
						for (int y = 0; y < 16; y++) {
							hess[((i*16 + x)*nlayers + j)*16 + y] += h[x].data[y];
						}
					}

					if (j < nlayers - 1 || s < dpsi->nqubits - 2) {  // skip (expensive) gate application at final iteration
						apply_gate_to_array(&Vlist[j], perms[j][s], perms[j][s + 1], &tmp[p], &tmp[1 - p]);
						p = 1 - p;
					}

					l++;
				}
			}
		}
	}

	// symmetrize diagonal blocks
	for (int i = 0; i < nlayers; i++) {
		for (int x = 0; x < 16; x++) {
			for (int y = 0; y <= x; y++) {
				numeric s = hess[((i*16 + x)*nlayers + i)*16 + y] + hess[((i*16 + y)*nlayers + i)*16 + x];
				hess[((i*16 + x)*nlayers + i)*16 + y] = s;
				hess[((i*16 + y)*nlayers + i)*16 + x] = s;
			}
		}
	}
	// copy off-diagonal blocks according to symmetry
	for (int i = 0; i < nlayers; i++) {
		for (int j = 0; j < i; j++) {
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					hess[((i*16 + x)*nlayers + j)*16 + y] = hess[((j*16 + y)*nlayers + i)*16 + x];
				}
			}
		}
	}
	
	aligned_free(h);
	free_statevector_array(&tmp[1]);
	free_statevector_array(&tmp[0]);
	free_quantum_circuit_cache(&grad_cache);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi,
/// with a gate hole in layer 'l'.
///
int apply_brickwall_unitary_gate_placeholder(const struct mat4x4 Vlist[], int nlayers, const int* perms[], int l, const struct quantum_circuit_cache* cache, struct statevector_array* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(cache->nqubits == psi_out->nqubits);
	assert(cache->ngates == nlayers * (cache->nqubits / 2));
	assert(psi_out->nstates == 16);
	assert(0 <= l && l < nlayers);

	// temporary statevector arrays
	struct statevector_array tmp[2] = { 0 };
	for (int i = 0; i < 2; i++) {
		if (allocate_statevector_array(cache->nqubits, 16, &tmp[i]) < 0) {
			fprintf(stderr, "memory allocation of a statevector array for %i qubits and 16 states failed\n", cache->nqubits);
			return -1;
		}
	}

	const intqs nentries = ((intqs)1 << psi_out->nqubits) * psi_out->nstates;
	memset(psi_out->data, 0, nentries * sizeof(numeric));

	// iterate over hole locations in layer 'l'
	for (int s = 0; s < cache->nqubits; s += 2)
	{
		int k = l * (cache->nqubits / 2) + s / 2;
		apply_gate_placeholder(perms[l][s], perms[l][s + 1], &cache->psi_list[k], &tmp[0]);

		k = 0;
		for (int i = l; i < nlayers; i++)
		{
			for (int j = 0; j < cache->nqubits; j += 2)
			{
				if (i == l && j <= s) {
					continue;
				}
				apply_gate_to_array(&Vlist[i], perms[i][j], perms[i][j + 1], &tmp[k], &tmp[1 - k]);
				k = 1 - k;
			}
		}

		// accumulate statevectors
		for (int i = 0; i < nentries; i++)
		{
			psi_out->data[i] += tmp[k].data[i];
		}
	}

	for (int i = 0; i < 2; i++) {
		free_statevector_array(&tmp[i]);
	}

	return 0;
}
