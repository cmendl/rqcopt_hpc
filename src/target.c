#include <memory.h>
#include <stdio.h>
#include "target.h"
#include "brickwall_circuit.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function Re tr[U^{\dagger} W] and its gate gradients,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target_and_gradient(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval, struct mat4x4 dVlist[])
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(L, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(L, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(L, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}

	struct brickwall_unitary_cache cache = { 0 };
	if (allocate_brickwall_unitary_cache(L, nlayers * (L / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_brickwall_unitary_cache' failed");
		return -1;
	}

	struct mat4x4* dVlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (dVlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));
	}

	double f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'Ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = conj(Upsi.data[a]);
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(Vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// f -= Re <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f -= creal(Upsi.data[a] * Wpsi.data[a]);
		}

		// brickwall unitary backward pass
		// note: overwriting 'psi' with gradient
		if (brickwall_unitary_backward(Vlist, nlayers, perms, &cache, &Upsi, &psi, dVlist_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward' failed internally");
			return -4;
		}
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dVlist[i], &dVlist_unit[i]);
		}
	}

	// complex-conjugate gradient matrix entries
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			dVlist[i].data[j] = conj(dVlist[i].data[j]);
		}
	}

	aligned_free(dVlist_unit);
	free_brickwall_unitary_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}
