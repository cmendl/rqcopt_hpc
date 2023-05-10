#include <memory.h>
#include <stdbool.h>
#include <assert.h>
#include "parallel_gates.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct two_qubit_gate gate_perm = { .data =
			{
				gate->data[ 0], gate->data[ 2], gate->data[ 1], gate->data[ 3],
				gate->data[ 8], gate->data[10], gate->data[ 9], gate->data[11],
				gate->data[ 4], gate->data[ 6], gate->data[ 5], gate->data[ 7],
				gate->data[12], gate->data[14], gate->data[13], gate->data[15],
			}
		};
		// flip i <-> j
		apply_gate(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[a*(4*n) +        b ];
				numeric y = psi->data[a*(4*n) + (  n + b)];
				numeric z = psi->data[a*(4*n) + (2*n + b)];
				numeric w = psi->data[a*(4*n) + (3*n + b)];
				psi_out->data[a*(4*n) +        b ] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
				psi_out->data[a*(4*n) + (  n + b)] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
				psi_out->data[a*(4*n) + (2*n + b)] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
				psi_out->data[a*(4*n) + (3*n + b)] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[a*(4*n*o) +      b *(2*o) +      c ];
					numeric y = psi->data[a*(4*n*o) +      b *(2*o) + (o + c)];
					numeric z = psi->data[a*(4*n*o) + (n + b)*(2*o) +      c ];
					numeric w = psi->data[a*(4*n*o) + (n + b)*(2*o) + (o + c)];
					psi_out->data[a*(4*n*o) +      b *(2*o) +      c ] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
					psi_out->data[a*(4*n*o) +      b *(2*o) + (o + c)] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
					psi_out->data[a*(4*n*o) + (n + b)*(2*o) +      c ] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
					psi_out->data[a*(4*n*o) + (n + b)*(2*o) + (o + c)] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a parallel sequence of two-qubit gates V to state psi,
/// optionally using a permutation of quantum wires.
///
int apply_parallel_gates(const struct two_qubit_gate* V, const struct statevector* restrict psi, const int* perm,
	struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	const int L = psi->nqubits;
	assert(L % 2 == 0);

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(L, &tmp) < 0) {
		return -1;
	}

	for (int i = 0; i < L; i += 2)
	{
		int p = ((L - i) / 2) % 2;
		const struct statevector* psi0 = (i == 0 ? psi : (p == 0 ? psi_out : &tmp));
		struct statevector* psi1 = (p == 0 ? &tmp : psi_out);
		apply_gate(V, inv_perm[i], inv_perm[i + 1], psi0, psi1);
	}

	free_statevector(&tmp);
	aligned_free(inv_perm);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the gradient of V x ... x V in direction Z to state psi.
///
int apply_parallel_gates_directed_grad(const struct two_qubit_gate* V, const struct two_qubit_gate* Z,
	const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	const int L = psi->nqubits;
	assert(L % 2 == 0);

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	// temporary statevectors
	struct statevector chi = { 0 };
	if (allocate_statevector(L, &chi) < 0) {
		return -1;
	}
	struct statevector tmp = { 0 };
	if (allocate_statevector(L, &tmp) < 0) {
		return -1;
	}

	for (int i = 0; i < L; i += 2)
	{
		struct statevector* phi = (i == 0 ? psi_out : &chi);

		for (int j = 0; j < L; j += 2)
		{
			int p = ((L - j) / 2) % 2;
			const struct statevector* psi0 = (j == 0 ? psi : (p == 0 ? phi : &tmp));
			struct statevector* psi1 = (p == 0 ? &tmp : phi);
			apply_gate(i == j ? Z : V, inv_perm[j], inv_perm[j + 1], psi0, psi1);
		}

		if (i > 0)
		{
			// accumulate statevectors
			const intqs n = (intqs)1 << L;
			for (intqs j = 0; j < n; j++)
			{
				psi_out->data[j] += chi.data[j];
			}
		}
	}

	free_statevector(&tmp);
	free_statevector(&chi);
	aligned_free(inv_perm);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the gradient of Re tr[U^{\dagger} (V x ... x V)] with respect to V,
/// using the provided matrix-free application of U to a state.
///
int parallel_gates_grad_matfree(const struct two_qubit_gate* V, int L, unitary_func Ufunc, void* fdata, const int* perm, struct two_qubit_gate* G)
{
	assert(L >= 2);
	assert(L % 2 == 0);

	bool is_identity_perm = true;
	for (int i = 0; i < L; i++) {
		if (perm[i] != i) {
			is_identity_perm = false;
			break;
		}
	}
	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	struct statevector psi0 = { 0 };
	struct statevector psi1 = { 0 };
	if (allocate_statevector(L, &psi0) < 0) { return -1; }
	if (allocate_statevector(L, &psi1) < 0) { return -1; }

	memset(G->data, 0, sizeof(G->data));

	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi0.data, 0, n * sizeof(numeric));
		if (is_identity_perm)
		{
			psi0.data[b] = 1;
		}
		else
		{		
			// transpose unit vector with entry 1 at b
			intqs k = 0;
			for (int i = 0; i < L; i++) {
				k += ((b >> i) & 1) * ((intqs)1 << (L - 1 - inv_perm[L - 1 - i]));
			}
			psi0.data[k] = 1;
		}
		// apply U
		int ret = Ufunc(&psi0, fdata, &psi1);
		if (ret < 0) {
			return ret;
		}
		struct statevector* psi;
		struct statevector* chi;
		if (is_identity_perm) {
			psi = &psi1;
			chi = &psi0;
		}
		else {
			transpose_statevector(&psi1, inv_perm, &psi0);
			psi = &psi0;
			chi = &psi1;
		}
		for (int i = 0; i < L; i += 2)
		{
			// use chi->data as temporary memory
			numeric* r = chi->data;
			numeric* s = psi->data;

			// inner product with (V x ... x V) |b>, where the i-th V is omitted
			for (int j = 0; j < i; j += 2)
			{
				// TODO: complex conjugation
				numeric x = V->data[     ((b >> j) & 3)];
				numeric y = V->data[ 4 + ((b >> j) & 3)];
				numeric z = V->data[ 8 + ((b >> j) & 3)];
				numeric w = V->data[12 + ((b >> j) & 3)];

				const intqs m = (intqs)1 << (L - j - 2);
				for (intqs k = 0; k < m; k++)
				{
					r[k] = (
						s[4*k    ] * x +
						s[4*k + 1] * y +
						s[4*k + 2] * z +
						s[4*k + 3] * w);
				}
				// avoid overwriting psi->data
				if (s == psi->data) {
					s = chi->data + (n >> 2);
				}
				// swap pointers r <-> s
				numeric* t = r;
				r = s;
				s = t;
			}
			for (int j = i + 2; j < L; j += 2)
			{
				// TODO: complex conjugation
				numeric x = V->data[     ((b >> j) & 3)];
				numeric y = V->data[ 4 + ((b >> j) & 3)];
				numeric z = V->data[ 8 + ((b >> j) & 3)];
				numeric w = V->data[12 + ((b >> j) & 3)];

				const intqs m = (intqs)1 << (L - j - 2);
				for (intqs k = 0; k < m; k++)
				{
					for (intqs l = 0; l < 4; l++)
					{
						r[4*k + l] = (
							s[4*(4*k    ) + l] * x +
							s[4*(4*k + 1) + l] * y +
							s[4*(4*k + 2) + l] * z +
							s[4*(4*k + 3) + l] * w);
					}
				}
				// avoid overwriting psi->data
				if (s == psi->data) {
					s = chi->data + (n >> 2);
				}
				// swap pointers r <-> s
				numeric* t = r;
				r = s;
				s = t;
			}
			G->data[     ((b >> i) & 3)] += s[0];
			G->data[ 4 + ((b >> i) & 3)] += s[1];
			G->data[ 8 + ((b >> i) & 3)] += s[2];
			G->data[12 + ((b >> i) & 3)] += s[3];
		}
	}

	free_statevector(&psi1);
	free_statevector(&psi0);
	aligned_free(inv_perm);

	return 0;
}
