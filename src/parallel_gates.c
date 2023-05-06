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
		const int m = 1 << i;
		const int n = 1 << (psi->nqubits - 1 - j);
		for (int a = 0; a < m; a++)
		{
			for (int b = 0; b < n; b++)
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
		const int m = 1 << i;
		const int n = 1 << (j - i - 1);
		const int o = 1 << (psi->nqubits - 1 - j);
		for (int a = 0; a < m; a++)
		{
			for (int b = 0; b < n; b++)
			{
				for (int c = 0; c < o; c++)
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
