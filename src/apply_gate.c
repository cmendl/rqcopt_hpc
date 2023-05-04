#include <assert.h>
#include "apply_gate.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply a single-qubit gate to qubit 'i' of a statevector.
///
void apply_single_qubit_gate(const struct single_qubit_gate* gate, int i, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	const int m = 1 << i;
	const int n = 1 << (psi->nqubits - 1 - i);
	for (int k = 0; k < m; k++)
	{
		for (int j = 0; j < n; j++)
		{
			numeric x = psi->data[k*(2*n) +      j ];
			numeric y = psi->data[k*(2*n) + (n + j)];
			psi_out->data[k*(2*n) +      j ] = gate->data[0] * x + gate->data[1] * y;
			psi_out->data[k*(2*n) + (n + j)] = gate->data[2] * x + gate->data[3] * y;
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_two_qubit_gate(const struct two_qubit_gate* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i < j);
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
