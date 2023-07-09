#include <memory.h>
#include <assert.h>
#include "gate.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_gate(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct mat4x4 gate_perm = { .data =
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
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
				psi_out->data[(a*4 + 2)*n + b] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
				psi_out->data[(a*4 + 3)*n + b] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
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
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector.
///
void apply_gate_backward(const struct mat4x4* gate, int i, int j, const struct statevector* restrict psi,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4* dgate)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(psi->nqubits == dpsi->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct mat4x4 gate_perm = { .data =
			{
				gate->data[ 0], gate->data[ 2], gate->data[ 1], gate->data[ 3],
				gate->data[ 8], gate->data[10], gate->data[ 9], gate->data[11],
				gate->data[ 4], gate->data[ 6], gate->data[ 5], gate->data[ 7],
				gate->data[12], gate->data[14], gate->data[13], gate->data[15],
			}
		};

		struct mat4x4 dgate_perm;

		// flip i <-> j
		apply_gate_backward(&gate_perm, j, i, psi, dpsi_out, dpsi, &dgate_perm);

		// undo transposition of first and second qubit wire for gate gradient
		dgate->data[ 0] = dgate_perm.data[ 0];  dgate->data[ 1] = dgate_perm.data[ 2];  dgate->data[ 2] = dgate_perm.data[ 1];  dgate->data[ 3] = dgate_perm.data[ 3];
		dgate->data[ 4] = dgate_perm.data[ 8];  dgate->data[ 5] = dgate_perm.data[10];  dgate->data[ 6] = dgate_perm.data[ 9];  dgate->data[ 7] = dgate_perm.data[11];
		dgate->data[ 8] = dgate_perm.data[ 4];  dgate->data[ 9] = dgate_perm.data[ 6];  dgate->data[10] = dgate_perm.data[ 5];  dgate->data[11] = dgate_perm.data[ 7];
		dgate->data[12] = dgate_perm.data[12];  dgate->data[13] = dgate_perm.data[14];  dgate->data[14] = dgate_perm.data[13];  dgate->data[15] = dgate_perm.data[15];
	}
	else if (j == i + 1)
	{
		memset(dgate->data, 0, sizeof(dgate->data));

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
				dgate->data[ 0] += dx * x;  dgate->data[ 1] += dx * y;  dgate->data[ 2] += dx * z;  dgate->data[ 3] += dx * w;
				dgate->data[ 4] += dy * x;  dgate->data[ 5] += dy * y;  dgate->data[ 6] += dy * z;  dgate->data[ 7] += dy * w;
				dgate->data[ 8] += dz * x;  dgate->data[ 9] += dz * y;  dgate->data[10] += dz * z;  dgate->data[11] += dz * w;
				dgate->data[12] += dw * x;  dgate->data[13] += dw * y;  dgate->data[14] += dw * z;  dgate->data[15] += dw * w;

				// gradient with respect to input vector 'psi'
				dpsi->data[(a*4    )*n + b] = gate->data[ 0] * dx + gate->data[ 4] * dy + gate->data[ 8] * dz + gate->data[12] * dw;
				dpsi->data[(a*4 + 1)*n + b] = gate->data[ 1] * dx + gate->data[ 5] * dy + gate->data[ 9] * dz + gate->data[13] * dw;
				dpsi->data[(a*4 + 2)*n + b] = gate->data[ 2] * dx + gate->data[ 6] * dy + gate->data[10] * dz + gate->data[14] * dw;
				dpsi->data[(a*4 + 3)*n + b] = gate->data[ 3] * dx + gate->data[ 7] * dy + gate->data[11] * dz + gate->data[15] * dw;
			}
		}
	}
	else
	{
		memset(dgate->data, 0, sizeof(dgate->data));

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgate->data[ 0] += dx * x;  dgate->data[ 1] += dx * y;  dgate->data[ 2] += dx * z;  dgate->data[ 3] += dx * w;
					dgate->data[ 4] += dy * x;  dgate->data[ 5] += dy * y;  dgate->data[ 6] += dy * z;  dgate->data[ 7] += dy * w;
					dgate->data[ 8] += dz * x;  dgate->data[ 9] += dz * y;  dgate->data[10] += dz * z;  dgate->data[11] += dz * w;
					dgate->data[12] += dw * x;  dgate->data[13] += dw * y;  dgate->data[14] += dw * z;  dgate->data[15] += dw * w;

					// gradient with respect to input vector 'psi'
					dpsi->data[(((a*2    )*n + b)*2    )*o + c] = gate->data[ 0] * dx + gate->data[ 4] * dy + gate->data[ 8] * dz + gate->data[12] * dw;
					dpsi->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->data[ 1] * dx + gate->data[ 5] * dy + gate->data[ 9] * dz + gate->data[13] * dw;
					dpsi->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->data[ 2] * dx + gate->data[ 6] * dy + gate->data[10] * dz + gate->data[14] * dw;
					dpsi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->data[ 3] * dx + gate->data[ 7] * dy + gate->data[11] * dz + gate->data[15] * dw;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_gate_to_array(const struct mat4x4* gate, int i, int j, const struct statevector_array* restrict psi, struct statevector_array* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(psi->nstates == psi_out->nstates);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct mat4x4 gate_perm = { .data =
			{
				gate->data[ 0], gate->data[ 2], gate->data[ 1], gate->data[ 3],
				gate->data[ 8], gate->data[10], gate->data[ 9], gate->data[11],
				gate->data[ 4], gate->data[ 6], gate->data[ 5], gate->data[ 7],
				gate->data[12], gate->data[14], gate->data[13], gate->data[15],
			}
		};
		// flip i <-> j
		apply_gate_to_array(&gate_perm, j, i, psi, psi_out);
	}
	else if (j == i + 1)
	{
		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric x = psi->data[(a*4    )*n + b];
				numeric y = psi->data[(a*4 + 1)*n + b];
				numeric z = psi->data[(a*4 + 2)*n + b];
				numeric w = psi->data[(a*4 + 3)*n + b];

				psi_out->data[(a*4    )*n + b] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
				psi_out->data[(a*4 + 1)*n + b] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
				psi_out->data[(a*4 + 2)*n + b] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
				psi_out->data[(a*4 + 3)*n + b] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
			}
		}
	}
	else
	{
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = ((intqs)1 << (psi->nqubits - 1 - j)) * psi->nstates;
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					psi_out->data[(((a*2    )*n + b)*2    )*o + c] = gate->data[ 0] * x + gate->data[ 1] * y + gate->data[ 2] * z + gate->data[ 3] * w;
					psi_out->data[(((a*2    )*n + b)*2 + 1)*o + c] = gate->data[ 4] * x + gate->data[ 5] * y + gate->data[ 6] * z + gate->data[ 7] * w;
					psi_out->data[(((a*2 + 1)*n + b)*2    )*o + c] = gate->data[ 8] * x + gate->data[ 9] * y + gate->data[10] * z + gate->data[11] * w;
					psi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c] = gate->data[12] * x + gate->data[13] * y + gate->data[14] * z + gate->data[15] * w;
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Gate gradients corresponding to backward pass for applying a two-qubit gate to qubits 'i' and 'j' of a statevector array.
///
void apply_gate_backward_array(const struct mat4x4* gate, int i, int j, const struct statevector_array* restrict psi,
	const struct statevector* restrict dpsi_out, struct mat4x4* dgates)
{
	assert(psi->nqubits == dpsi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);

	if (j < i)
	{
		// transpose first and second qubit wire of gate
		const struct mat4x4 gate_perm = { .data =
			{
				gate->data[ 0], gate->data[ 2], gate->data[ 1], gate->data[ 3],
				gate->data[ 8], gate->data[10], gate->data[ 9], gate->data[11],
				gate->data[ 4], gate->data[ 6], gate->data[ 5], gate->data[ 7],
				gate->data[12], gate->data[14], gate->data[13], gate->data[15],
			}
		};

		struct mat4x4* dgates_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nstates * sizeof(struct mat4x4));

		// flip i <-> j
		apply_gate_backward_array(&gate_perm, j, i, psi, dpsi_out, dgates_perm);

		// undo transposition of first and second qubit wire for gate gradient
		for (int k = 0; k < psi->nstates; k++)
		{
			// undo transposition of first and second qubit wire for gate gradient
			dgates[k].data[ 0] = dgates_perm[k].data[ 0];  dgates[k].data[ 1] = dgates_perm[k].data[ 2];  dgates[k].data[ 2] = dgates_perm[k].data[ 1];  dgates[k].data[ 3] = dgates_perm[k].data[ 3];
			dgates[k].data[ 4] = dgates_perm[k].data[ 8];  dgates[k].data[ 5] = dgates_perm[k].data[10];  dgates[k].data[ 6] = dgates_perm[k].data[ 9];  dgates[k].data[ 7] = dgates_perm[k].data[11];
			dgates[k].data[ 8] = dgates_perm[k].data[ 4];  dgates[k].data[ 9] = dgates_perm[k].data[ 6];  dgates[k].data[10] = dgates_perm[k].data[ 5];  dgates[k].data[11] = dgates_perm[k].data[ 7];
			dgates[k].data[12] = dgates_perm[k].data[12];  dgates[k].data[13] = dgates_perm[k].data[14];  dgates[k].data[14] = dgates_perm[k].data[13];  dgates[k].data[15] = dgates_perm[k].data[15];
		}

		aligned_free(dgates_perm);
	}
	else if (j == i + 1)
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].data, 0, sizeof(dgates[k].data));
		}

		// special case: neighboring wires
		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				numeric dx = dpsi_out->data[(a*4    )*n + b];
				numeric dy = dpsi_out->data[(a*4 + 1)*n + b];
				numeric dz = dpsi_out->data[(a*4 + 2)*n + b];
				numeric dw = dpsi_out->data[(a*4 + 3)*n + b];

				for (int k = 0; k < psi->nstates; k++)
				{
					numeric x = psi->data[((a*4    )*n + b)*psi->nstates + k];
					numeric y = psi->data[((a*4 + 1)*n + b)*psi->nstates + k];
					numeric z = psi->data[((a*4 + 2)*n + b)*psi->nstates + k];
					numeric w = psi->data[((a*4 + 3)*n + b)*psi->nstates + k];

					// gradient with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
					dgates[k].data[ 0] += dx * x;  dgates[k].data[ 1] += dx * y;  dgates[k].data[ 2] += dx * z;  dgates[k].data[ 3] += dx * w;
					dgates[k].data[ 4] += dy * x;  dgates[k].data[ 5] += dy * y;  dgates[k].data[ 6] += dy * z;  dgates[k].data[ 7] += dy * w;
					dgates[k].data[ 8] += dz * x;  dgates[k].data[ 9] += dz * y;  dgates[k].data[10] += dz * z;  dgates[k].data[11] += dz * w;
					dgates[k].data[12] += dw * x;  dgates[k].data[13] += dw * y;  dgates[k].data[14] += dw * z;  dgates[k].data[15] += dw * w;
				}
			}
		}
	}
	else
	{
		for (int k = 0; k < psi->nstates; k++) {
			memset(dgates[k].data, 0, sizeof(dgates[k].data));
		}

		const intqs m = (intqs)1 << i;
		const intqs n = (intqs)1 << (j - i - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - j);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric dx = dpsi_out->data[(((a*2    )*n + b)*2    )*o + c];
					numeric dy = dpsi_out->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric dz = dpsi_out->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric dw = dpsi_out->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					for (int k = 0; k < psi->nstates; k++)
					{
						numeric x = psi->data[((((a*2    )*n + b)*2    )*o + c)*psi->nstates + k];
						numeric y = psi->data[((((a*2    )*n + b)*2 + 1)*o + c)*psi->nstates + k];
						numeric z = psi->data[((((a*2 + 1)*n + b)*2    )*o + c)*psi->nstates + k];
						numeric w = psi->data[((((a*2 + 1)*n + b)*2 + 1)*o + c)*psi->nstates + k];

						// gradient of target function with respect to gate entries (outer product between 'dpsi_out' and 'psi' entries)
						dgates[k].data[ 0] += dx * x;  dgates[k].data[ 1] += dx * y;  dgates[k].data[ 2] += dx * z;  dgates[k].data[ 3] += dx * w;
						dgates[k].data[ 4] += dy * x;  dgates[k].data[ 5] += dy * y;  dgates[k].data[ 6] += dy * z;  dgates[k].data[ 7] += dy * w;
						dgates[k].data[ 8] += dz * x;  dgates[k].data[ 9] += dz * y;  dgates[k].data[10] += dz * z;  dgates[k].data[11] += dz * w;
						dgates[k].data[12] += dw * x;  dgates[k].data[13] += dw * y;  dgates[k].data[14] += dw * z;  dgates[k].data[15] += dw * w;
					}
				}
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply a two-qubit gate "placeholder" acting on qubits 'i' and 'j' of a statevector.
/// 
/// Outputs a statevector array containing 16 vectors, corresponding to the placeholder gate entries.
///
void apply_gate_placeholder(const int i, const int j, const struct statevector* psi, struct statevector_array* psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	assert(0 <= i && i < psi->nqubits);
	assert(0 <= j && j < psi->nqubits);
	assert(i != j);
	assert(psi_out->nstates == 16);

	memset(psi_out->data, 0, ((size_t)1 << psi_out->nqubits) * psi_out->nstates * sizeof(numeric));

	if (i < j)
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
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 1] = y;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 2] = z;
							psi_out->data[((((((a*2 + k)*n + b)*2 + l)*o + c)*2 + k)*2 + l)*4 + 3] = w;
						}
					}
				}
			}
		}
	}
	else // i > j
	{
		const intqs m = (intqs)1 << j;
		const intqs n = (intqs)1 << (i - j - 1);
		const intqs o = (intqs)1 << (psi->nqubits - 1 - i);
		for (intqs a = 0; a < m; a++)
		{
			for (intqs b = 0; b < n; b++)
			{
				for (intqs c = 0; c < o; c++)
				{
					numeric x = psi->data[(((a*2    )*n + b)*2    )*o + c];
					numeric y = psi->data[(((a*2    )*n + b)*2 + 1)*o + c];
					numeric z = psi->data[(((a*2 + 1)*n + b)*2    )*o + c];
					numeric w = psi->data[(((a*2 + 1)*n + b)*2 + 1)*o + c];

					// equivalent to outer product with 4x4 identity matrix and transpositions
					for (int k = 0; k < 2; k++)
					{
						for (int l = 0; l < 2; l++)
						{
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 0] = x;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 1] = z;
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 2] = y;  // note: flipping y <-> z
							psi_out->data[((((((a*2 + l)*n + b)*2 + k)*o + c)*2 + k)*2 + l)*4 + 3] = w;
						}
					}
				}
			}
		}
	}
}
