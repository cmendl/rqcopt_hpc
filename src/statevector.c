#include <stdio.h>
#include <assert.h>
#include "statevector.h"
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a statevector.
///
int allocate_statevector(int nqubits, struct statevector* psi)
{
	size_t size = ((size_t)1 << nqubits) * sizeof(numeric);
	if (size < MEM_DATA_ALIGN) {
		size = MEM_DATA_ALIGN;
	}

	psi->nqubits = nqubits;
	psi->data = aligned_alloc(MEM_DATA_ALIGN, size);
	if (psi->data == NULL)
	{
		fprintf(stderr, "allocating statevector memory (%zu bytes) failed\n", size);
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Free (release memory of) a statevector.
///
void free_statevector(struct statevector* psi)
{
	if (psi->data != NULL)
	{
		aligned_free(psi->data);
		psi->data = NULL;
	}
	psi->nqubits = 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Transpose the entries of a statevector (permute qubit wires).
///
void transpose_statevector(const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_trans)
{
	assert(psi->nqubits == psi_trans->nqubits);

	intqs* strides = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(intqs));
	for (int i = 0; i < psi->nqubits; i++)
	{
		assert(0 <= perm[i] && perm[i] < psi->nqubits);
		strides[psi->nqubits - 1 - i] = (intqs)1 << (psi->nqubits - 1 - perm[i]);
	}

	const int n = (intqs)1 << psi->nqubits;
	for (intqs j = 0; j < n; j++)
	{
		intqs k = 0;
		for (int i = 0; i < psi->nqubits; i++) {
			k += ((j >> i) & 1) * strides[i];
		}
		psi_trans->data[j] = psi->data[k];
	}

	aligned_free(strides);
}
