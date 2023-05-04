#include <stdio.h>
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
		fprintf(stderr, "allocating statevector memory (%zu bytes) failed", size);
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
