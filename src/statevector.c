#include <stdio.h>
#include <cblas.h>
#include <assert.h>
#include "statevector.h"
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a statevector.
///
int allocate_statevector(int nqubits, struct statevector* psi)
{
	psi->nqubits = nqubits;

	const size_t size = ((size_t)1 << nqubits) * sizeof(numeric);
	psi->data = aligned_malloc(size);
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
/// \brief Normalize a statevector. If the vector is zero, it remains unchanged.
///
void normalize_statevector(struct statevector* restrict psi)
{
	const intqs n = (intqs)1 << psi->nqubits;

	#ifdef COMPLEX_CIRCUIT

	double nrm = cblas_dznrm2(n, psi->data, 1);
	if (nrm > 0) {
		cblas_zdscal(n, 1. / nrm, psi->data, 1);
	}

	#else

	double nrm = cblas_dnrm2(n, psi->data, 1);
	if (nrm > 0) {
		cblas_dscal(n, 1. / nrm, psi->data, 1);
	}

	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Transpose the entries of a statevector (permute qubit wires).
///
void transpose_statevector(const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_trans)
{
	assert(psi->nqubits == psi_trans->nqubits);

	intqs* strides = aligned_malloc(psi->nqubits * sizeof(intqs));
	for (int i = 0; i < psi->nqubits; i++)
	{
		assert(0 <= perm[i] && perm[i] < psi->nqubits);
		strides[psi->nqubits - 1 - i] = (intqs)1 << (psi->nqubits - 1 - perm[i]);
	}

	const intqs n = (intqs)1 << psi->nqubits;
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


//________________________________________________________________________________________________________________________
///
/// \brief Fill the statevector entries according to the Haar random (uniform) distribution.
///
void haar_random_statevector(struct statevector* psi, struct rng_state* rng)
{
	// initialize 'psi' with (independent and identically distributed) Gaussian entries, then normalize

	const intqs n = (intqs)1 << psi->nqubits;

	#ifdef COMPLEX_CIRCUIT

	for (intqs j = 0; j < n; j++)
	{
		psi->data[j] = crandn(rng);
	}

	#else

	for (intqs j = 0; j < n; j++)
	{
		psi->data[j] = randn(rng);
	}

	#endif

	normalize_statevector(psi);
}


//________________________________________________________________________________________________________________________
///
/// \brief Add two statevectors.
///
void add_statevectors(const struct statevector* restrict psi1, const struct statevector* restrict psi2, struct statevector* restrict psi_sum)
{
	assert(psi1->nqubits == psi2->nqubits);
	assert(psi1->nqubits == psi_sum->nqubits);

	const intqs n = (intqs)1 << psi1->nqubits;
	for (intqs j = 0; j < n; j++)
	{
		psi_sum->data[j] = psi1->data[j] + psi2->data[j];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Allocate a statevector array.
///
int allocate_statevector_array(int nqubits, int nstates, struct statevector_array* psi_array)
{
	psi_array->nqubits = nqubits;
	psi_array->nstates = nstates;

	const size_t size = ((size_t)1 << nqubits) * nstates * sizeof(numeric);
	psi_array->data = aligned_malloc(size);
	if (psi_array->data == NULL)
	{
		fprintf(stderr, "allocating statevector array memory (%zu bytes) failed\n", size);
		return -1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Free (release memory of) a statevector array.
///
void free_statevector_array(struct statevector_array* psi_array)
{
	if (psi_array->data != NULL)
	{
		aligned_free(psi_array->data);
		psi_array->data = NULL;
	}
	psi_array->nqubits = 0;
	psi_array->nstates = 0;
}
