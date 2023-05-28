#pragma once

#include <stdlib.h>
#include <malloc.h>


#ifdef COMPLEX_CIRCUIT

#include <complex.h>

/// \brief numeric scalar type (used for quantum gates and statevectors)
typedef double complex numeric;

#define _abs cabs

#else

#include <math.h>

/// \brief numeric scalar type (used for quantum gates and statevectors)
typedef double numeric;

#define _abs fabs
#define creal
#define conj

#endif


#define MEM_DATA_ALIGN 64


#ifdef _WIN32

inline void* aligned_alloc(size_t alignment, size_t size)
{
	return _aligned_malloc(size, alignment);
}

inline void aligned_free(void* memblock)
{
	_aligned_free(memblock);
}

#else

static inline void aligned_free(void* memblock)
{
	free(memblock);
}

#endif


/// \brief integer type for addressing entries of a quantum statevector 
typedef long int intqs;
