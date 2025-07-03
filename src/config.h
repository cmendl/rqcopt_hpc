#pragma once

#include <stdlib.h>


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


//________________________________________________________________________________________________________________________
///
/// \brief Allocate 'size' bytes of uninitialized storage, and return a pointer to the allocated memory block.
///
static inline void* aligned_malloc(size_t size)
{
	#ifdef _WIN32
	return _aligned_malloc(size, MEM_DATA_ALIGN);
	#else
	// round 'size' up to the next multiple of 'MEM_DATA_ALIGN', which must be a power of 2
	return aligned_alloc(MEM_DATA_ALIGN, (size + MEM_DATA_ALIGN - 1) & (-MEM_DATA_ALIGN));
	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Deallocate a previously allocated memory block.
///
static inline void aligned_free(void* memblock)
{
	#ifdef _WIN32
	_aligned_free(memblock);
	#else
	free(memblock);
	#endif
}


/// \brief integer type for addressing entries of a quantum statevector 
typedef long int intqs;
