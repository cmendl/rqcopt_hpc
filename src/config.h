#pragma once

#include <stdlib.h>
#include <malloc.h>


/// \brief numeric scalar type
typedef double numeric;


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

inline void aligned_free(void* memblock)
{
	free(memblock);
}

#endif


/// \brief integer type for addressing entries of a quantum statevector 
typedef long int intqs;
