#include "timing.h"
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif


//________________________________________________________________________________________________________________________
///
/// \brief Get current time tick.
///
uint64_t get_ticks()
{
	#ifdef _WIN32

	LARGE_INTEGER t;
	QueryPerformanceCounter(&t);
	return (uint64_t)(t.QuadPart);

	#else

	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)(1000000000ULL * t.tv_sec + t.tv_nsec);

	#endif
}


//________________________________________________________________________________________________________________________
///
/// \brief Get tick resolution.
///
uint64_t get_tick_resolution()
{
	#ifdef _WIN32

	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return (uint64_t)(freq.QuadPart);

	#else // clock_gettime has nanosecond resolution

	return (uint64_t)1000000000ULL;

	#endif
}
