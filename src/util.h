#pragma once

#include <math.h>
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Relative distance (deviation) of two numbers.
///
static inline double reldist(numeric x, numeric y, double eps)
{
	// TODO: complex abs in case 'numeric' is a complex number
	return fabs(x - y) / fmax(fabs(x) + fabs(y), eps);
}


double uniform_distance(size_t n, const numeric* x, const numeric* y);

double relative_distance(size_t n, const numeric* x, const numeric* y, double eps);


void inverse_permutation(int n, const int* restrict perm, int* restrict inv_perm);
