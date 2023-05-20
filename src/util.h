#pragma once

#include <math.h>
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Relative distance (deviation) of two numbers.
///
static inline double reldist(numeric x, numeric y, double eps)
{
	return _abs(x - y) / fmax(_abs(x) + _abs(y), eps);
}


double uniform_distance(size_t n, const numeric* x, const numeric* y);

double relative_distance(size_t n, const numeric* x, const numeric* y, double eps);


void inverse_permutation(int n, const int* restrict perm, int* restrict inv_perm);
