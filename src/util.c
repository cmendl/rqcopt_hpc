#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between vectors 'x' and 'y'.
///
double uniform_distance(int n, const numeric* x, const numeric* y)
{
	double d = 0;
	for (int i = 0; i < n; i++)
	{
		d = fmax(d, fabs(x[i] - y[i]));
	}

	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum element-wise relative distance of two vectors.
///
double relative_distance(int n, const numeric* x, const numeric* y, double eps)
{
	double d = 0;
	for (int i = 0; i < n; i++)
	{
		d = fmax(d, reldist(x[i], y[i], eps));
	}

	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inverse of 'perm', a permutation of the numbers 0, 1, ..., n-1.
///
void inverse_permutation(int n, const int* restrict perm, int* restrict inv_perm)
{
	for (int i = 0; i < n; i++)
	{
		inv_perm[perm[i]] = i;
	}
}
