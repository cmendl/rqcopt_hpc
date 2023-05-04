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
