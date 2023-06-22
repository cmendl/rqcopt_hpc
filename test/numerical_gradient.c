#include <memory.h>
#include "numerical_gradient.h"


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the gradient via difference quotient (f(x + h e_i) - f(x - h e_i)) / (2 h)
///
void numerical_gradient(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const double h, numeric* restrict grad)
{
	numeric *xmod = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(numeric));
	memcpy(xmod, x, n * sizeof(numeric));

	// y = f(x)
	numeric *y = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(numeric));

	for (int i = 0; i < n; i++)
	{
		numeric ydotdy;

		// f(x + h e_i)
		xmod[i] = x[i] + h;
		f(xmod, params, y);
		ydotdy = 0;
		for (int j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] = ydotdy;

		// f(x - h e_i)
		xmod[i] = x[i] - h;
		f(xmod, params, y);
		ydotdy = 0;
		for (int j = 0; j < m; j++) {
			ydotdy += dy[j] * y[j];
		}
		grad[i] -= ydotdy;

		grad[i] /= (2*h);

		// restore original value
		xmod[i] = x[i];
	}

	aligned_free(y);
	aligned_free(xmod);
}
