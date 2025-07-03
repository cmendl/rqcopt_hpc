#include <memory.h>
#include "numerical_gradient.h"


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient via difference quotient (f(x + h e_i) - f(x - h e_i)) / (2 h).
///
void numerical_gradient_backward(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad)
{
	numeric *xmod = aligned_malloc(n * sizeof(numeric));
	memcpy(xmod, x, n * sizeof(numeric));

	// y = f(x)
	numeric *y = aligned_malloc(m * sizeof(numeric));

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


//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the forward-mode gradient via difference quotient (f(x + h dir) - f(x - h dir)) / (2 h).
///
void numerical_gradient_forward(generic_func f, void* params, const int n, const numeric* restrict x, const numeric* restrict dir, const int m, const numeric h, numeric* restrict grad)
{
	numeric *xd = aligned_malloc(n * sizeof(numeric));
	numeric *yp = aligned_malloc(m * sizeof(numeric));
	numeric *yn = aligned_malloc(m * sizeof(numeric));

	// yp = f(x + h*dir)
	for (int i = 0; i < n; i++)
	{
		xd[i] = x[i] + h*dir[i];
	}
	f(xd, params, yp);

	// yn = f(x - h*dir)
	for (int i = 0; i < n; i++)
	{
		xd[i] = x[i] - h*dir[i];
	}
	f(xd, params, yn);

	for (int i = 0; i < m; i++)
	{
		grad[i] = (yp[i] - yn[i]) / (2*h);
	}

	aligned_free(yn);
	aligned_free(yp);
	aligned_free(xd);
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient (Wirtinger convention) via difference quotients.
///
void numerical_gradient_backward_wirtinger(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad)
{
	numerical_gradient_backward(f, params, n, x, m, dy, h, grad);

	// h -> i*h
	numeric *grad_c = aligned_malloc(n * sizeof(numeric));
	numerical_gradient_backward(f, params, n, x, m, dy, h * I, grad_c);
	for (int i = 0; i < n; i++)
	{
		grad[i] = 0.5 * (grad[i] + grad_c[i]);
	}
	aligned_free(grad_c);
}

//________________________________________________________________________________________________________________________
///
/// \brief Numerically approximate the backward-mode gradient (conjugated Wirtinger convention) via difference quotients.
///
void numerical_gradient_backward_conjugated_wirtinger(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad)
{
	numerical_gradient_backward(f, params, n, x, m, dy, h, grad);

	// h -> i*h
	numeric *grad_c = aligned_malloc(n * sizeof(numeric));
	numerical_gradient_backward(f, params, n, x, m, dy, h * I, grad_c);
	for (int i = 0; i < n; i++)
	{
		grad[i] = 0.5 * (grad[i] - grad_c[i]);
	}
	aligned_free(grad_c);
}

#endif
