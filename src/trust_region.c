#include <math.h>
#include <memory.h>
#include <cblas.h>
#include <stdio.h>
#include <assert.h>
#include "trust_region.h"
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
///
static void solve_quadratic_equation(double p, double q, double x[2])
{
	// require non-negative discriminant
	assert(p * p - q >= 0);

	if (p == 0)
	{
		double a = sqrt(-q);
		x[0] = -a;
		x[1] = a;
		return;
	}

	double sign = copysign(1.0, p);
	double a = -(p + sign * sqrt(p * p - q));
	double b = q / a;
	if (a <= b)
	{
		x[0] = a;
		x[1] = b;
	}
	else
	{
		x[0] = b;
		x[1] = a;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Move to the unit ball boundary by solving
/// || b + t*d || == radius
/// for t with t > 0.
///
static double move_to_boundary(const double* b, const double* d, int n, double radius)
{
	double dsq = cblas_ddot(n, d, 1, d, 1);
	if (dsq == 0)
	{
		fprintf(stderr, "input vector 'd' is zero\n");
		return 0;
	}
	double p = cblas_ddot(n, b, 1, d, 1) / dsq;
	double q = (cblas_ddot(n, b, 1, b, 1) - radius*radius) / dsq;
	double t[2];
	solve_quadratic_equation(p, q, t);
	if (t[1] < 0) {
		fprintf(stderr, "encountered t < 0: t = %g\n", t[1]);
	}
	return t[1];
}


//________________________________________________________________________________________________________________________
///
/// \brief Truncated CG (tCG) method for the trust-region subproblem:
///    minimize   <grad, z> + 1/2 <z, H z>
///    subject to <z, z> <= radius^2
///
/// References:
///   - Algorithm 11 in:
///     P.-A. Absil, R. Mahony, Rodolphe Sepulchre
///     Optimization Algorithms on Matrix Manifolds
///     Princeton University Press (2008)
///   - Trond Steihaug
///     The conjugate gradient method and trust regions in large scale optimization
///     SIAM Journal on Numerical Analysis 20, 626-637 (1983)
///
bool truncated_cg(const double* grad, const double* hess, int n, double radius, const struct truncated_cg_params* params, double* z)
{
	double* r  = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	double* d  = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	double* Hd = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));

	// r = grad
	memcpy(r, grad, n * sizeof(double));
	double rsq = cblas_ddot(n, r, 1, r, 1);
	const double stoptol = fmax(params->abstol, params->reltol * sqrt(rsq));

	// initialize z = 0 (zero vector)
	memset(z, 0, n * sizeof(double));

	// initialize d = -r
	memcpy(d, r, n * sizeof(double));
	cblas_dscal(n, -1.0, d, 1);

	for (int j = 0; j < params->maxiter; j++)
	{
		// Hd = hess @ d
		cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, hess, n, d, 1, 0, Hd, 1);
		double dHd = cblas_ddot(n, d, 1, Hd, 1);
		double t = move_to_boundary(z, d, n, radius);
		double alpha = rsq / dHd;
		if (dHd <= 0 || alpha > t)
		{
			// return with move to boundary
			// z += t * d
			cblas_daxpy(n, t, d, 1, z, 1);
			aligned_free(Hd);
			aligned_free(d);
			aligned_free(r);
			return true;
		}
		// update iterates
		// r += alpha * Hd
		cblas_daxpy(n, alpha, Hd, 1, r, 1);
		// z += alpha * d
		cblas_daxpy(n, alpha, d, 1, z, 1);
		double rsq_next = cblas_ddot(n, r, 1, r, 1);
		if (sqrt(rsq_next) <= stoptol)
		{
			// early stopping
			break;
		}
		double beta = rsq_next / rsq;
		// d = -r + beta * d
		cblas_dscal(n, beta, d, 1);
		cblas_daxpy(n, -1.0, r, 1, d, 1);
		rsq = rsq_next;
	}

	// early stopping or maxiter reached
	aligned_free(Hd);
	aligned_free(d);
	aligned_free(r);
	return false;
}


//________________________________________________________________________________________________________________________
///
/// \brief Optimization via the Riemannian trust-region (RTR) algorithm.
///
/// Reference:
///     Algorithm 10 in:
///     P.-A. Absil, R. Mahony, Rodolphe Sepulchre
///     Optimization Algorithms on Matrix Manifolds
///     Princeton University Press (2008)
///
void riemannian_trust_region_optimize(target_func f, void* fdata, retract_func retract, void* rdata,
	gradient_func gradfunc, void* graddata, hessian_func hessfunc, void* hessdata, const int n,
	const double* x_init, const int m, const struct rtr_params* params, double* f_iter, double* g_iter, double* x_final)
{
	assert(0 <= params->rho_trust && params->rho_trust < 0.25);

	double* x = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	memcpy(x, x_init, m * sizeof(double));

	double* x_next = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));

	double radius = params->radius_init;
	if (params->gfunc != NULL) {
		g_iter[0] = params->gfunc(x, params->gdata);
	}

	double* grad = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	double* hess = aligned_alloc(MEM_DATA_ALIGN, n * n * sizeof(double));
	double* eta  = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));
	double* Heta = aligned_alloc(MEM_DATA_ALIGN, n * sizeof(double));

	for (int k = 0; k < params->niter; k++)
	{
		gradfunc(x, graddata, grad);
		hessfunc(x, hessdata, hess);
		bool on_boundary = truncated_cg(grad, hess, n, radius, &params->tcg_params, eta);
		retract(x, eta, rdata, x_next);

		double fx = f(x, fdata);
		f_iter[k] = fx;

		// Eq. (7.7)
		// Heta = hess @ eta
		cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, hess, n, eta, 1, 0, Heta, 1);
		double rho = (f(x_next, fdata) - fx) / (cblas_ddot(n, grad, 1, eta, 1) + 0.5 * cblas_ddot(n, eta, 1, Heta, 1));
		if (rho < 0.25) {
			// reduce radius
			radius *= 0.25;
		}
		else if (rho > 0.75 && on_boundary) {
			// enlarge radius
			radius = fmin(2 * radius, params->maxradius);
		}

		if (rho > params->rho_trust) {
			// swap pointers (instead of copying data from x_next to x)
			double* y = x;
			x = x_next;
			x_next = y;
		}

		if (params->gfunc != NULL) {
			g_iter[k + 1] = params->gfunc(x, params->gdata);
		}
	}

	memcpy(x_final, x, m * sizeof(double));

	aligned_free(Heta);
	aligned_free(eta);
	aligned_free(hess);
	aligned_free(grad);
	aligned_free(x_next);
	aligned_free(x);
}
