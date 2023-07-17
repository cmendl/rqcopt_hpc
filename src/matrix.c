#include <memory.h>
#include <stdio.h>
#include "matrix.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Set matrix 'a' to the zero matrix.
///
void zero_matrix(struct mat4x4* a)
{
	memset(a->data, 0, sizeof(a->data));
}


//________________________________________________________________________________________________________________________
///
/// \brief Set matrix 'a' to the identity matrix.
///
void identity_matrix(struct mat4x4* a)
{
	zero_matrix(a);

	a->data[ 0] = 1;
	a->data[ 5] = 1;
	a->data[10] = 1;
	a->data[15] = 1;
}


//________________________________________________________________________________________________________________________
///
/// \brief Scale matrix 'a' by factor 'x'.
///
void scale_matrix(struct mat4x4* restrict a, const double x)
{
	for (int i = 0; i < 16; i++)
	{
		a->data[i] *= x;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Complex-conjugate the entries of a matrix.
///
void conjugate_matrix(struct mat4x4* a)
{
	for (int i = 0; i < 16; i++)
	{
		a->data[i] = conj(a->data[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Add matrix 'b' to matrix 'a'.
///
void add_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b)
{
	for (int i = 0; i < 16; i++)
	{
		a->data[i] += b->data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Subtract matrix 'b' from matrix 'a'.
///
void sub_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b)
{
	for (int i = 0; i < 16; i++)
	{
		a->data[i] -= b->data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Add matrices 'a' and 'b' and store the result in 'c'.
///
void add_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c)
{
	for (int i = 0; i < 16; i++)
	{
		c->data[i] = a->data[i] + b->data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Subtract matrices 'a' and 'b' and store the result in 'c'.
///
void sub_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c)
{
	for (int i = 0; i < 16; i++)
	{
		c->data[i] = a->data[i] - b->data[i];
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the adjoint (conjugate transpose) matrix.
///
void adjoint(const struct mat4x4* restrict a, struct mat4x4* restrict ah)
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			ah->data[4*i + j] = conj(a->data[4*j + i]);
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize a matrix by projecting it onto the symmetric subspace.
///
void symm(const struct mat4x4* restrict w, struct mat4x4* restrict z)
{
	// diagonal entries
	z->data[ 0] = creal(w->data[ 0]);
	z->data[ 5] = creal(w->data[ 5]);
	z->data[10] = creal(w->data[10]);
	z->data[15] = creal(w->data[15]);

	// upper triangular part
	z->data[ 1] = 0.5 * (w->data[ 1] + conj(w->data[ 4]));
	z->data[ 2] = 0.5 * (w->data[ 2] + conj(w->data[ 8]));
	z->data[ 3] = 0.5 * (w->data[ 3] + conj(w->data[12]));
	z->data[ 6] = 0.5 * (w->data[ 6] + conj(w->data[ 9]));
	z->data[ 7] = 0.5 * (w->data[ 7] + conj(w->data[13]));
	z->data[11] = 0.5 * (w->data[11] + conj(w->data[14]));

	// lower triangular part
	z->data[ 4] = conj(z->data[ 1]);
	z->data[ 8] = conj(z->data[ 2]);
	z->data[ 9] = conj(z->data[ 6]);
	z->data[12] = conj(z->data[ 3]);
	z->data[13] = conj(z->data[ 7]);
	z->data[14] = conj(z->data[11]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Antisymmetrize a matrix by projecting it onto the antisymmetric (skew-symmetric) subspace.
///
void antisymm(const struct mat4x4* restrict w, struct mat4x4* restrict z)
{
	// diagonal entries
	#ifdef COMPLEX_CIRCUIT
	z->data[ 0] = I * cimag(w->data[ 0]);
	z->data[ 5] = I * cimag(w->data[ 5]);
	z->data[10] = I * cimag(w->data[10]);
	z->data[15] = I * cimag(w->data[15]);
	#else
	z->data[ 0] = 0;
	z->data[ 5] = 0;
	z->data[10] = 0;
	z->data[15] = 0;
	#endif

	// upper triangular part
	z->data[ 1] = 0.5 * (w->data[ 1] - conj(w->data[ 4]));
	z->data[ 2] = 0.5 * (w->data[ 2] - conj(w->data[ 8]));
	z->data[ 3] = 0.5 * (w->data[ 3] - conj(w->data[12]));
	z->data[ 6] = 0.5 * (w->data[ 6] - conj(w->data[ 9]));
	z->data[ 7] = 0.5 * (w->data[ 7] - conj(w->data[13]));
	z->data[11] = 0.5 * (w->data[11] - conj(w->data[14]));

	// lower triangular part
	z->data[ 4] = -conj(z->data[ 1]);
	z->data[ 8] = -conj(z->data[ 2]);
	z->data[ 9] = -conj(z->data[ 6]);
	z->data[12] = -conj(z->data[ 3]);
	z->data[13] = -conj(z->data[ 7]);
	z->data[14] = -conj(z->data[11]);
}


#ifdef COMPLEX_CIRCUIT


//________________________________________________________________________________________________________________________
///
/// \brief Map a real-valued square matrix to an anti-symmetric matrix of the same dimension.
///
void real_to_antisymm(const double* r, struct mat4x4* w)
{
	// diagonal entries
	w->data[ 0] = I * r[ 0];
	w->data[ 5] = I * r[ 5];
	w->data[10] = I * r[10];
	w->data[15] = I * r[15];

	// upper triangular part
	w->data[ 1] = 0.5*(r[ 1] - r[ 4]) + 0.5*I*(r[ 1] + r[ 4]);
	w->data[ 2] = 0.5*(r[ 2] - r[ 8]) + 0.5*I*(r[ 2] + r[ 8]);
	w->data[ 3] = 0.5*(r[ 3] - r[12]) + 0.5*I*(r[ 3] + r[12]);
	w->data[ 6] = 0.5*(r[ 6] - r[ 9]) + 0.5*I*(r[ 6] + r[ 9]);
	w->data[ 7] = 0.5*(r[ 7] - r[13]) + 0.5*I*(r[ 7] + r[13]);
	w->data[11] = 0.5*(r[11] - r[14]) + 0.5*I*(r[11] + r[14]);

	// lower triangular part
	w->data[ 4] = -conj(w->data[ 1]);
	w->data[ 8] = -conj(w->data[ 2]);
	w->data[ 9] = -conj(w->data[ 6]);
	w->data[12] = -conj(w->data[ 3]);
	w->data[13] = -conj(w->data[ 7]);
	w->data[14] = -conj(w->data[11]);
}


//________________________________________________________________________________________________________________________
///
/// \brief Map an anti-symmetric matrix to a real-valued square matrix of the same dimension.
///
void antisymm_to_real(const struct mat4x4* w, double* r)
{
	for (int i = 0; i < 16; i++)
	{
		r[i] = creal(w->data[i]) + cimag(w->data[i]);
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Map a real-valued square matrix to a tangent vector of the unitary matrix manifold at point 'v'.
///
void real_to_unitary_tangent(const double* r, const struct mat4x4* restrict v, struct mat4x4* restrict z)
{
	struct mat4x4 a;
	real_to_antisymm(r, &a);
	multiply_matrices(v, &a, z);
}


//________________________________________________________________________________________________________________________
///
/// \brief Map a tangent vector of the unitary matrix manifold at point 'v' to a real-valued square matrix.
///
void unitary_tangent_to_real(const struct mat4x4* restrict v, const struct mat4x4* restrict z, double* r)
{
	struct mat4x4 w, t;
	adjoint(v, &w);
	multiply_matrices(&w, z, &t);
	antisymm(&t, &w);
	antisymm_to_real(&w, r);
}


#else // COMPLEX_CIRCUIT not defined


//________________________________________________________________________________________________________________________
///
/// \brief Isometrically map a real vector with 6 entries to a real skew-symmetric matrix.
///
void real_to_skew(const double* r, struct mat4x4* w)
{
	// diagonal entries
	w->data[ 0] = 0;
	w->data[ 5] = 0;
	w->data[10] = 0;
	w->data[15] = 0;

	// 1/sqrt(2) factor to preserve inner products
	const double inv_sqrt2 = 0.70710678118654752;

	// upper triangular part
	w->data[ 1] = inv_sqrt2 * r[0];
	w->data[ 2] = inv_sqrt2 * r[1];
	w->data[ 3] = inv_sqrt2 * r[2];
	w->data[ 6] = inv_sqrt2 * r[3];
	w->data[ 7] = inv_sqrt2 * r[4];
	w->data[11] = inv_sqrt2 * r[5];

	// lower triangular part
	w->data[ 4] = -w->data[ 1];
	w->data[ 8] = -w->data[ 2];
	w->data[ 9] = -w->data[ 6];
	w->data[12] = -w->data[ 3];
	w->data[13] = -w->data[ 7];
	w->data[14] = -w->data[11];
}


//________________________________________________________________________________________________________________________
///
/// \brief Isometrically map a real skew-symmetric matrix to a real vector with 6 entries.
///
void skew_to_real(const struct mat4x4* w, double* r)
{
	// sqrt(2) factor to preserve inner products
	const double sqrt2 = 1.41421356237309505;

	r[0] = sqrt2 * w->data[ 1];
	r[1] = sqrt2 * w->data[ 2];
	r[2] = sqrt2 * w->data[ 3];
	r[3] = sqrt2 * w->data[ 6];
	r[4] = sqrt2 * w->data[ 7];
	r[5] = sqrt2 * w->data[11];
}


//________________________________________________________________________________________________________________________
///
/// \brief Map a real vector with 6 entries to a tangent vector of the orthogonal matrix manifold at point 'v'.
///
void real_to_ortho_tangent(const double* r, const struct mat4x4* restrict v, struct mat4x4* restrict z)
{
	struct mat4x4 a;
	real_to_skew(r, &a);
	multiply_matrices(v, &a, z);
}


//________________________________________________________________________________________________________________________
///
/// \brief Map a tangent vector of the orthogonal matrix manifold at point 'v' to a real vector with 6 entries.
///
void ortho_tangent_to_real(const struct mat4x4* restrict v, const struct mat4x4* restrict z, double* r)
{
	struct mat4x4 w, t;
	adjoint(v, &w);
	multiply_matrices(&w, z, &t);
	antisymm(&t, &w);
	skew_to_real(&w, r);
}


#endif


//________________________________________________________________________________________________________________________
///
/// \brief Multiply two 4x4 matrices.
///
void multiply_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c)
{
	zero_matrix(c);

	// straightforward implementation; not performance critical, so not switching to BLAS yet
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			for (int k = 0; k < 4; k++)
			{
				c->data[4*i + k] += a->data[4*i + j] * b->data[4*j + k];
			}
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Project 'z' onto the tangent plane at the unitary matrix 'u'.
///
void project_unitary_tangent(const struct mat4x4* restrict u, const struct mat4x4* restrict z, struct mat4x4* restrict p)
{
	// formula remains valid for 'u' an isometry (element of the Stiefel manifold)

	struct mat4x4 v, w;

	adjoint(u, &v);
	multiply_matrices(&v, z, &w);   // w = u^{\dagger} @ z
	symm(&w, &v);                   // v = symm(w)
	multiply_matrices(u, &v, &w);   // w = u @ v
	sub_matrices(z, &w, p);         // p = z - w
}


//________________________________________________________________________________________________________________________
///
/// \brief Swap two rows of a matrix.
///
static inline void swap_rows(struct mat4x4* a, int i, int j)
{
	for (int k = 0; k < 4; k++)
	{
		numeric tmp      = a->data[4*i + k];
		a->data[4*i + k] = a->data[4*j + k];
		a->data[4*j + k] = tmp;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inverse matrix by Gaussian elimination, returning -1 if the matrix is singular.
///
int inverse_matrix(const struct mat4x4* restrict a, struct mat4x4* restrict ainv)
{
	// copy 'a' (for applying row operations)
	struct mat4x4 m;
	memcpy(m.data, a->data, sizeof(m.data));

	identity_matrix(ainv);

	for (int k = 0; k < 4; k++)
	{
		// search for pivot element in k-th column, starting from entry at (k, k)
		int i_max = k;
		double p = _abs(m.data[4*k + k]);
		for (int i = k + 1; i < 4; i++) {
			if (_abs(m.data[4*i + k]) > p) {
				i_max = i;
				p = _abs(m.data[4*i + k]);
			}
		}
		if (p == 0) {
			// matrix is singular
			return -1;
		}
		if (p < 1e-12) {
			fprintf(stderr, "warning: encountered an almost singular matrix in 'inverse_matrix', p = %g\n", p);
		}

		// swap pivot row with current row
		if (i_max != k)
		{
			swap_rows(&m,   k, i_max);
			swap_rows(ainv, k, i_max);
		}

		for (int i = 0; i < 4; i++)
		{
			if (i == k) {
				continue;
			}

			numeric s = m.data[4*i + k] / m.data[4*k + k];

			// subtract 's' times k-th row from i-th row
			for (int j = k + 1; j < 4; j++) {
				m.data[4*i + j] -= s * m.data[4*k + j];
			}
			m.data[4*i + k] = 0;

			// apply same row operation to 'ainv'
			for (int j = 0; j < 4; j++) {
				ainv->data[4*i + j] -= s * ainv->data[4*k + j];
			}
		}

		// divide k-th row by m[k, k]
		for (int j = k + 1; j < 4; j++) {
			m.data[4*k + j] /= m.data[4*k + k];
		}
		for (int j = 0; j < 4; j++) {
			ainv->data[4*k + j] /= m.data[4*k + k];
		}
		m.data[4*k + k] = 1;
	}

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the unitary polar factor 'u' in the polar decomposition 'a = u p' of a matrix,
/// assuming that 'a' is not singular.
///
/// Reference:
///     Nicholas J. Higham
///     Computing the polar decomposition - with applications
///     SIAM J. Sci. Stat. Comput. 7, 1160 - 1174 (1986)
///
void polar_factor(const struct mat4x4* restrict a, struct mat4x4* restrict u)
{
	memcpy(u->data, a->data, sizeof(u->data));

	for (int k = 0; k < 14; k++)
	{
		// w = u^{-\dagger}
		struct mat4x4 v, w;
		inverse_matrix(u, &v);
		adjoint(&v, &w);

		// u = (u + w)/2
		add_matrix(u, &w);
		scale_matrix(u, 0.5);

		if (k >= 4) {
			// early stopping
			if (uniform_distance(16, u->data, w.data) < 1e-14) {
				break;
			}
		}
	}
}
