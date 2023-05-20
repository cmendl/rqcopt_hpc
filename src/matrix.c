#include <memory.h>
#include "matrix.h"


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
/// \brief Scale matrix 'a' by factor 'x'.
///
void scale_matrix(struct mat4x4* restrict a, const numeric x)
{
	for (int i = 0; i < 16; i++)
	{
		a->data[i] *= x;
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

#endif


//________________________________________________________________________________________________________________________
///
/// \brief Multiply two 4x4 matrices.
///
void multiply(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c)
{
	memset(c->data, 0, sizeof(c->data));

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
	multiply(&v, z, &w);    // w = u^{\dagger} @ z
	symm(&w, &v);           // v = symm(w)
	multiply(u, &v, &w);    // w = u @ v
	sub_matrices(z, &w, p); // p = z - w
}
