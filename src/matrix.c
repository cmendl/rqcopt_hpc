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
			// TODO: complex conjugation
			ah->data[4*i + j] = a->data[4*j + i];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Symmetrize a matrix by projecting it onto the symmetric subspace.
///
void symm(const struct mat4x4* restrict w, struct mat4x4* restrict z)
{
	// copy diagonal entries
	z->data[ 0] = w->data[ 0];
	z->data[ 5] = w->data[ 5];
	z->data[10] = w->data[10];
	z->data[15] = w->data[15];

	// upper triangular part
	// TODO: complex conjugation
	z->data[ 1] = 0.5 * (w->data[ 1] + w->data[ 4]);
	z->data[ 2] = 0.5 * (w->data[ 2] + w->data[ 8]);
	z->data[ 3] = 0.5 * (w->data[ 3] + w->data[12]);
	z->data[ 6] = 0.5 * (w->data[ 6] + w->data[ 9]);
	z->data[ 7] = 0.5 * (w->data[ 7] + w->data[13]);
	z->data[11] = 0.5 * (w->data[11] + w->data[14]);

	// lower triangular part
	// TODO: complex conjugation
	z->data[ 4] = z->data[ 1];
	z->data[ 8] = z->data[ 2];
	z->data[ 9] = z->data[ 6];
	z->data[12] = z->data[ 3];
	z->data[13] = z->data[ 7];
	z->data[14] = z->data[11];
}


//________________________________________________________________________________________________________________________
///
/// \brief Antisymmetrize a matrix by projecting it onto the antisymmetric (skew-symmetric) subspace.
///
void antisymm(const struct mat4x4* restrict w, struct mat4x4* restrict z)
{
	// diagonal entries
	// TODO: imaginary part
	z->data[ 0] = 0;
	z->data[ 5] = 0;
	z->data[10] = 0;
	z->data[15] = 0;

	// upper triangular part
	// TODO: complex conjugation
	z->data[ 1] = 0.5 * (w->data[ 1] - w->data[ 4]);
	z->data[ 2] = 0.5 * (w->data[ 2] - w->data[ 8]);
	z->data[ 3] = 0.5 * (w->data[ 3] - w->data[12]);
	z->data[ 6] = 0.5 * (w->data[ 6] - w->data[ 9]);
	z->data[ 7] = 0.5 * (w->data[ 7] - w->data[13]);
	z->data[11] = 0.5 * (w->data[11] - w->data[14]);

	// lower triangular part
	// TODO: complex conjugation
	z->data[ 4] = -z->data[ 1];
	z->data[ 8] = -z->data[ 2];
	z->data[ 9] = -z->data[ 6];
	z->data[12] = -z->data[ 3];
	z->data[13] = -z->data[ 7];
	z->data[14] = -z->data[11];
}


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
