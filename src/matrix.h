#pragma once

#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief 4x4 matrix in row-major order.
///
struct mat4x4
{
	numeric data[16];  //!< 4x4 matrix entries
};


void zero_matrix(struct mat4x4* a);

void identity_matrix(struct mat4x4* a);


void scale_matrix(struct mat4x4* restrict a, const double x);


void add_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);
void sub_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);

void add_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);
void sub_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);


void adjoint(const struct mat4x4* restrict a, struct mat4x4* restrict ah);

void symm(const struct mat4x4* restrict w, struct mat4x4* restrict z);
void antisymm(const struct mat4x4* restrict w, struct mat4x4* restrict z);


#ifdef COMPLEX_CIRCUIT

void real_to_antisymm(const double* r, struct mat4x4* w);

void antisymm_to_real(const struct mat4x4* w, double* r);

#endif


void multiply_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);


void project_unitary_tangent(const struct mat4x4* restrict u, const struct mat4x4* restrict z, struct mat4x4* restrict p);


int inverse_matrix(const struct mat4x4* restrict a, struct mat4x4* restrict ainv);


void polar_factor(const struct mat4x4* restrict a, struct mat4x4* restrict u);
