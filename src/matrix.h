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


void conjugate_matrix(struct mat4x4* a);


void add_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);
void sub_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);

void add_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);
void sub_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);


void transpose(const struct mat4x4* restrict a, struct mat4x4* restrict at);
void adjoint(const struct mat4x4* restrict a, struct mat4x4* restrict ah);


void symm(const struct mat4x4* restrict w, struct mat4x4* restrict z);
void antisymm(const struct mat4x4* restrict w, struct mat4x4* restrict z);


#ifdef COMPLEX_CIRCUIT
static const int num_tangent_params = 16;
#else
static const int num_tangent_params = 6;
#endif

void real_to_antisymm(const double* r, struct mat4x4* w);
void antisymm_to_real(const struct mat4x4* w, double* r);

void real_to_tangent(const double* r, const struct mat4x4* restrict v, struct mat4x4* restrict z);
void tangent_to_real(const struct mat4x4* restrict v, const struct mat4x4* restrict z, double* r);


void project_tangent(const struct mat4x4* restrict u, const struct mat4x4* restrict z, struct mat4x4* restrict p);


void multiply_matrices(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);


void symmetric_triple_matrix_product(const struct mat4x4* restrict a, const struct mat4x4* restrict b, const struct mat4x4* restrict c, struct mat4x4* restrict ret);


int inverse_matrix(const struct mat4x4* restrict a, struct mat4x4* restrict ainv);


void polar_factor(const struct mat4x4* restrict a, struct mat4x4* restrict u);


void retract(const struct mat4x4* restrict u, const double* restrict eta, struct mat4x4* restrict v);
