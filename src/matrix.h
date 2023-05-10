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


void add_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);

void sub_matrix(struct mat4x4* restrict a, const struct mat4x4* restrict b);

void scale_matrix(struct mat4x4* restrict a, const numeric x);


void adjoint(const struct mat4x4* restrict a, struct mat4x4* restrict ah);

void symm(const struct mat4x4* restrict w, struct mat4x4* restrict z);
void antisymm(const struct mat4x4* restrict w, struct mat4x4* restrict z);


void multiply(const struct mat4x4* restrict a, const struct mat4x4* restrict b, struct mat4x4* restrict c);


void project_unitary_tangent(const struct mat4x4* restrict u, const struct mat4x4* restrict z, struct mat4x4* restrict p);
