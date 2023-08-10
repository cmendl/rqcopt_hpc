#pragma once

#include "config.h"


typedef void (*generic_func)(const numeric* restrict x, void* params, numeric* restrict y);


void numerical_gradient(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad);


#ifdef COMPLEX_CIRCUIT

void numerical_gradient_wirtinger(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad);

void numerical_gradient_conjugated_wirtinger(generic_func f, void* params, const int n, const numeric* restrict x, const int m, const numeric* restrict dy, const numeric h, numeric* restrict grad);

#endif
