#pragma once

#include <math.h>
#include <hdf5.h>
#include "config.h"


//________________________________________________________________________________________________________________________
///
/// \brief Relative distance (deviation) of two numbers.
///
static inline double reldist(numeric x, numeric y, double eps)
{
	return _abs(x - y) / fmax(_abs(x) + _abs(y), eps);
}


double uniform_distance_real(size_t n, const double* x, const double* y);

double uniform_distance(size_t n, const numeric* x, const numeric* y);

double relative_distance(size_t n, const numeric* x, const numeric* y, double eps);


herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data);
