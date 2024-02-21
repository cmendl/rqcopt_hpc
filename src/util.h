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


double uniform_distance_real(long n, const double* x, const double* y);

double uniform_distance(long n, const numeric* x, const numeric* y);

double relative_distance(long n, const numeric* x, const numeric* y, double eps);


herr_t get_hdf5_dataset_ndims(hid_t file, const char* name, int* ndims);

herr_t get_hdf5_dataset_dims(hid_t file, const char* name, hsize_t* dims);

herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t read_hdf5_attribute(hid_t file, const char* name, hid_t mem_type, void* data);

herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data);

herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data);
