#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between vectors 'x' and 'y'.
///
double uniform_distance(size_t n, const numeric* x, const numeric* y)
{
	double d = 0;
	for (size_t i = 0; i < n; i++)
	{
		d = fmax(d, _abs(x[i] - y[i]));
	}

	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Maximum element-wise relative distance of two vectors.
///
double relative_distance(size_t n, const numeric* x, const numeric* y, double eps)
{
	double d = 0;
	for (size_t i = 0; i < n; i++)
	{
		d = fmax(d, reldist(x[i], y[i], eps));
	}

	return d;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the inverse of 'perm', a permutation of the numbers 0, 1, ..., n-1.
///
void inverse_permutation(int n, const int* restrict perm, int* restrict inv_perm)
{
	for (int i = 0; i < n; i++)
	{
		inv_perm[perm[i]] = i;
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Read an HDF5 dataset from a file.
///
herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data)
{
	hid_t dset = H5Dopen(file, name, H5P_DEFAULT);
	if (dset < 0)
	{
		fprintf(stderr, "'H5Aopen' for '%s' failed, return value: %" PRId64 "\n", name, dset);
		return -1;
	}

	herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0)
	{
		fprintf(stderr, "'H5Aread' failed, return value: %d\n", status);
		return status;
	}

	return 0;
}
