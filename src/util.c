#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Uniform distance (infinity norm) between real vectors 'x' and 'y'.
///
double uniform_distance_real(size_t n, const double* x, const double* y)
{
	double d = 0;
	for (size_t i = 0; i < n; i++)
	{
		d = fmax(d, fabs(x[i] - y[i]));
	}

	return d;
}


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

	H5Dclose(dset);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 dataset to a file.
///
herr_t write_hdf5_dataset(hid_t file, const char* name, int degree, const hsize_t dims[], hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate_simple(degree, dims, NULL);
	if (space < 0) {
		fprintf(stderr, "'H5Screate_simple' failed, return value: %" PRId64 "\n", space);
		return -1;
	}
	
	// property list to disable time tracking
	hid_t cplist = H5Pcreate(H5P_DATASET_CREATE);
	herr_t status = H5Pset_obj_track_times(cplist, 0);
	if (status < 0) {
		fprintf(stderr, "creating property list failed, return value: %d\n", status);
		return status;
	}

	// create dataset
	hid_t dset = H5Dcreate(file, name, mem_type_store, space, H5P_DEFAULT, cplist, H5P_DEFAULT);
	if (dset < 0) {
		fprintf(stderr, "'H5Dcreate' failed, return value: %" PRId64 "\n", dset);
		return -1;
	}

	// write the data to the dataset
	status = H5Dwrite(dset, mem_type_input, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
	if (status < 0) {
		fprintf(stderr, "'H5Dwrite' failed, return value: %d\n", status);
		return status;
	}

	H5Dclose(dset);
	H5Sclose(space);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write an HDF5 scalar attribute to a file.
///
herr_t write_hdf5_scalar_attribute(hid_t file, const char* name, hid_t mem_type_store, hid_t mem_type_input, const void* data)
{
	// create dataspace
	hid_t space = H5Screate(H5S_SCALAR);
	if (space < 0) {
		fprintf(stderr, "'H5Screate' failed, return value: %" PRId64 "\n", space);
		return -1;
	}

	// create attribute
	hid_t attr = H5Acreate(file, name, mem_type_store, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0) {
		fprintf(stderr, "'H5Acreate' failed, return value: %" PRId64 "\n", attr);
		return -1;
	}

	herr_t status = H5Awrite(attr, mem_type_input, data);
	if (status < 0) {
		fprintf(stderr, "'H5Awrite' failed, return value: %d\n", status);
		return status;
	}

	H5Aclose(attr);
	H5Sclose(space);

	return 0;
}
