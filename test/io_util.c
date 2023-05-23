#include <inttypes.h>
#include <stdio.h>
#include "io_util.h"


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
