#pragma once

#include <hdf5.h>


herr_t read_hdf5_dataset(hid_t file, const char* name, hid_t mem_type, void* data);
