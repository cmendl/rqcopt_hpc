#pragma once

#include <stdio.h>
#include <stdbool.h>


int read_data(const char* filename, void* data, const size_t size, const size_t n);

int write_data(const char* filename, const void* data, const size_t size, const size_t n, const bool append);
