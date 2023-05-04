#include "file_io.h"


//________________________________________________________________________________________________________________________
///
/// \brief Read 'n' items of size 'size' from file 'filename', expecting the file size to be exactly n*size.
///
int read_data(const char* filename, void* data, const size_t size, const size_t n)
{
	FILE* fd;
	errno_t ret = fopen_s(&fd, filename, "rb");
	if (ret != 0)
	{
		fprintf(stderr, "'fopen_s' failed during call of 'read_data', error code %d\n", ret);
		return -1;
	}

	// obtain the file size
	fseek(fd, 0, SEEK_END);
	long filesize = ftell(fd);
	rewind(fd);
	if ((size_t)filesize != n*size)
	{
		fprintf(stderr, "'read_data' failed: expected file size does not match\n");
		return -2;
	}

	// copy the file into the data array
	if (fread(data, size, n, fd) != n)
	{
		fprintf(stderr, "'fread' failed during call of 'read_data'\n");
		return -3;
	}

	fclose(fd);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Write 'n' items of size 'size' to file 'filename'.
///
int write_data(const char* filename, const void* data, const size_t size, const size_t n, const bool append)
{
	const char* mode = append ? "ab" : "wb";

	FILE* fd;
	fopen_s(&fd, filename, mode);
	if (fd == NULL)
	{
		fprintf(stderr, "'fopen_s' failed during call of 'write_data'\n");
		return -1;
	}

	// write data array to file
	if (fwrite(data, size, n, fd) != n)
	{
		fprintf(stderr, "'fwrite' failed during call of 'write_data'\n");
		return -3;
	}

	fclose(fd);

	return 0;
}
