#include "config.h"
#include "matrix.h"
#include "util.h"
#include "io_util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_symm()
{
	struct mat4x4 w;
	for (int i = 0; i < 16; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		w.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
		#else
		w.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
	}

	struct mat4x4 z;
	symm(&w, &z);

	// reference calculation
	// add adjoint matrix to 'w'
	struct mat4x4 wh;
	adjoint(&w, &wh);
	add_matrix(&w, &wh);
	scale_matrix(&w, 0.5);

	// compare
	if (uniform_distance(16, z.data, w.data) > 1e-14) {
		return "symmetrized matrix does not agree with reference";
	}

	return 0;
}


char* test_antisymm()
{
	struct mat4x4 w;
	for (int i = 0; i < 16; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		w.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
		#else
		w.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
	}

	struct mat4x4 z;
	antisymm(&w, &z);

	// reference calculation
	// subtract adjoint matrix from 'w'
	struct mat4x4 wh;
	adjoint(&w, &wh);
	sub_matrix(&w, &wh);
	scale_matrix(&w, 0.5);

	// compare
	if (uniform_distance(16, z.data, w.data) > 1e-14) {
		return "anti-symmetrized matrix does not agree with reference";
	}

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_real_to_antisymm()
{
	double r[16];
	for (int i = 0; i < 16; i++)
	{
		r[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
	}

	struct mat4x4 w;
	real_to_antisymm(r, &w);

	// 'w' must indeed be anti-symmetric
	struct mat4x4 z;
	antisymm(&w, &z);
	if (uniform_distance(16, w.data, z.data) > 1e-14) {
		return "matrix returned by 'real_to_antisymm' is not anti-symmetric";
	}

	double s[16];
	antisymm_to_real(&w, s);
	// 's' must match 'r'
	double d = 0;
	for (int i = 0; i < 16; i++)
	{
		d = fmax(d, fabs(s[i] - r[i]));
	}
	if (d > 1e-14) {
		return "converting from real to anti-symmetric matrix and back does not result in original matrix";
	}

	return 0;
}

#endif


char* test_multiply()
{
	hid_t file = H5Fopen("../test/data/test_multiply" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_multiply failed";
	}

	struct mat4x4 a;
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 b;
	if (read_hdf5_dataset(file, "b", H5T_NATIVE_DOUBLE, b.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 cref;
	if (read_hdf5_dataset(file, "c", H5T_NATIVE_DOUBLE, cref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 c;
	multiply(&a, &b, &c);
	// compare
	if (uniform_distance(16, c.data, cref.data) > 1e-12) {
		return "matrix product does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}


char* test_project_unitary_tangent()
{
	hid_t file = H5Fopen("../test/data/test_project_unitary_tangent" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_project_unitary_tangent failed";
	}

	struct mat4x4 u;
	if (read_hdf5_dataset(file, "u", H5T_NATIVE_DOUBLE, u.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	struct mat4x4 z;
	if (read_hdf5_dataset(file, "z", H5T_NATIVE_DOUBLE, z.data) < 0) {
		return "reading matrix entries from disk failed";
	}
	struct mat4x4 pref;
	if (read_hdf5_dataset(file, "p", H5T_NATIVE_DOUBLE, pref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 p;
	project_unitary_tangent(&u, &z, &p);
	// compare
	if (uniform_distance(16, p.data, pref.data) > 1e-12) {
		return "projected matrix does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}
