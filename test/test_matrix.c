#include "config.h"
#include "matrix.h"
#include "util.h"


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


char* test_real_to_antisymm()
{
	double* r = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	for (int i = 0; i < num_tangent_params; i++)
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

	double* s = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	antisymm_to_real(&w, s);
	// 's' must match 'r'
	double d = 0;
	for (int i = 0; i < num_tangent_params; i++)
	{
		d = fmax(d, fabs(s[i] - r[i]));
	}
	if (d > 1e-14) {
		return "converting from real vector to anti-symmetric matrix and back does not result in original vector";
	}

	aligned_free(s);
	aligned_free(r);

	return 0;
}


char* test_real_to_tangent()
{
	hid_t file = H5Fopen("../test/data/test_real_to_tangent" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_real_to_tangent failed";
	}

	double* r = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	if (read_hdf5_dataset(file, "r", H5T_NATIVE_DOUBLE, r) < 0) {
		return "reading vector entries from disk failed";
	}

	struct mat4x4 v;
	if (read_hdf5_dataset(file, "v", H5T_NATIVE_DOUBLE, v.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 z;
	real_to_tangent(r, &v, &z);

	double* s = aligned_alloc(MEM_DATA_ALIGN, num_tangent_params * sizeof(double));
	tangent_to_real(&v, &z, s);
	// 's' must match 'r'
	double d = 0;
	for (int i = 0; i < num_tangent_params; i++)
	{
		d = fmax(d, fabs(s[i] - r[i]));
	}
	if (d > 1e-14) {
		return "converting from real vector to tangent matrix and back does not result in original vector";
	}

	aligned_free(s);
	aligned_free(r);

	H5Fclose(file);

	return 0;
}


char* test_project_tangent()
{
	hid_t file = H5Fopen("../test/data/test_project_tangent" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_project_tangent failed";
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
	project_tangent(&u, &z, &p);
	// compare
	if (uniform_distance(16, p.data, pref.data) > 1e-12) {
		return "projected matrix does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}


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
	multiply_matrices(&a, &b, &c);
	// compare
	if (uniform_distance(16, c.data, cref.data) > 1e-12) {
		return "matrix product does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}


char* test_inverse_matrix()
{
	struct mat4x4 a;
	for (int i = 0; i < 16; i++)
	{
		#ifdef COMPLEX_CIRCUIT
		a.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1 + ((((11 + i) * 3571) % 541) / 270.0 - 1) * I;
		#else
		a.data[i] = (((5 + i) * 7919) % 229) / 107.0 - 1;
		#endif
	}

	struct mat4x4 ainv;
	inverse_matrix(&a, &ainv);

	struct mat4x4 prod;
	multiply_matrices(&a, &ainv, &prod);

	struct mat4x4 id;
	identity_matrix(&id);

	// must be (close to) identity matrix
	if (uniform_distance(16, prod.data, id.data) > 1e-14) {
		return "matrix times its inverse is not close to identity";
	}

	// generate a singular matrix
	for (int j = 0; j < 4; j++)
	{
		a.data[4*2 + j] = a.data[j];
	}
	int ret = inverse_matrix(&a, &ainv);
	if (ret != -1) {
		return "missing singular matrix indicator";
	}

	return 0;
}


char* test_polar_factor()
{
	hid_t file = H5Fopen("../test/data/test_polar_factor" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_polar_factor failed";
	}

	struct mat4x4 a;
	if (read_hdf5_dataset(file, "a", H5T_NATIVE_DOUBLE, a.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 uref;
	if (read_hdf5_dataset(file, "u", H5T_NATIVE_DOUBLE, uref.data) < 0) {
		return "reading matrix entries from disk failed";
	}

	struct mat4x4 u;
	polar_factor(&a, &u);

	// compare
	if (uniform_distance(16, u.data, uref.data) > 1e-12) {
		return "polar factor does not agree with reference";
	}

	H5Fclose(file);

	return 0;
}
