#include <memory.h>
#include <assert.h>
#include "target.h"
#include "numerical_gradient.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		#ifndef COMPLEX_CIRCUIT
		psi_out->data[i] = -1.1 * psi->data[((i + 3) * 113) % n] - 0.7 * psi->data[((i + 9) * 173) % n] + 0.5 * psi->data[i] + 0.3 * psi->data[((i + 4) * 199) % n];
		#else
		psi_out->data[i] = (-1.1 + 0.8*I) * psi->data[((i + 3) * 113) % n] + (0.4 - 0.7*I) * psi->data[((i + 9) * 173) % n] + (0.5 + 0.1*I) * psi->data[i] + (-0.3 + 0.2*I) * psi->data[((i + 4) * 199) % n];
		#endif
	}

	return 0;
}


char* test_unitary_target()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_unitary_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_unitary_target failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		double f;
		if (unitary_target(ufunc, NULL, Vlist, nlayers[i], L, pperms, &f) < 0) {
			return "'unitary_target' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value not match reference";
		}
	}

	H5Fclose(file);

	return 0;
}


struct unitary_target_params
{
	linear_func ufunc;
	int nqubits;
	int nlayers;
	const int** perms;
};

// wrapper of unitary target function
static void unitary_target_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct unitary_target_params* params = p;

	struct mat4x4* Vlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < params->nlayers; i++) {
		memcpy(Vlist[i].data, &x[i * 16], sizeof(Vlist[i].data));
	}

	double f;
	unitary_target(params->ufunc, NULL, Vlist, params->nlayers, params->nqubits, params->perms, &f);
	*y = f;

	aligned_free(Vlist);
}


char* test_unitary_target_and_gradient()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_unitary_target_and_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_unitary_target_and_gradient failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		double f;
		struct mat4x4 dVlist[5];
		if (unitary_target_and_gradient(ufunc, NULL, Vlist, nlayers[i], L, pperms, &f, dVlist) < 0) {
			return "'unitary_target_and_gradient' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value not match reference";
		}

		// numerical gradient
		const double h = 1e-5;
		struct unitary_target_params params = {
			.ufunc = ufunc,
			.nqubits = L,
			.nlayers = nlayers[i],
			.perms = pperms,
		};
		struct mat4x4 dVlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_wirtinger(unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)Vlist, 1, &dy, h, (numeric*)dVlist_num);
		// convert from Wirtinger convention
		for (int j = 0; j < nlayers[i]; j++) {
			for (int k = 0; k < 16; k++) {
				dVlist_num[j].data[k] = 2 * dVlist_num[j].data[k];
			}
		}
		#else
		numerical_gradient(unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)Vlist, 1, &dy, h, (numeric*)dVlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_num[j].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dVlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_ref[j].data) > 1e-12) {
				return "computed target function gradient does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_unitary_target_and_gradient_vector()
{
	int L = 6;
	int nlayers = 3;

	hid_t file = H5Fopen("../test/data/test_unitary_target_and_gradient_vector" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_unitary_target_and_gradient_vector failed";
	}

	struct mat4x4 Vlist[3];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[3][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };

	const int m = nlayers * 16;

	double f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (unitary_target_and_gradient_vector(ufunc, NULL, Vlist, nlayers, L, pperms, &f, grad) < 0) {
		return "'unitary_target_and_gradient_vector' failed internally";
	}

	double f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (fabs(f - f_ref) > 1e-12) {
		return "computed target function value not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	double d = 0;
	for (int i = 0; i < nlayers * 16; i++)
	{
		d = fmax(d, fabs(grad[i] - grad_ref[i]));
	}
	if (d > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	aligned_free(grad_ref);
	aligned_free(grad);
	H5Fclose(file);

	return 0;
}

#endif


char* test_unitary_target_gradient_hessian()
{
	int L = 6;

	hid_t file = H5Fopen("../test/data/test_unitary_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_unitary_target_gradient_hessian failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		const int m = nlayers[i] * 16;

		double f;
		struct mat4x4 dVlist[5];
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (unitary_target_gradient_hessian(ufunc, NULL, Vlist, nlayers[i], L, pperms, &f, dVlist, hess) < 0) {
			return "'unitary_target_gradient_hessian' failed internally";
		}

		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess[j*m + k] - hess[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}
		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value not match reference";
		}

		// numerical gradient
		const double h = 1e-5;
		struct unitary_target_params params = {
			.ufunc = ufunc,
			.nqubits = L,
			.nlayers = nlayers[i],
			.perms = pperms,
		};
		struct mat4x4 dVlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_wirtinger(unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)Vlist, 1, &dy, h, (numeric*)dVlist_num);
		// convert from Wirtinger convention
		for (int j = 0; j < nlayers[i]; j++) {
			for (int k = 0; k < 16; k++) {
				dVlist_num[j].data[k] = 2 * dVlist_num[j].data[k];
			}
		}
		#else
		numerical_gradient(unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)Vlist, 1, &dy, h, (numeric*)dVlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_num[j].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dVlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dVlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dVlist[j].data, dVlist_ref[j].data) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}

		sprintf(varname, "hess%i", i);
		numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess_ref) < 0) {
			return "reading reference Hessian matrix from disk failed";
		}

		// compare with reference
		if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
			return "computed brickwall circuit Hessian matrix does not match reference";
		}

		aligned_free(hess_ref);
		aligned_free(hess);
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_unitary_target_gradient_vector_hessian_matrix()
{
	int L = 6;
	int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_unitary_target_gradient_vector_hessian_matrix" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_unitary_target_gradient_vector_hessian_matrix failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	const int m = nlayers * 16;

	double f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	double* H = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));

	if (unitary_target_gradient_vector_hessian_matrix(ufunc, NULL, Vlist, nlayers, L, pperms, &f, grad, H) < 0) {
		return "'unitary_target_gradient_vector_hessian_matrix' failed internally";
	}

	double f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (fabs(f - f_ref) > 1e-12) {
		return "computed target function value not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	double d = 0;
	for (int i = 0; i < m; i++)
	{
		d = fmax(d, fabs(grad[i] - grad_ref[i]));
	}
	if (d > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	// check symmetry
	double es = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			es = fmax(es, fabs(H[i * m + j] - H[j * m + i]));
		}
	}
	if (es > 1e-14) {
		return "computed gate Hessian matrix is not symmetric";
	}

	double* H_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H_ref == NULL) {
		return "memory allocation for reference Hessian matrix failed";
	}
	if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H_ref) < 0) {
		return "reading reference Hessian matrix from disk failed";
	}

	// compare with reference
	d = 0;
	for (int i = 0; i < m * m; i++)
	{
		d = fmax(d, fabs(H[i] - H_ref[i]));
	}
	if (d > 1e-13) {
		return "computed brickwall circuit Hessian matrix does not match reference";
	}

	aligned_free(H_ref);
	aligned_free(grad_ref);
	aligned_free(H);
	aligned_free(grad);

	H5Fclose(file);

	return 0;
}

#endif


char* test_blockenc_target()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_blockenc_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_blockenc_target failed";
	}

	struct mat4x4 Vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		double f;
		if (blockenc_target(ufunc, NULL, Vlist, nlayers[i], L, pperms, &f) < 0) {
			return "'blockenc_target' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value not match reference";
		}
	}

	H5Fclose(file);

	return 0;
}
