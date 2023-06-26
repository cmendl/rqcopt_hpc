#include <assert.h>
#include "target.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


static int ufunc(const struct statevector* restrict psi, void*, struct statevector* restrict psi_out)
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


char* test_target_and_gradient()
{
	int L = 8;

	hid_t file = H5Fopen("../test/data/test_target_and_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_target_and_gradient failed";
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
		if (target_and_gradient(ufunc, NULL, Vlist, nlayers[i], L, pperms, &f, dVlist) < 0) {
			return "'target_and_gradient' failed internally";
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
	}

	H5Fclose(file);

	return 0;
}
