#include <memory.h>
#include "gate.h"
#include "numerical_gradient.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_gate()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_gate" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate failed";
	}

	struct statevector psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector(L, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector(L, &chi3ref) < 0) { return "memory allocation failed"; }

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi1", H5T_NATIVE_DOUBLE, chi1ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi2", H5T_NATIVE_DOUBLE, chi2ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi3", H5T_NATIVE_DOUBLE, chi3ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// apply the gate
	apply_gate(&V, 2, 5, &psi, &chi1);
	apply_gate(&V, 4, 1, &psi, &chi2);
	apply_gate(&V, 3, 4, &psi, &chi3);

	// compare with reference
	if (uniform_distance((size_t)1 << L, chi1.data, chi1ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((size_t)1 << L, chi2.data, chi2ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }
	if (uniform_distance((size_t)1 << L, chi3.data, chi3ref.data) > 1e-12) { return "quantum state after applying gate does not match reference"; }

	free_statevector(&chi3ref);
	free_statevector(&chi3);
	free_statevector(&chi2ref);
	free_statevector(&chi2);
	free_statevector(&chi1ref);
	free_statevector(&chi1);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


struct apply_gate_psi_params
{
	int nqubits;
	const struct mat4x4* gate;
	int i, j;
};

// wrapper of apply_gate as a function of 'psi'
static void apply_gate_psi(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct apply_gate_psi_params* params = p;

	struct statevector psi;
	allocate_statevector(params->nqubits, &psi);
	memcpy(psi.data, x, ((size_t)1 << params->nqubits) * sizeof(numeric));

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	apply_gate(params->gate, params->i, params->j, &psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
	free_statevector(&psi);
}

struct apply_gate_v_params
{
	const struct statevector* psi;
	int i, j;
};

// wrapper of apply_gate as a function of the gate
static void apply_gate_v(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct apply_gate_v_params* params = p;

	struct statevector psi_out;
	allocate_statevector(params->psi->nqubits, &psi_out);

	apply_gate((struct mat4x4*)x, params->i, params->j, params->psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << psi_out.nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
}

char* test_apply_gate_backward()
{
	int L = 9;

	hid_t file = H5Fopen("../test/data/test_apply_gate_backward" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate_backward failed";
	}
	
	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	// input statevector
	struct statevector psi;
	if (allocate_statevector(L, &psi) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct statevector dpsi_out;
	if (allocate_statevector(L, &dpsi_out) < 0) { return "memory allocation failed"; }
	if (read_hdf5_dataset(file, "dpsi_out", H5T_NATIVE_DOUBLE, dpsi_out.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	const int i_list[3] = { 2, 4, 3 };
	const int j_list[3] = { 5, 1, 4 };

	for (int k = 0; k < 3; k++)
	{
		// backward pass
		struct statevector dpsi;
		if (allocate_statevector(L, &dpsi) < 0) { return "memory allocation failed"; }
		struct mat4x4 dV;
		apply_gate_backward(&V, i_list[k], j_list[k], &psi, &dpsi_out, &dpsi, &dV);

		const double h = 1e-5;

		// numerical gradient with respect to 'psi'
		struct apply_gate_psi_params params_psi = {
			.nqubits = L,
			.gate = &V,
			.i = i_list[k],
			.j = j_list[k],
		};
		struct statevector dpsi_num;
		if (allocate_statevector(L, &dpsi_num) < 0) { return "memory allocation failed"; }
		numerical_gradient(apply_gate_psi, &params_psi, (size_t)1 << L, psi.data, (size_t)1 << L, dpsi_out.data, h, dpsi_num.data);

		// compare
		if (uniform_distance((size_t)1 << L, dpsi.data, dpsi_num.data) > 1e-8) {
			return "gradient of 'apply_gate' with respect to 'psi' does not match finite difference approximation";
		}

		// numerical gradient with respect to the gate
		struct apply_gate_v_params params_v = {
			.psi = &psi,
			.i = i_list[k],
			.j = j_list[k],
		};
		struct mat4x4 dV_num;
		numerical_gradient(apply_gate_v, &params_v, 16, V.data, (size_t)1 << L, dpsi_out.data, h, dV_num.data);

		// compare
		if (uniform_distance(16, dV.data, dV_num.data) > 1e-8) {
			return "gradient of 'apply_gate' with respect to 'V' does not match finite difference approximation";
		}

		free_statevector(&dpsi_num);
		free_statevector(&dpsi);
	}

	free_statevector(&dpsi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


char* test_apply_gate_to_array()
{
	int L = 6;
	int nstates = 5;

	hid_t file = H5Fopen("../test/data/test_apply_gate_to_array" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate_to_array failed";
	}

	struct statevector_array psi, chi1, chi1ref, chi2, chi2ref, chi3, chi3ref;
	if (allocate_statevector_array(L, nstates, &psi)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi1)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi1ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi2)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi2ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi3)    < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, nstates, &chi3ref) < 0) { return "memory allocation failed"; }

	struct mat4x4 V;
	if (read_hdf5_dataset(file, "V", H5T_NATIVE_DOUBLE, V.data) < 0) {
		return "reading two-qubit quantum gate entries from disk failed";
	}

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi1", H5T_NATIVE_DOUBLE, chi1ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi2", H5T_NATIVE_DOUBLE, chi2ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "chi3", H5T_NATIVE_DOUBLE, chi3ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// apply the gate
	apply_gate_to_array(&V, 2, 5, &psi, &chi1);
	apply_gate_to_array(&V, 4, 1, &psi, &chi2);
	apply_gate_to_array(&V, 3, 4, &psi, &chi3);

	// compare with reference
	if (uniform_distance(((size_t)1 << L) * nstates, chi1.data, chi1ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }
	if (uniform_distance(((size_t)1 << L) * nstates, chi2.data, chi2ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }
	if (uniform_distance(((size_t)1 << L) * nstates, chi3.data, chi3ref.data) > 1e-12) { return "quantum state array after applying gate does not match reference"; }

	free_statevector_array(&chi3ref);
	free_statevector_array(&chi3);
	free_statevector_array(&chi2ref);
	free_statevector_array(&chi2);
	free_statevector_array(&chi1ref);
	free_statevector_array(&chi1);
	free_statevector_array(&psi);

	H5Fclose(file);

	return 0;
}


char* test_apply_gate_placeholder()
{
	int L = 7;

	hid_t file = H5Fopen("../test/data/test_apply_gate_placeholder" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_gate_placeholder failed";
	}

	struct statevector psi;
	if (allocate_statevector(L, &psi) < 0) { return "memory allocation failed"; }
	struct statevector_array psi_out, psi_out_ref;
	if (allocate_statevector_array(L, 16, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector_array(L, 16, &psi_out_ref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}
	if (read_hdf5_dataset(file, "psi_out", H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
		return "reading output statevector array data from disk failed";
	}

	apply_gate_placeholder(2, 5, &psi, &psi_out);

	// compare with reference
	if (uniform_distance(((size_t)1 << L)*16, psi_out.data, psi_out_ref.data) > 1e-12) {
		return "quantum state array after applying gate placeholder does not match reference";
	}

	free_statevector_array(&psi_out_ref);
	free_statevector_array(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}
