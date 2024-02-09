#include <memory.h>
#include <cblas.h>
#include <assert.h>
#include "config.h"
#include "matrix.h"
#include "statevector.h"
#include "quantum_circuit.h"
#include "numerical_gradient.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


char* test_apply_quantum_circuit()
{
	const int nqubits = 8;
	const int ngates  = 4;

	hid_t file = H5Fopen("../test/data/test_apply_quantum_circuit" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_apply_quantum_circuit failed";
	}

	struct statevector psi, psi_out, psi_out_ref;
	if (allocate_statevector(nqubits, &psi)         < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &psi_out_ref) < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	struct mat4x4* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	for (int i = 0; i < ngates; i++)
	{
		char varname[32];
		sprintf(varname, "G%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, gates[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		return "reading wire indices from disk failed";
	}

	// apply quantum circuit
	if (apply_quantum_circuit(gates, ngates, wires, &psi, &psi_out) < 0) {
		return "'apply_quantum_circuit' failed internally";
	}

	if (read_hdf5_dataset(file, "psi_out", H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}

	// compare with reference
	if (uniform_distance((size_t)1 << nqubits, psi_out.data, psi_out_ref.data) > 1e-12) {
		return "quantum state after applying quantum circuit does not match reference";
	}

	aligned_free(wires);
	aligned_free(gates);
	free_statevector(&psi_out_ref);
	free_statevector(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}


struct quantum_circuit_forward_psi_params
{
	const struct mat4x4* gates;
	const int* wires;
	int nqubits;
	int ngates;
};

// wrapper of quantum_circuit_forward as a function of 'psi'
static void quantum_circuit_forward_psi(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct quantum_circuit_forward_psi_params* params = p;

	struct statevector psi;
	allocate_statevector(params->nqubits, &psi);
	memcpy(psi.data, x, ((size_t)1 << params->nqubits) * sizeof(numeric));

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	apply_quantum_circuit(params->gates, params->ngates, params->wires, &psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
	free_statevector(&psi);
}

struct quantum_circuit_forward_gates_params
{
	const int* wires;
	const struct statevector* psi;
	int nqubits;
	int ngates;
};

// wrapper of quantum_circuit_forward as a function of the gates
static void quantum_circuit_forward_gates(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct quantum_circuit_forward_gates_params* params = p;

	struct statevector psi_out;
	allocate_statevector(params->nqubits, &psi_out);

	apply_quantum_circuit((struct mat4x4*)x, params->ngates, params->wires, params->psi, &psi_out);
	memcpy(y, psi_out.data, ((size_t)1 << params->nqubits) * sizeof(numeric));

	free_statevector(&psi_out);
}


char* test_quantum_circuit_backward()
{
	const int nqubits = 6;
	const int ngates  = 5;

	hid_t file = H5Fopen("../test/data/test_quantum_circuit_backward" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_quantum_circuit_backward failed";
	}

	struct statevector psi, psi_out, psi_out_ref, dphi, dpsi;
	if (allocate_statevector(nqubits, &psi)         < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &psi_out)     < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &psi_out_ref) < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &dphi)        < 0) { return "memory allocation failed"; }
	if (allocate_statevector(nqubits, &dpsi)        < 0) { return "memory allocation failed"; }

	if (read_hdf5_dataset(file, "psi", H5T_NATIVE_DOUBLE, psi.data) < 0) {
		return "reading input statevector data from disk failed";
	}

	if (read_hdf5_dataset(file, "dphi", H5T_NATIVE_DOUBLE, dphi.data) < 0) {
		return "reading upstream gradient data from disk failed";
	}

	struct mat4x4* gates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	for (int i = 0; i < ngates; i++)
	{
		char varname[32];
		sprintf(varname, "G%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, gates[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		return "reading wire indices from disk failed";
	}

	struct quantum_circuit_cache cache;
	if (allocate_quantum_circuit_cache(nqubits, ngates, &cache) < 0) {
		return "'allocate_quantum_circuit_cache' failed";
	}

	// quantum circuit forward pass
	if (quantum_circuit_forward(gates, ngates, wires, &psi, &cache, &psi_out) < 0) {
		return "'quantum_circuit_forward' failed internally";
	}

	if (read_hdf5_dataset(file, "psi_out", H5T_NATIVE_DOUBLE, psi_out_ref.data) < 0) {
		return "reading output statevector data from disk failed";
	}
	// compare output state of forward pass with reference
	if (uniform_distance((size_t)1 << nqubits, psi_out.data, psi_out_ref.data) > 1e-12) {
		return "quantum state after applying quantum circuit does not match reference";
	}

	// brickwall unitary backward pass
	struct mat4x4* dgates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	if (quantum_circuit_backward(gates, ngates, wires, &cache, &dphi, &dpsi, dgates) < 0) {
		return "'quantum_circuit_backward' failed internally";
	}

	const double h = 1e-5;

	// numerical gradient with respect to 'psi'
	struct quantum_circuit_forward_psi_params params_psi = {
		.gates   = gates,
		.wires   = wires,
		.nqubits = nqubits,
		.ngates  = ngates,
	};
	struct statevector dpsi_num;
	if (allocate_statevector(nqubits, &dpsi_num) < 0) { return "memory allocation failed"; }
	numerical_gradient(quantum_circuit_forward_psi, &params_psi, (size_t)1 << nqubits, psi.data, (size_t)1 << nqubits, dphi.data, h, dpsi_num.data);
	// compare
	if (uniform_distance((size_t)1 << nqubits, dpsi.data, dpsi_num.data) > 1e-8) {
		return "gradient with respect to 'psi' computed by 'quantum_circuit_backward' does not match finite difference approximation";
	}

	// numerical gradient with respect to gates
	struct quantum_circuit_forward_gates_params params_gates = {
		.psi     = &psi,
		.wires   = wires,
		.nqubits = nqubits,
		.ngates  = ngates,
	};
	struct mat4x4* dgates_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	numerical_gradient(quantum_circuit_forward_gates, &params_gates, ngates * 16, (numeric*)gates, (size_t)1 << nqubits, dphi.data, h, (numeric*)dgates_num);
	// compare
	if (uniform_distance(ngates * 16, (numeric*)dgates, (numeric*)dgates_num) > 1e-8) {
		return "gradient with respect to gates computed by 'quantum_circuit_backward' does not match finite difference approximation";
	}

	aligned_free(dgates_num);
	free_statevector(&dpsi_num);
	aligned_free(dgates);
	free_quantum_circuit_cache(&cache);
	aligned_free(wires);
	aligned_free(gates);
	free_statevector(&dpsi);
	free_statevector(&dphi);
	free_statevector(&psi_out_ref);
	free_statevector(&psi_out);
	free_statevector(&psi);

	H5Fclose(file);

	return 0;
}
