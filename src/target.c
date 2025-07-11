#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "target.h"
#include "quantum_circuit.h"
#include "brickwall_circuit.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C],
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int circuit_unitary_target(linear_func ufunc, void* udata, const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits, numeric* fval)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi = { 0 };
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		ret = apply_quantum_circuit(gates, ngates, wires, &psi, &Cpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			return -1;
		}

		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}
	}

	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Approximately evaluate target function -tr[U^{\dagger} C],
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state
/// via random state vector sampling.
///
int circuit_unitary_target_sampling(linear_func ufunc, void* udata, const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits, const long nsamples, struct rng_state* rng, numeric* fval)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi = { 0 };
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	const intqs n = (intqs)1 << nqubits;

	numeric f = 0;
	// approximate trace by sampling random state vectors
	for (long b = 0; b < nsamples; b++)
	{
		int ret;

		haar_random_statevector(&psi, rng);

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		ret = apply_quantum_circuit(gates, ngates, wires, &psi, &Cpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_quantum_circuit' failed, return value: %i\n", ret);
			return -1;
		}

		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}
	}

	// re-normalize
	const double scale = (double)n / nsamples;
	f *= scale;

	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C] and its gate gradients,
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int circuit_unitary_target_and_gradient(linear_func ufunc, void* udata, const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits, numeric* fval, struct mat4x4 dgates[])
{
	// temporary statevectors
	struct statevector psi;
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi;
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi;
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache;
	if (allocate_quantum_circuit_cache(nqubits, ngates, &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct mat4x4* dgates_unit = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (dgates_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", ngates);
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		memset(dgates[i].data, 0, sizeof(dgates[i].data));
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		int ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// quantum circuit forward pass
		if (quantum_circuit_forward(gates, ngates, wires, &psi, &cache, &Cpsi) < 0) {
			fprintf(stderr, "'quantum_circuit_forward' failed internally");
			return -3;
		}

		// f += <Upsi | Cpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}

		// quantum circuit backward pass
		// note: overwriting 'psi' with gradient
		if (quantum_circuit_backward(gates, ngates, wires, &cache, &Upsi, &psi, dgates_unit) < 0) {
			fprintf(stderr, "'quantum_circuit_backward' failed internally");
			return -4;
		}
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < ngates; i++)
		{
			add_matrix(&dgates[i], &dgates_unit[i]);
		}
	}

	aligned_free(dgates_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Represent target function -tr[U^{\dagger} C] and its gradient as real vector,
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int circuit_unitary_target_and_projected_gradient(linear_func ufunc, void* udata, const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits, numeric* fval, double* grad_vec)
{
	struct mat4x4* dgates = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (dgates == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	int ret = circuit_unitary_target_and_gradient(ufunc, udata, gates, ngates, wires, nqubits, fval, dgates);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < ngates; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dgates[i]);
		tangent_to_real(&gates[i], &dgates[i], &grad_vec[i * num_tangent_params]);
	}

	aligned_free(dgates);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C], its gate gradients and the Hessian-vector product,
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int circuit_unitary_target_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const struct mat4x4 gatedirs[], const int ngates, const int wires[], const int nqubits,
	numeric* fval, struct mat4x4 dgates[], struct mat4x4 hess_gatedirs[])
{
	// temporary statevectors
	struct statevector psi;
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi;
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Cpsi;
	if (allocate_statevector(nqubits, &Cpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct mat4x4* dgates_unit = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (dgates_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", ngates);
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		memset(dgates[i].data, 0, sizeof(dgates[i].data));
	}

	struct mat4x4* hess_gatedirs_unit = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (hess_gatedirs_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", ngates);
		return -1;
	}

	for (int i = 0; i < ngates; i++)
	{
		memset(hess_gatedirs[i].data, 0, sizeof(hess_gatedirs[i].data));
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		int ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// circuit unitary Hessian-vector product computation
		if (quantum_circuit_gates_hessian_vector_product(gates, gatedirs, ngates, wires, &psi, &Upsi, &Cpsi, dgates_unit, hess_gatedirs_unit) < 0) {
			fprintf(stderr, "'quantum_circuit_gates_hessian_vector_product' failed internally");
			return -3;
		}

		// f += <Upsi | Cpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Cpsi.data[a];
		}

		// accumulate gate gradients and Hessian-vector products for current unit vector
		for (int i = 0; i < ngates; i++)
		{
			add_matrix(&dgates[i], &dgates_unit[i]);
			add_matrix(&hess_gatedirs[i], &hess_gatedirs_unit[i]);
		}
	}

	aligned_free(hess_gatedirs_unit);
	aligned_free(dgates_unit);
	free_statevector(&Cpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} C], its projected gate gradients and the Hessian-vector product,
/// where C is the quantum circuit constructed from two-qubit gates,
/// using the provided matrix-free application of U to a state.
///
int circuit_unitary_target_projected_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const struct mat4x4 gatedirs[], const int ngates, const int wires[], const int nqubits,
	numeric* fval, double* restrict grad_vec, double* restrict hvp_vec)
{
	struct mat4x4* dgates = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (dgates == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	struct mat4x4* hess_gatedirs = aligned_malloc(ngates * sizeof(struct mat4x4));
	if (hess_gatedirs == NULL) {
		fprintf(stderr, "allocating temporary memory for Hessian-vector product matrices failed\n");
		return -1;
	}

	if (circuit_unitary_target_hessian_vector_product(ufunc, udata, gates, gatedirs, ngates, wires, nqubits, fval, dgates, hess_gatedirs) < 0) {
		fprintf(stderr, "'circuit_unitary_target_hessian_vector_product' failed internally");
		return -2;
	}

	for (int i = 0; i < ngates; i++)
	{
		// project gradient onto Stiefel manifold
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dgates[i]);
		tangent_to_real(&gates[i], &dgates[i], &grad_vec[i * num_tangent_params]);

		// project Hessian-vector products
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&hess_gatedirs[i]);
		// additional terms resulting from the projection of the gradient onto the Stiefel manifold
		struct mat4x4 gradh;
		// note: dgates[i] has already been conjugated
		adjoint(&dgates[i], &gradh);
		// D -= 0.5 * (Z @ grad^{\dagger} @ G + G @ grad^{\dagger} @ Z)
		struct mat4x4 pderiv;
		symmetric_triple_matrix_product(&gatedirs[i], &gradh, &gates[i], &pderiv);
		sub_matrix(&hess_gatedirs[i], &pderiv);
		// represent tangent vector of Stiefel manifold at gates[i] as real vector
		tangent_to_real(&gates[i], &hess_gatedirs[i], &hvp_vec[i * num_tangent_params]);
	}

	aligned_free(hess_gatedirs);
	aligned_free(dgates);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Convert a brickwall to a sequential circuit.
///
static inline void brickwall_to_sequential(const int nqubits, const int nlayers, const struct mat4x4* restrict vlist, const int* perms[], struct mat4x4* restrict gates, int* restrict wires)
{
	assert(nqubits % 2 == 0);
	const int ngateslayer = nqubits / 2;

	for (int i = 0; i < nlayers; i++)
	{
		for (int j = 0; j < ngateslayer; j++)
		{
			// duplicate gate vlist[i]
			memcpy(gates[i*ngateslayer + j].data, vlist[i].data, sizeof(vlist[i].data));
			wires[2 * (i*ngateslayer + j)    ] = perms[i][2*j    ];
			wires[2 * (i*ngateslayer + j) + 1] = perms[i][2*j + 1];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W],
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target(linear_func ufunc, void* udata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval)
{
	const int ngates = nlayers * (nqubits / 2);

	struct mat4x4* gates = aligned_malloc(ngates * sizeof(struct mat4x4));
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	int ret = circuit_unitary_target(ufunc, udata, gates, ngates, wires, nqubits, fval);

	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Approximately evaluate target function -tr[U^{\dagger} W],
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state
/// via random state vector sampling.
///
int brickwall_unitary_target_sampling(linear_func ufunc, void* udata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], const long nsamples, struct rng_state* rng, numeric* fval)
{
	const int ngates = nlayers * (nqubits / 2);

	struct mat4x4* gates = aligned_malloc(ngates * sizeof(struct mat4x4));
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	int ret = circuit_unitary_target_sampling(ufunc, udata, gates, ngates, wires, nqubits, nsamples, rng, fval);

	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W] and its gate gradients,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_and_gradient(linear_func ufunc, void* udata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, struct mat4x4 dvlist[])
{
	const int ngates = nlayers * (nqubits / 2);

	struct mat4x4* gates = aligned_malloc(ngates * sizeof(struct mat4x4));
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates, wires);

	struct mat4x4* dgates = aligned_malloc(ngates * sizeof(struct mat4x4));
	int ret = circuit_unitary_target_and_gradient(ufunc, udata, gates, ngates, wires, nqubits, fval, dgates);

	// accumulate gradients
	const int ngateslayer = nqubits / 2;
	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].data, 0, sizeof(dvlist[i].data));

		for (int j = 0; j < ngateslayer; j++)
		{
			add_matrix(&dvlist[i], &dgates[i*ngateslayer + j]);
		}
	}

	aligned_free(dgates);
	aligned_free(wires);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Represent target function -tr[U^{\dagger} W] and its gradient as real vector,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_and_projected_gradient(linear_func ufunc, void* udata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], numeric* fval, double* grad_vec)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	int ret = brickwall_unitary_target_and_gradient(ufunc, udata, vlist, nlayers, nqubits, perms, fval, dvlist);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * num_tangent_params]);
	}

	aligned_free(dvlist);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its gate gradients and the Hessian-vector product,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const struct mat4x4 vdirs[], const int nlayers, const int* perms[], const int nqubits,
	numeric* fval, struct mat4x4 dvlist[], struct mat4x4 hess_vdirs[])
{
	const int ngates = nlayers * (nqubits / 2);

	struct mat4x4* gates    = aligned_malloc(ngates * sizeof(struct mat4x4));
	struct mat4x4* gatedirs = aligned_malloc(ngates * sizeof(struct mat4x4));
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	brickwall_to_sequential(nqubits, nlayers, vlist, perms, gates,    wires);
	brickwall_to_sequential(nqubits, nlayers, vdirs, perms, gatedirs, wires);

	struct mat4x4* dgates = aligned_malloc(ngates * sizeof(struct mat4x4));
	struct mat4x4* hess_gatedirs = aligned_malloc(ngates * sizeof(struct mat4x4));

	int ret = circuit_unitary_target_hessian_vector_product(ufunc, udata, gates, gatedirs, ngates, wires, nqubits, fval, dgates, hess_gatedirs);

	// accumulate gradients and Hessian-vector products
	const int ngateslayer = nqubits / 2;
	for (int i = 0; i < nlayers; i++)
	{
		memset(    dvlist[i].data, 0, sizeof(    dvlist[i].data));
		memset(hess_vdirs[i].data, 0, sizeof(hess_vdirs[i].data));

		for (int j = 0; j < ngateslayer; j++)
		{
			add_matrix(    &dvlist[i],        &dgates[i*ngateslayer + j]);
			add_matrix(&hess_vdirs[i], &hess_gatedirs[i*ngateslayer + j]);
		}
	}

	aligned_free(hess_gatedirs);
	aligned_free(dgates);
	aligned_free(wires);
	aligned_free(gatedirs);
	aligned_free(gates);

	return ret;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its projected gate gradients and the Hessian-vector product,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_projected_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const struct mat4x4 vdirs[], const int nlayers, const int* perms[], const int nqubits,
	numeric* fval, double* restrict grad_vec, double* restrict hvp_vec)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	struct mat4x4* hess_vdirs = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (hess_vdirs == NULL) {
		fprintf(stderr, "allocating temporary memory for Hessian-vector product matrices failed\n");
		return -1;
	}

	if (brickwall_unitary_target_hessian_vector_product(ufunc, udata, vlist, vdirs, nlayers, perms, nqubits, fval, dvlist, hess_vdirs) < 0) {
		fprintf(stderr, "'brickwall_unitary_target_hessian_vector_product' failed internally");
		return -2;
	}

	for (int i = 0; i < nlayers; i++)
	{
		// project gradient onto Stiefel manifold
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * num_tangent_params]);

		// project Hessian-vector products
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&hess_vdirs[i]);
		// additional terms resulting from the projection of the gradient onto the Stiefel manifold
		struct mat4x4 gradh;
		// note: dvlist[i] has already been conjugated
		adjoint(&dvlist[i], &gradh);
		// D -= 0.5 * (Z @ grad^{\dagger} @ G + G @ grad^{\dagger} @ Z)
		struct mat4x4 pderiv;
		symmetric_triple_matrix_product(&vdirs[i], &gradh, &vlist[i], &pderiv);
		sub_matrix(&hess_vdirs[i], &pderiv);
		// represent tangent vector of Stiefel manifold at vlist[i] as real vector
		tangent_to_real(&vlist[i], &hess_vdirs[i], &hvp_vec[i * num_tangent_params]);
	}

	aligned_free(hess_vdirs);
	aligned_free(dvlist);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its gate gradients and Hessian,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, struct mat4x4 dvlist[], numeric* hess)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(nqubits, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct mat4x4* dvlist_unit = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].data, 0, sizeof(dvlist[i].data));
	}

	const int m = nlayers * 16;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric* hess_unit = aligned_malloc(m * m * sizeof(numeric));
	if (hess_unit == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	numeric f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < n; b++)
	{
		int ret;

		memset(psi.data, 0, n * sizeof(numeric));
		psi.data[b] = 1;

		ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Wpsi.data[a];
		}

		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'psi' with gradient
		if (brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}

		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dvlist[i], &dvlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m*m; i++)
		{
			hess[i] += hess_unit[i];
		}
	}

	aligned_free(hess_unit);
	aligned_free(dvlist_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Approximately evaluate target function -tr[U^{\dagger} W], its gate gradients and Hessian,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state
/// via random state vector sampling.
///
int brickwall_unitary_target_gradient_hessian_sampling(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[],
	const long nsamples, struct rng_state* rng,
	numeric* fval, struct mat4x4 dvlist[], numeric* hess)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(nqubits, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(nqubits, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct mat4x4* dvlist_unit = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].data, 0, sizeof(dvlist[i].data));
	}

	const int m = nlayers * 16;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric* hess_unit = aligned_malloc(m * m * sizeof(numeric));
	if (hess_unit == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	const intqs n = (intqs)1 << nqubits;

	numeric f = 0;
	// approximate trace by sampling random state vectors
	for (long b = 0; b < nsamples; b++)
	{
		haar_random_statevector(&psi, rng);

		int ret = ufunc(&psi, udata, &Upsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'ufunc' failed, return value: %i\n", ret);
			return -2;
		}
		// negate and complex-conjugate entries
		for (intqs a = 0; a < n; a++)
		{
			Upsi.data[a] = -conj(Upsi.data[a]);
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// f += <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += Upsi.data[a] * Wpsi.data[a];
		}

		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'psi' with gradient
		if (brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &Upsi, &psi, dvlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}

		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dvlist[i], &dvlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m*m; i++)
		{
			hess[i] += hess_unit[i];
		}
	}

	// re-normalize
	const double scale = (double)n / nsamples;
	f *= scale;
	for (int i = 0; i < nlayers; i++)
	{
		scale_matrix(&dvlist[i], scale);
	}
	for (int i = 0; i < m*m; i++)
	{
		hess[i] *= scale;
	}

	aligned_free(hess_unit);
	aligned_free(dvlist_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Project the Euclidean Hessian according to the unitary Stiefel manifold structure.
///
static inline void brickwall_unitary_project_hessian_matrix(const int nlayers, const struct mat4x4* restrict vlist, const struct mat4x4* restrict dvlist, const numeric* restrict hess, double* restrict H)
{
	const int m = nlayers * 16;

	// project blocks of the Hessian matrix
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = i; j < nlayers; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				// unit vector
				double r[16] = { 0 };
				r[k] = 1;
				struct mat4x4 Z;
				real_to_tangent(r, &vlist[j], &Z);

				// could use zgemv for matrix vector multiplication, but not performance critical
				struct mat4x4 G = { 0 };
				for (int x = 0; x < 16; x++) {
					for (int y = 0; y < 16; y++) {
						G.data[x] += hess[((i*16 + x)*nlayers + j)*16 + y] * Z.data[y];
					}
				}
				// conjugation due to convention for derivative
				conjugate_matrix(&G);
				
				if (i == j)
				{
					struct mat4x4 Gproj;
					project_tangent(&vlist[i], &G, &Gproj);
					memcpy(G.data, Gproj.data, sizeof(G.data));
					// additional terms resulting from the projection of the gradient
					// onto the Stiefel manifold (unitary matrices)
					struct mat4x4 gradh;
					adjoint(&dvlist[i], &gradh);
					// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
					struct mat4x4 T;
					symmetric_triple_matrix_product(&Z, &gradh, &vlist[i], &T);
					sub_matrix(&G, &T);
				}

				// represent tangent vector of Stiefel manifold at vlist[i] as real vector
				tangent_to_real(&vlist[i], &G, r);
				for (int x = 0; x < 16; x++) {
					H[((i*16 + x)*nlayers + j)*16 + k] = r[x];
				}
			}
		}
	}

	// copy upper triangular part according to symmetry
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < i; j++) {
			H[i*m + j] = H[j*m + i];
		}
	}
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -tr[U^{\dagger} W], its gate gradient as real vector and Hessian matrix,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state.
///
int brickwall_unitary_target_gradient_vector_hessian_matrix(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[],
	numeric* fval, double* grad_vec, double* H)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	const int m = nlayers * 16;
	numeric* hess = aligned_malloc(m * m * sizeof(numeric));
	if (hess == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	int ret = brickwall_unitary_target_gradient_hessian(ufunc, udata, vlist, nlayers, nqubits, perms, fval, dvlist, hess);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 16]);
	}

	brickwall_unitary_project_hessian_matrix(nlayers, vlist, dvlist, hess, H);

	aligned_free(hess);
	aligned_free(dvlist);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Approximately evaluate target function -tr[U^{\dagger} W], its gate gradient as real vector and Hessian matrix,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of U to a state
/// via random state vector sampling.
///
int brickwall_unitary_target_gradient_vector_hessian_matrix_sampling(linear_func ufunc, void* udata,
	const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[],
	const long nsamples, struct rng_state* rng,
	numeric* fval, double* grad_vec, double* H)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	const int m = nlayers * 16;
	numeric* hess = aligned_malloc(m * m * sizeof(numeric));
	if (hess == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	int ret = brickwall_unitary_target_gradient_hessian_sampling(ufunc, udata, vlist, nlayers, nqubits, perms, nsamples, rng, fval, dvlist, hess);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 16]);
	}

	brickwall_unitary_project_hessian_matrix(nlayers, vlist, dvlist, hess, H);

	aligned_free(hess);
	aligned_free(dvlist);

	return 0;
}

#endif


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function || P^{\dagger} W P - H ||_F^2 / 2
/// to approximate the matrix H based on block-encoding with projector P,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of H to a state.
///
int brickwall_blockenc_target(linear_func hfunc, void* hdata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], double* fval)
{
	assert(nqubits % 2 == 0);

	// temporary statevectors
	// half number of qubits
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits / 2, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	// half number of qubits
	struct statevector Hpsi = { 0 };
	if (allocate_statevector(nqubits / 2, &Hpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	struct statevector chi = { 0 };
	if (allocate_statevector(nqubits, &chi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wchi = { 0 };
	if (allocate_statevector(nqubits, &Wchi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	double f = 0;
	// implement Frobenius norm via summation over unit vectors
	const intqs s = (intqs)1 << (nqubits / 2);
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < s; b++)
	{
		int ret;

		memset(psi.data, 0, s * sizeof(numeric));
		psi.data[b] = 1;

		ret = hfunc(&psi, hdata, &Hpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'hfunc' failed, return value: %i\n", ret);
			return -2;
		}

		// interleave |0> states (corresponds to application of projector P)
		memset(chi.data, 0, n * sizeof(numeric));
		{
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = b & ((intqs)1 << i);
				c += 2 * t * t;
			}
			chi.data[c] = 1;
		}

		ret = apply_brickwall_unitary(vlist, nlayers, perms, &chi, &Wchi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_brickwall_unitary' failed, return value: %i\n", ret);
			return -1;
		}

		for (intqs a = 0; a < s; a++)
		{
			// index corresponding to applying P^{\dagger}
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = a & ((intqs)1 << i);
				c += 2 * t * t;
			}

			// entry at index 'a' of P^{\dagger} W P |psi> - H |psi>
			numeric d = Wchi.data[c] - Hpsi.data[a];

			#ifdef COMPLEX_CIRCUIT
			f += 0.5 * (creal(d)*creal(d) + cimag(d)*cimag(d));
			#else
			f += 0.5 * d*d;
			#endif
		}
	}

	free_statevector(&Wchi);
	free_statevector(&chi);
	free_statevector(&Hpsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function || P^{\dagger} W P - H ||_F^2 / 2 and its gate gradients
/// to approximate the matrix H based on block-encoding with projector P,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of H to a state.
///
int brickwall_blockenc_target_and_gradient(linear_func hfunc, void* hdata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], double* fval, struct mat4x4 dvlist[])
{
	assert(nqubits % 2 == 0);

	// temporary statevectors
	// half number of qubits
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits / 2, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	// half number of qubits
	struct statevector Hpsi = { 0 };
	if (allocate_statevector(nqubits / 2, &Hpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	struct statevector chi = { 0 };
	if (allocate_statevector(nqubits, &chi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wchi = { 0 };
	if (allocate_statevector(nqubits, &Wchi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct mat4x4* dvlist_unit = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].data, 0, sizeof(dvlist[i].data));
	}

	double f = 0;
	// implement Frobenius norm via summation over unit vectors
	const intqs s = (intqs)1 << (nqubits / 2);
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < s; b++)
	{
		int ret;

		memset(psi.data, 0, s * sizeof(numeric));
		psi.data[b] = 1;

		ret = hfunc(&psi, hdata, &Hpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'hfunc' failed, return value: %i\n", ret);
			return -2;
		}

		// interleave |0> states (corresponds to application of projector P)
		memset(chi.data, 0, n * sizeof(numeric));
		{
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = b & ((intqs)1 << i);
				c += 2 * t * t;
			}
			chi.data[c] = 1;
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(vlist, nlayers, perms, &chi, &cache, &Wchi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// store upstream gradient in 'chi'
		memset(chi.data, 0, n * sizeof(numeric));
		for (intqs a = 0; a < s; a++)
		{
			// index corresponding to applying P^{\dagger}
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = a & ((intqs)1 << i);
				c += 2 * t * t;
			}

			// entry at index 'a' of P^{\dagger} W P |psi> - H |psi>
			numeric d = Wchi.data[c] - Hpsi.data[a];

			#ifdef COMPLEX_CIRCUIT
			f += 0.5 * (creal(d)*creal(d) + cimag(d)*cimag(d));
			#else
			f += 0.5 * d*d;
			#endif

			// overwrite chi with conj(P (P^{\dagger} W P |psi> - H |psi>))
			chi.data[c] = conj(d);
		}

		// brickwall unitary backward pass, using 'chi' as upstream gradient
		// note: overwriting 'Wchi' with gradient
		if (brickwall_unitary_backward(vlist, nlayers, perms, &cache, &chi, &Wchi, dvlist_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward' failed internally");
			return -4;
		}
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dvlist[i], &dvlist_unit[i]);
		}
	}

	aligned_free(dvlist_unit);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wchi);
	free_statevector(&chi);
	free_statevector(&Hpsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Represent target function || P^{\dagger} W P - H ||_F^2 / 2 and its gradient as real vector,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of H to a state.
///
int brickwall_blockenc_target_and_gradient_vector(linear_func hfunc, void* hdata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], double* fval, double* grad_vec)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	int ret = brickwall_blockenc_target_and_gradient(hfunc, hdata, vlist, nlayers, nqubits, perms, fval, dvlist);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 16]);
	}

	aligned_free(dvlist);

	return 0;
}

#endif


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function || P^{\dagger} W P - H ||_F^2 / 2, its gate gradients and Hessian
/// to approximate the matrix H based on block-encoding with projector P,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of H to a state.
///
/// Returning two Hessian matrices for the complex version, since the matrices in 'vlist'
/// appear with and without complex-conjugation in target function.
///
int brickwall_blockenc_target_gradient_hessian(linear_func hfunc, void* hdata,
	const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[],
	double* fval, struct mat4x4 dvlist[],
	#ifdef COMPLEX_CIRCUIT
	numeric* hess1, numeric* hess2
	#else
	numeric* hess
	#endif
	)
{
	// temporary statevectors
	// half number of qubits
	struct statevector psi = { 0 };
	if (allocate_statevector(nqubits / 2, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	// half number of qubits
	struct statevector Hpsi = { 0 };
	if (allocate_statevector(nqubits / 2, &Hpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits / 2);
		return -1;
	}
	struct statevector chi = { 0 };
	if (allocate_statevector(nqubits, &chi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}
	struct statevector Wchi = { 0 };
	if (allocate_statevector(nqubits, &Wchi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", nqubits);
		return -1;
	}

	struct quantum_circuit_cache cache = { 0 };
	if (allocate_quantum_circuit_cache(nqubits, nlayers * (nqubits / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_quantum_circuit_cache' failed");
		return -1;
	}

	struct statevector_array* phi = aligned_malloc(nlayers * sizeof(struct statevector_array));
	if (phi == NULL) {
		fprintf(stderr, "memory allocation for %i statevector_array structs failed\n", nlayers);
		return -1;
	}
	for (int i = 0; i < nlayers; i++)
	{
		// half number of qubits
		if (allocate_statevector_array(nqubits / 2, 16, &phi[i]) < 0) {
			return -1;
		}
	}
	struct statevector_array tmp;
	if (allocate_statevector_array(nqubits, 16, &tmp) < 0) {
		return -1;
	}

	struct mat4x4* dvlist_unit = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dvlist[i].data, 0, sizeof(dvlist[i].data));
	}

	const int m = nlayers * 16;
	#ifdef COMPLEX_CIRCUIT
	memset(hess1, 0, m * m * sizeof(numeric));
	memset(hess2, 0, m * m * sizeof(numeric));
	#else
	memset(hess, 0, m * m * sizeof(numeric));
	#endif

	numeric* hess_unit1 = aligned_malloc(m * m * sizeof(numeric));
	if (hess_unit1 == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}
	numeric* hess_unit2 = aligned_malloc(m * m * sizeof(numeric));
	if (hess_unit2 == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	double f = 0;
	// implement Frobenius norm via summation over unit vectors
	const intqs s = (intqs)1 << (nqubits / 2);
	const intqs n = (intqs)1 << nqubits;
	for (intqs b = 0; b < s; b++)
	{
		int ret;

		memset(psi.data, 0, s * sizeof(numeric));
		psi.data[b] = 1;

		ret = hfunc(&psi, hdata, &Hpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'hfunc' failed, return value: %i\n", ret);
			return -2;
		}

		// interleave |0> states (corresponds to application of projector P)
		memset(chi.data, 0, n * sizeof(numeric));
		{
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = b & ((intqs)1 << i);
				c += 2 * t * t;
			}
			chi.data[c] = 1;
		}

		// brickwall unitary forward pass
		if (brickwall_unitary_forward(vlist, nlayers, perms, &chi, &cache, &Wchi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// store upstream gradient in 'chi'
		memset(chi.data, 0, n * sizeof(numeric));
		for (intqs a = 0; a < s; a++)
		{
			// index corresponding to applying P^{\dagger}
			intqs c = 0;
			for (int i = 0; i < nqubits / 2; i++)
			{
				intqs t = a & ((intqs)1 << i);
				c += 2 * t * t;
			}

			// entry at index 'a' of P^{\dagger} W P |psi> - H |psi>
			numeric d = Wchi.data[c] - Hpsi.data[a];

			#ifdef COMPLEX_CIRCUIT
			f += 0.5 * (creal(d)*creal(d) + cimag(d)*cimag(d));
			#else
			f += 0.5 * d*d;
			#endif

			// overwrite chi with conj(P (P^{\dagger} W P |psi> - H |psi>))
			chi.data[c] = conj(d);
		}

		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'Wchi' with gradient
		if (brickwall_unitary_backward_hessian(vlist, nlayers, perms, &cache, &chi, &Wchi, dvlist_unit, hess_unit1) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}

		// contribution by outer product of gradients to Hessian
		for (int l = 0; l < nlayers; l++)
		{
			if (apply_brickwall_unitary_gate_placeholder(vlist, nlayers, perms, l, &cache, &tmp) < 0) {
				fprintf(stderr, "'apply_brickwall_unitary_gate_placeholder' failed internally");
				return -4;
			}

			// apply P^{\dagger} to output states
			for (intqs a = 0; a < s; a++)
			{
				// index corresponding to applying P^{\dagger}
				intqs c = 0;
				for (int i = 0; i < nqubits / 2; i++)
				{
					intqs t = a & ((intqs)1 << i);
					c += 2 * t * t;
				}

				for (int j = 0; j < 16; j++)
				{
					phi[l].data[a*16 + j] = tmp.data[c*16 + j];
				}
			}
		}
		// outer product
		memset(hess_unit2, 0, m * m * sizeof(numeric));
		for (int l = 0; l < nlayers; l++) {
			for (int k = 0; k <= l; k++) {
				for (intqs a = 0; a < s; a++) {
					for (int i = 0; i < 16; i++) {
						for (int j = 0; j < 16; j++) {
							hess_unit2[((l*16 + i)*nlayers + k)*16 + j] += conj(phi[l].data[a*16 + i]) * phi[k].data[a*16 + j];
						}
					}
				}
			}
		}
		// copy off-diagonal blocks according to symmetry
		for (int l = 0; l < nlayers; l++) {
			for (int k = l + 1; k < nlayers; k++) {
				for (int i = 0; i < 16; i++) {
					for (int j = 0; j < 16; j++) {
						hess_unit2[((l*16 + i)*nlayers + k)*16 + j] = conj(hess_unit2[((k*16 + j)*nlayers + l)*16 + i]);
					}
				}
			}
		}

		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dvlist[i], &dvlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m * m; i++)
		{
			#ifdef COMPLEX_CIRCUIT
			hess1[i] += hess_unit1[i];
			hess2[i] += hess_unit2[i];
			#else
			hess[i] += hess_unit1[i] + hess_unit2[i];
			#endif
		}
	}

	aligned_free(hess_unit2);
	aligned_free(hess_unit1);
	aligned_free(dvlist_unit);
	free_statevector_array(&tmp);
	for (int i = 0; i < nlayers; i++)
	{
		free_statevector_array(&phi[i]);
	}
	aligned_free(phi);
	free_quantum_circuit_cache(&cache);
	free_statevector(&Wchi);
	free_statevector(&chi);
	free_statevector(&Hpsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function || P^{\dagger} W P - H ||_F^2 / 2, its gate gradient as real vector and Hessian matrix
/// to approximate the matrix H based on block-encoding with projector P,
/// where W is the brickwall circuit constructed from the gates in vlist,
/// using the provided matrix-free application of H to a state.
///
int brickwall_blockenc_target_gradient_vector_hessian_matrix(linear_func hfunc, void* hdata, const struct mat4x4 vlist[], const int nlayers, const int nqubits, const int* perms[], double* fval, double* grad_vec, double* H)
{
	struct mat4x4* dvlist = aligned_malloc(nlayers * sizeof(struct mat4x4));
	if (dvlist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	const int m = nlayers * 16;
	numeric* hess1 = aligned_malloc(m * m * sizeof(numeric));
	numeric* hess2 = aligned_malloc(m * m * sizeof(numeric));

	int ret = brickwall_blockenc_target_gradient_hessian(hfunc, hdata, vlist, nlayers, nqubits, perms, fval, dvlist, hess1, hess2);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dvlist[i]);
		tangent_to_real(&vlist[i], &dvlist[i], &grad_vec[i * 16]);
	}

	// project blocks of Hessian matrix
	for (int i = 0; i < nlayers; i++)
	{
		for (int j = i; j < nlayers; j++)
		{
			for (int k = 0; k < 16; k++)
			{
				// unit vector
				double r[16] = { 0 };
				r[k] = 1;
				struct mat4x4 Z;
				real_to_tangent(r, &vlist[j], &Z);

				// could use zgemv for matrix vector multiplication, but not performance critical
				struct mat4x4 G1 = { 0 };
				struct mat4x4 G2 = { 0 };
				for (int x = 0; x < 16; x++) {
					for (int y = 0; y < 16; y++) {
						G1.data[x] += hess1[((i*16 + x)*nlayers + j)*16 + y] * Z.data[y];
						G2.data[x] += hess2[((i*16 + x)*nlayers + j)*16 + y] * Z.data[y];
					}
				}
				// conjugation due to convention for derivative
				conjugate_matrix(&G1);
				struct mat4x4 G = { 0 };
				add_matrices(&G1, &G2, &G);

				if (i == j)
				{
					struct mat4x4 Gproj;
					project_tangent(&vlist[i], &G, &Gproj);
					memcpy(G.data, Gproj.data, sizeof(G.data));
					// additional terms resulting from the projection of the gradient
					// onto the Stiefel manifold (unitary matrices)
					struct mat4x4 gradh;
					adjoint(&dvlist[i], &gradh);
					// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
					struct mat4x4 T;
					symmetric_triple_matrix_product(&Z, &gradh, &vlist[i], &T);
					sub_matrix(&G, &T);
				}

				// represent tangent vector of Stiefel manifold at vlist[i] as real vector
				tangent_to_real(&vlist[i], &G, r);
				for (int x = 0; x < 16; x++) {
					H[((i*16 + x)*nlayers + j)*16 + k] = r[x];
				}
			}
		}
	}

	// copy upper triangular part according to symmetry
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < i; j++) {
			H[i*m + j] = H[j*m + i];
		}
	}

	aligned_free(hess2);
	aligned_free(hess1);
	aligned_free(dvlist);

	return 0;
}

#endif
