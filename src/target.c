#include <memory.h>
#include <stdio.h>
#include "target.h"
#include "brickwall_circuit.h"


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -Re tr[U^{\dagger} W],
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(L, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(L, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(L, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}

	double f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
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

		ret = apply_brickwall_unitary(Vlist, nlayers, &psi, perms, &Wpsi);
		if (ret < 0) {
			fprintf(stderr, "call of 'apply_brickwall_unitary' failed, return value: %i\n", ret);
			return -1;
		}

		// f += Re <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += creal(Upsi.data[a] * Wpsi.data[a]);
		}
	}

	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -Re tr[U^{\dagger} W] and its gate gradients,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target_and_gradient(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval, struct mat4x4 dVlist[])
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(L, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(L, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(L, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}

	struct brickwall_unitary_cache cache = { 0 };
	if (allocate_brickwall_unitary_cache(L, nlayers * (L / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_brickwall_unitary_cache' failed");
		return -1;
	}

	struct mat4x4* dVlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (dVlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));
	}

	double f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
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
		if (brickwall_unitary_forward(Vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// f += Re <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += creal(Upsi.data[a] * Wpsi.data[a]);
		}

		// brickwall unitary backward pass
		// note: overwriting 'psi' with gradient
		if (brickwall_unitary_backward(Vlist, nlayers, perms, &cache, &Upsi, &psi, dVlist_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward' failed internally");
			return -4;
		}
		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dVlist[i], &dVlist_unit[i]);
		}
	}

	aligned_free(dVlist_unit);
	free_brickwall_unitary_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Represent target function -Re tr[U^{\dagger} W] and its gradient as real vector,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target_and_gradient_vector(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval, double* grad_vec)
{
	struct mat4x4* dVlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (dVlist == NULL) {
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	int ret = target_and_gradient(ufunc, udata, Vlist, nlayers, L, perms, fval, dVlist);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dVlist[i]);
		unitary_tangent_to_real(&Vlist[i], &dVlist[i], &grad_vec[i * 16]);
	}

	aligned_free(dVlist);

	return 0;
}

#endif


//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -Re tr[U^{\dagger} W], its gate gradients and Hessian,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target_gradient_hessian(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval, struct mat4x4 dVlist[], numeric* hess)
{
	// temporary statevectors
	struct statevector psi = { 0 };
	if (allocate_statevector(L, &psi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Upsi = { 0 };
	if (allocate_statevector(L, &Upsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}
	struct statevector Wpsi = { 0 };
	if (allocate_statevector(L, &Wpsi) < 0) {
		fprintf(stderr, "memory allocation for a statevector with %i qubits failed\n", L);
		return -1;
	}

	struct brickwall_unitary_cache cache = { 0 };
	if (allocate_brickwall_unitary_cache(L, nlayers * (L / 2), &cache) < 0) {
		fprintf(stderr, "'allocate_brickwall_unitary_cache' failed");
		return -1;
	}

	struct mat4x4* dVlist_unit = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (dVlist_unit == NULL) {
		fprintf(stderr, "memory allocation for %i temporary quantum gates failed\n", nlayers);
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		memset(dVlist[i].data, 0, sizeof(dVlist[i].data));
	}

	const int m = nlayers * 16;
	memset(hess, 0, m * m * sizeof(numeric));

	numeric* hess_unit = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
	if (hess_unit == NULL) {
		fprintf(stderr, "memory allocation for temporary Hessian matrix failed\n");
		return -1;
	}

	double f = 0;
	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
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
		if (brickwall_unitary_forward(Vlist, nlayers, perms, &psi, &cache, &Wpsi) < 0) {
			fprintf(stderr, "'brickwall_unitary_forward' failed internally");
			return -3;
		}

		// f += Re <Upsi | Wpsi>
		for (intqs a = 0; a < n; a++)
		{
			f += creal(Upsi.data[a] * Wpsi.data[a]);
		}

		// brickwall unitary backward pass and Hessian computation
		// note: overwriting 'psi' with gradient
		if (brickwall_unitary_backward_hessian(Vlist, nlayers, perms, &cache, &Upsi, &psi, dVlist_unit, hess_unit) < 0) {
			fprintf(stderr, "'brickwall_unitary_backward_hessian' failed internally");
			return -4;
		}

		// accumulate gate gradients for current unit vector
		for (int i = 0; i < nlayers; i++)
		{
			add_matrix(&dVlist[i], &dVlist_unit[i]);
		}

		// accumulate Hessian matrix for current unit vector
		for (int i = 0; i < m*m; i++)
		{
			hess[i] += hess_unit[i];
		}
	}

	aligned_free(hess_unit);
	aligned_free(dVlist_unit);
	free_brickwall_unitary_cache(&cache);
	free_statevector(&Wpsi);
	free_statevector(&Upsi);
	free_statevector(&psi);

	*fval = f;

	return 0;
}


#ifdef COMPLEX_CIRCUIT

//________________________________________________________________________________________________________________________
///
/// \brief Compute (a @ b @ c + c @ b @ a) / 2.
///
static void symmetric_triple_matrix_product(const struct mat4x4* restrict a, const struct mat4x4* restrict b, const struct mat4x4* restrict c, struct mat4x4* restrict ret)
{
	struct mat4x4 u, v;

	multiply_matrices(a, b, &u);
	multiply_matrices(&u, c, ret);

	multiply_matrices(c, b, &u);
	multiply_matrices(&u, a, &v);

	add_matrix(ret, &v);
	scale_matrix(ret, 0.5);
}

//________________________________________________________________________________________________________________________
///
/// \brief Evaluate target function -Re tr[U^{\dagger} W], its gate gradient as real vector and Hessian matrix,
/// where W is the brickwall circuit constructed from the gates in Vlist,
/// using the provided matrix-free application of U to a state.
///
int target_gradient_vector_hessian_matrix(unitary_func ufunc, void* udata, const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[], double* fval, double* grad_vec, double* H)
{
	struct mat4x4* dVlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (dVlist == NULL)
	{
		fprintf(stderr, "allocating temporary memory for gradient matrices failed\n");
		return -1;
	}

	const int m = nlayers * 16;
	numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));

	int ret = target_gradient_hessian(ufunc, udata, Vlist, nlayers, L, perms, fval, dVlist, hess);
	if (ret < 0) {
		return ret;
	}

	// project gradient onto unitary manifold, represent as anti-symmetric matrix and then concatenate to a vector
	for (int i = 0; i < nlayers; i++)
	{
		// conjugate gate gradient entries (by derivative convention)
		conjugate_matrix(&dVlist[i]);
		unitary_tangent_to_real(&Vlist[i], &dVlist[i], &grad_vec[i * 16]);
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
				real_to_unitary_tangent(r, &Vlist[j], &Z);

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
					project_unitary_tangent(&Vlist[i], &G, &Gproj);
					memcpy(G.data, Gproj.data, sizeof(G.data));
					// additional terms resulting from the projection of the gradient
					// onto the Stiefel manifold (unitary matrices)
					struct mat4x4 gradh;
					adjoint(&dVlist[i], &gradh);
					// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
					struct mat4x4 T;
					symmetric_triple_matrix_product(&Z, &gradh, &Vlist[i], &T);
					sub_matrix(&G, &T);
				}

				// represent tangent vector of Stiefel manifold at Vlist[i] as real vector
				unitary_tangent_to_real(&Vlist[i], &G, r);
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

	aligned_free(hess);
	aligned_free(dVlist);

	return 0;
}

#endif
