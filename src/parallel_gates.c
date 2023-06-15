#include <memory.h>
#include <assert.h>
#include "parallel_gates.h"
#include "gate.h"
#include "util.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply a parallel sequence of two-qubit gates V to state psi,
/// optionally using a permutation of quantum wires.
///
int apply_parallel_gates(const struct mat4x4* V, const struct statevector* restrict psi, const int* perm,
	struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	const int L = psi->nqubits;
	assert(L % 2 == 0);

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	// temporary statevector
	struct statevector tmp = { 0 };
	if (allocate_statevector(L, &tmp) < 0) {
		return -1;
	}

	for (int i = 0; i < L; i += 2)
	{
		int p = ((L - i) / 2) % 2;
		const struct statevector* psi0 = (i == 0 ? psi : (p == 0 ? psi_out : &tmp));
		struct statevector* psi1 = (p == 0 ? &tmp : psi_out);
		apply_gate(V, inv_perm[i], inv_perm[i + 1], psi0, psi1);
	}

	free_statevector(&tmp);
	aligned_free(inv_perm);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the gradient of V x ... x V in direction Z to state psi.
///
int apply_parallel_gates_directed_grad(const struct mat4x4* V, const struct mat4x4* Z,
	const struct statevector* restrict psi, const int* perm, struct statevector* restrict psi_out)
{
	assert(psi->nqubits == psi_out->nqubits);
	const int L = psi->nqubits;
	assert(L % 2 == 0);

	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, psi->nqubits * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	// temporary statevectors
	struct statevector chi = { 0 };
	if (allocate_statevector(L, &chi) < 0) {
		return -1;
	}
	struct statevector tmp = { 0 };
	if (allocate_statevector(L, &tmp) < 0) {
		return -1;
	}

	for (int i = 0; i < L; i += 2)
	{
		struct statevector* phi = (i == 0 ? psi_out : &chi);

		for (int j = 0; j < L; j += 2)
		{
			int p = ((L - j) / 2) % 2;
			const struct statevector* psi0 = (j == 0 ? psi : (p == 0 ? phi : &tmp));
			struct statevector* psi1 = (p == 0 ? &tmp : phi);
			apply_gate(i == j ? Z : V, inv_perm[j], inv_perm[j + 1], psi0, psi1);
		}

		if (i > 0)
		{
			// accumulate statevectors
			const intqs n = (intqs)1 << L;
			for (intqs j = 0; j < n; j++)
			{
				psi_out->data[j] += chi.data[j];
			}
		}
	}

	free_statevector(&tmp);
	free_statevector(&chi);
	aligned_free(inv_perm);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Compute the gradient of Re tr[U^{\dagger} (V x ... x V)] with respect to V,
/// using the provided matrix-free application of U to a state.
///
int parallel_gates_grad_matfree(const struct mat4x4* restrict V, int L, unitary_func Ufunc, void* fdata, const int* perm, struct mat4x4* restrict G)
{
	assert(L >= 2);
	assert(L % 2 == 0);

	bool is_identity_perm = true;
	for (int i = 0; i < L; i++) {
		if (perm[i] != i) {
			is_identity_perm = false;
			break;
		}
	}
	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
	inverse_permutation(L, perm, inv_perm);

	struct statevector psi0 = { 0 };
	struct statevector psi1 = { 0 };
	if (allocate_statevector(L, &psi0) < 0) { return -1; }
	if (allocate_statevector(L, &psi1) < 0) { return -1; }

	memset(G->data, 0, sizeof(G->data));

	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi0.data, 0, n * sizeof(numeric));
		if (is_identity_perm)
		{
			psi0.data[b] = 1;
		}
		else
		{
			// transpose unit vector with entry 1 at b
			intqs a = 0;
			for (int i = 0; i < L; i++) {
				a += ((b >> i) & 1) * ((intqs)1 << (L - 1 - inv_perm[L - 1 - i]));
			}
			psi0.data[a] = 1;
		}
		// apply U
		int ret = Ufunc(&psi0, fdata, &psi1);
		if (ret < 0) {
			return ret;
		}
		struct statevector* psi;
		struct statevector* chi;
		if (is_identity_perm) {
			psi = &psi1;
			chi = &psi0;
		}
		else {
			transpose_statevector(&psi1, inv_perm, &psi0);
			psi = &psi0;
			chi = &psi1;
		}
		for (int i = 0; i < L; i += 2)
		{
			// use chi->data as temporary memory
			numeric* r = chi->data;
			numeric* s = psi->data;

			// inner product with (V x ... x V) |b>, where the i-th V is omitted
			for (int j = 0; j < i; j += 2)
			{
				numeric x = conj(V->data[     ((b >> j) & 3)]);
				numeric y = conj(V->data[ 4 + ((b >> j) & 3)]);
				numeric z = conj(V->data[ 8 + ((b >> j) & 3)]);
				numeric w = conj(V->data[12 + ((b >> j) & 3)]);

				const intqs m = (intqs)1 << (L - j - 2);
				for (intqs a = 0; a < m; a++)
				{
					r[a] = (
						s[4*a    ] * x +
						s[4*a + 1] * y +
						s[4*a + 2] * z +
						s[4*a + 3] * w);
				}
				// avoid overwriting psi->data
				if (s == psi->data) {
					s = chi->data + (n >> 2);
				}
				// swap pointers r <-> s
				numeric* t = r;
				r = s;
				s = t;
			}
			for (int j = i + 2; j < L; j += 2)
			{
				numeric x = conj(V->data[     ((b >> j) & 3)]);
				numeric y = conj(V->data[ 4 + ((b >> j) & 3)]);
				numeric z = conj(V->data[ 8 + ((b >> j) & 3)]);
				numeric w = conj(V->data[12 + ((b >> j) & 3)]);

				const intqs m = (intqs)1 << (L - j - 2);
				for (intqs a = 0; a < m; a++)
				{
					for (intqs l = 0; l < 4; l++)
					{
						r[4*a + l] = (
							s[4*(4*a    ) + l] * x +
							s[4*(4*a + 1) + l] * y +
							s[4*(4*a + 2) + l] * z +
							s[4*(4*a + 3) + l] * w);
					}
				}
				// avoid overwriting psi->data
				if (s == psi->data) {
					s = chi->data + (n >> 2);
				}
				// swap pointers r <-> s
				numeric* t = r;
				r = s;
				s = t;
			}
			G->data[     ((b >> i) & 3)] += s[0];
			G->data[ 4 + ((b >> i) & 3)] += s[1];
			G->data[ 8 + ((b >> i) & 3)] += s[2];
			G->data[12 + ((b >> i) & 3)] += s[3];
		}
	}

	free_statevector(&psi1);
	free_statevector(&psi0);
	aligned_free(inv_perm);

	return 0;
}


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
/// \brief Compute the Hessian of V -> Re tr[U^{\dagger} (V x ... x V)] in direction Z,
/// using the provided matrix-free application of U to a state.
///
int parallel_gates_hess_matfree(const struct mat4x4* restrict V, int L, const struct mat4x4* restrict Z,
	unitary_func Ufunc, void* fdata, const int* perm, bool unitary_proj, struct mat4x4* restrict G)
{
	assert(L >= 2);
	assert(L % 2 == 0);

	bool is_identity_perm = true;
	for (int i = 0; i < L; i++) {
		if (perm[i] != i) {
			is_identity_perm = false;
			break;
		}
	}
	int* inv_perm = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
	inverse_permutation(L, perm, inv_perm);
	
	struct statevector psi0 = { 0 };
	struct statevector psi1 = { 0 };
	if (allocate_statevector(L, &psi0) < 0) { return -1; }
	if (allocate_statevector(L, &psi1) < 0) { return -1; }

	memset(G->data, 0, sizeof(G->data));

	// implement trace via summation over unit vectors
	const intqs n = (intqs)1 << L;
	for (intqs b = 0; b < n; b++)
	{
		memset(psi0.data, 0, n * sizeof(numeric));
		if (is_identity_perm)
		{
			psi0.data[b] = 1;
		}
		else
		{		
			// transpose unit vector with entry 1 at b
			intqs a = 0;
			for (int i = 0; i < L; i++) {
				a += ((b >> i) & 1) * ((intqs)1 << (L - 1 - inv_perm[L - 1 - i]));
			}
			psi0.data[a] = 1;
		}
		// apply U
		int ret = Ufunc(&psi0, fdata, &psi1);
		if (ret < 0) {
			return ret;
		}
		struct statevector* psi;
		struct statevector* chi;
		if (is_identity_perm) {
			psi = &psi1;
			chi = &psi0;
		}
		else {
			transpose_statevector(&psi1, inv_perm, &psi0);
			psi = &psi0;
			chi = &psi1;
		}
		for (int i = 0; i < L; i += 2)
		{
			for (int j = 0; j < L; j += 2)
			{
				if (j == i) {
					continue;
				}

				// use chi->data as temporary memory
				numeric* r = chi->data;
				numeric* s = psi->data;

				for (int k = 0; k < i; k += 2)
				{
					const numeric* data = (k == j ? Z->data : V->data);
					numeric x = conj(data[     ((b >> k) & 3)]);
					numeric y = conj(data[ 4 + ((b >> k) & 3)]);
					numeric z = conj(data[ 8 + ((b >> k) & 3)]);
					numeric w = conj(data[12 + ((b >> k) & 3)]);

					const intqs m = (intqs)1 << (L - k - 2);
					for (intqs a = 0; a < m; a++)
					{
						r[a] = (
							s[4*a    ] * x +
							s[4*a + 1] * y +
							s[4*a + 2] * z +
							s[4*a + 3] * w);
					}
					// avoid overwriting psi->data
					if (s == psi->data) {
						s = chi->data + (n >> 2);
					}
					// swap pointers r <-> s
					numeric* t = r;
					r = s;
					s = t;
				}
				for (int k = i + 2; k < L; k += 2)
				{
					const numeric* data = (k == j ? Z->data : V->data);
					numeric x = conj(data[     ((b >> k) & 3)]);
					numeric y = conj(data[ 4 + ((b >> k) & 3)]);
					numeric z = conj(data[ 8 + ((b >> k) & 3)]);
					numeric w = conj(data[12 + ((b >> k) & 3)]);

					const intqs m = (intqs)1 << (L - k - 2);
					for (intqs a = 0; a < m; a++)
					{
						for (intqs l = 0; l < 4; l++)
						{
							r[4*a + l] = (
								s[4*(4*a    ) + l] * x +
								s[4*(4*a + 1) + l] * y +
								s[4*(4*a + 2) + l] * z +
								s[4*(4*a + 3) + l] * w);
						}
					}
					// avoid overwriting psi->data
					if (s == psi->data) {
						s = chi->data + (n >> 2);
					}
					// swap pointers r <-> s
					numeric* t = r;
					r = s;
					s = t;
				}
				G->data[     ((b >> i) & 3)] += s[0];
				G->data[ 4 + ((b >> i) & 3)] += s[1];
				G->data[ 8 + ((b >> i) & 3)] += s[2];
				G->data[12 + ((b >> i) & 3)] += s[3];
			}
		}
	}

	if (unitary_proj)
	{
		struct mat4x4 Gproj;
		project_unitary_tangent(V, G, &Gproj);
		memcpy(G->data, Gproj.data, sizeof(G->data));
		// additional terms resulting from the projection of the gradient
		// onto the Stiefel manifold (unitary matrices)
		struct mat4x4 grad;
		parallel_gates_grad_matfree(V, L, Ufunc, fdata, perm, &grad);
		struct mat4x4 gradh;
		adjoint(&grad, &gradh);
		// G -= 0.5 * (Z @ grad^{\dagger} @ V + V @ grad^{\dagger} @ Z)
		struct mat4x4 T;
		symmetric_triple_matrix_product(Z, &gradh, V, &T);
		sub_matrix(G, &T);

		struct mat4x4 Zproj;
		project_unitary_tangent(V, Z, &Zproj);
		if (uniform_distance(16, Z->data, Zproj.data) > 1e-14)
		{
			// G -= 0.5 * (Z @ V^{\dagger} + V @ Z^{\dagger}) @ grad
			struct mat4x4 W;
			adjoint(V, &W);
			multiply_matrices(Z, &W, &T);
			symm(&T, &W);
			multiply_matrices(&W, &grad, &T);
			sub_matrix(G, &T);
		}
	}

	free_statevector(&psi1);
	free_statevector(&psi0);
	aligned_free(inv_perm);

	return 0;
}
