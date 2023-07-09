#pragma once

#include "matrix.h"
#include "statevector.h"
#include "parallel_gates.h"


int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);

int apply_adjoint_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers,
	const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out);


//________________________________________________________________________________________________________________________
///
/// \brief Temporary cache required by backward pass of a brickwall circuit,
/// storing the sequence of intermediate statevectors.
///
struct brickwall_unitary_cache
{
	int nqubits;
	int nstates;
	struct statevector* psi_list;
};

int allocate_brickwall_unitary_cache(const int nqubits, const int nstates, struct brickwall_unitary_cache* cache);

void free_brickwall_unitary_cache(struct brickwall_unitary_cache* cache);

int brickwall_unitary_forward(const struct mat4x4 Vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct brickwall_unitary_cache* cache, struct statevector* restrict psi_out);

int brickwall_unitary_backward(const struct mat4x4 Vlist[], int nlayers, const int* perms[], const struct brickwall_unitary_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dVlist[]);

int brickwall_unitary_backward_hessian(const struct mat4x4 Vlist[], int nlayers, const int* perms[], const struct brickwall_unitary_cache* cache,
	const struct statevector* restrict dpsi_out, struct statevector* restrict dpsi, struct mat4x4 dVlist[], numeric* hess);


int brickwall_unitary_grad_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	unitary_func Ufunc, void* fdata, const int* perms[], struct mat4x4 Glist[]);

#ifdef COMPLEX_CIRCUIT

int brickwall_unitary_gradient_vector_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	unitary_func Ufunc, void* fdata, const int* perms[], double* grad_vec);

#endif

int brickwall_unitary_hess_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	const struct mat4x4* Z, int k, unitary_func Ufunc, void* fdata, const int* perms[], bool unitary_proj, struct mat4x4 dVlist[]);

#ifdef COMPLEX_CIRCUIT

int brickwall_unitary_hessian_matrix_matfree(const struct mat4x4 Vlist[], int nlayers, int L,
	unitary_func Ufunc, void* fdata, const int* perms[], double* H);

#endif
