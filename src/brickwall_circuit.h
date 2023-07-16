#pragma once

#include "matrix.h"
#include "statevector.h"


int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers, const int* perms[],
	const struct statevector* restrict psi, struct statevector* restrict psi_out);


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
