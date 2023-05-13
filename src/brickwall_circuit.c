#include <assert.h>
#include "brickwall_circuit.h"
#include "parallel_gates.h"


//________________________________________________________________________________________________________________________
///
/// \brief Apply the unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers, const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	// temporary statevector
	struct statevector chi = { 0 };
	if (allocate_statevector(psi->nqubits, &chi) < 0) {
		return -1;
	}

	for (int i = 0; i < nlayers; i++)
	{
		int p = (nlayers - i) % 2;
		const struct statevector* psi0 = (i == 0 ? psi : (p == 0 ? psi_out : &chi));
		struct statevector* psi1 = (p == 0 ? &chi : psi_out);
		int ret = apply_parallel_gates(&Vlist[i], psi0, perms[i], psi1);
		if (ret < 0) {
			free_statevector(&chi);
			return ret;
		}
	}

	free_statevector(&chi);

	return 0;
}


//________________________________________________________________________________________________________________________
///
/// \brief Apply the adjoint unitary matrix representation of a brickwall-type
/// quantum circuit with periodic boundary conditions to state psi.
///
int apply_adjoint_brickwall_unitary(const struct mat4x4 Vlist[], int nlayers, const struct statevector* restrict psi, const int* perms[], struct statevector* restrict psi_out)
{
	assert(nlayers >= 1);
	assert(psi->nqubits == psi_out->nqubits);

	// temporary statevector
	struct statevector chi = { 0 };
	if (allocate_statevector(psi->nqubits, &chi) < 0) {
		return -1;
	}

	for (int i = nlayers - 1; i >= 0; i--)
	{
		int p = i % 2;
		const struct statevector* psi0 = (i == nlayers - 1 ? psi : (p == 0 ? &chi : psi_out));
		struct statevector* psi1 = (p == 0 ? psi_out : &chi);
		struct mat4x4 Vh;
		adjoint(&Vlist[i], &Vh);
		int ret = apply_parallel_gates(&Vh, psi0, perms[i], psi1);
		if (ret < 0) {
			free_statevector(&chi);
			return ret;
		}
	}

	free_statevector(&chi);

	return 0;
}
