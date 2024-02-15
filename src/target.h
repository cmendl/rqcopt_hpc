#pragma once

#include "matrix.h"
#include "statevector.h"


typedef int (*linear_func)(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out);


int circuit_unitary_target(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits,
	double* fval);


int circuit_unitary_target_and_gradient(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits,
	double* fval, struct mat4x4 dgates[]);

int circuit_unitary_target_and_projected_gradient(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const int ngates, const int wires[], const int nqubits,
	double* fval, double* grad_vec);


int circuit_unitary_target_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const struct mat4x4 gatedirs[], const int ngates, const int wires[], const int nqubits,
	double* fval, struct mat4x4 dgates[], struct mat4x4 hess_gatedirs[]);

int circuit_unitary_target_projected_hessian_vector_product(linear_func ufunc, void* udata,
	const struct mat4x4 gates[], const struct mat4x4 gatedirs[], const int ngates, const int wires[], const int nqubits,
	double* fval, double* restrict grad_vec, double* restrict hvp_vec);


//________________________________________________________________________________________________________________________
//


int brickwall_unitary_target(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval);


int brickwall_unitary_target_and_gradient(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[]);

#ifdef COMPLEX_CIRCUIT
int brickwall_unitary_target_and_gradient_vector(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec);
#endif


int brickwall_unitary_target_gradient_hessian(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[], numeric* hess);

#ifdef COMPLEX_CIRCUIT
int brickwall_unitary_target_gradient_vector_hessian_matrix(linear_func ufunc, void* udata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec, double* H);
#endif


//________________________________________________________________________________________________________________________
//


int brickwall_blockenc_target(linear_func hfunc, void* hdata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval);


int brickwall_blockenc_target_and_gradient(linear_func hfunc, void* hdata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[]);

#ifdef COMPLEX_CIRCUIT
int brickwall_blockenc_target_and_gradient_vector(linear_func hfunc, void* hdata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec);
#endif


int brickwall_blockenc_target_gradient_hessian(linear_func hfunc, void* hdata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, struct mat4x4 dVlist[],
	#ifdef COMPLEX_CIRCUIT
	numeric* hess1, numeric* hess2
	#else
	numeric* hess
	#endif
	);

#ifdef COMPLEX_CIRCUIT
int brickwall_blockenc_target_gradient_vector_hessian_matrix(linear_func hfunc, void* hdata,
	const struct mat4x4 Vlist[], const int nlayers, const int L, const int* perms[],
	double* fval, double* grad_vec, double* H);
#endif
