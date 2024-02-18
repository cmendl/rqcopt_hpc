#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "config.h"
#include "circuit_opt.h"


struct f_target_data
{
	linear_func ufunc;
	void* udata;
	const int* wires;
	int ngates;
	int nqubits;
};


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of quantum circuit target function evaluation.
///
static double f(const double* x, void* fdata)
{
	struct f_target_data* data = fdata;

	numeric fval;
	if (circuit_unitary_target(data->ufunc, data->udata, (const struct mat4x4*)x, data->ngates, data->wires, data->nqubits, &fval) < 0) {
		fprintf(stderr, "target function evaluation failed internally\n");
		return -1;
	}

	return creal(fval);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of quantum circuit target function and gradient evaluation.
///
static double f_deriv(const double* restrict x, void* fdata, double* restrict grad)
{
	struct f_target_data* data = fdata;

	numeric fval;
	if (circuit_unitary_target_and_projected_gradient(data->ufunc, data->udata, (const struct mat4x4*)x, data->ngates, data->wires, data->nqubits, &fval, grad) < 0) {
		fprintf(stderr, "target function and derivative evaluation failed internally\n");
		return -1;
	}

	return creal(fval);
}


//________________________________________________________________________________________________________________________
///
/// \brief Wrapper of Hessian-vector product evaluation for quantum circuit target function.
///
static void f_hvp(const double* restrict x, void* fdata, double* restrict vec, double* restrict hvp)
{
	const struct mat4x4* gates = (const struct mat4x4*)x;

	const struct f_target_data* data = fdata;

	// convert input vectors to matrices in tangent space
	struct mat4x4* gatedirs = aligned_alloc(MEM_DATA_ALIGN, data->ngates * sizeof(struct mat4x4));
	for (int i = 0; i < data->ngates; i++)
	{
		real_to_tangent(&vec[i * num_tangent_params], &gates[i], &gatedirs[i]);
	}

	numeric fval;
	double* grad_vec = aligned_alloc(MEM_DATA_ALIGN, data->ngates * num_tangent_params * sizeof(double));

	circuit_unitary_target_projected_hessian_vector_product(data->ufunc, data->udata, gates, gatedirs, data->ngates, data->wires, data->nqubits, &fval, grad_vec, hvp);

	aligned_free(grad_vec);
	aligned_free(gatedirs);
}


//________________________________________________________________________________________________________________________
///
/// \brief Retraction, with tangent direction represented as anti-symmetric matrices.
///
static void retract_unitary_list(const double* restrict x, const double* restrict eta, void* rdata, double* restrict xs)
{
	const int ngates = *((int*)rdata);
	assert(ngates > 0);

	const struct mat4x4* gates  = (const struct mat4x4*)x;
	struct mat4x4* retractgates = (struct mat4x4*)xs;

	for (int i = 0; i < ngates; i++)
	{
		struct mat4x4 z;
		real_to_antisymm(&eta[i * 16], &z);
		// add identity matrix
		z.data[ 0]++;
		z.data[ 5]++;
		z.data[10]++;
		z.data[15]++;

		struct mat4x4 w;
		multiply_matrices(&gates[i], &z, &w);

		polar_factor(&w, &retractgates[i]);
	}

}


//________________________________________________________________________________________________________________________
///
/// \brief Optimize a quantum circuit consisting of two-qubit gates to approximate a unitary matrix `U` using a trust-region method.
///
void optimize_quantum_circuit(linear_func ufunc, void* udata, const struct mat4x4 gates_start[], const int ngates, const int nqubits, const int wires[], struct rtr_params* params, const int niter, double* f_iter, struct mat4x4 gates_opt[])
{
	// target function data
	struct f_target_data fdata = {
		.ufunc   = ufunc,
		.udata   = udata,
		.wires   = wires,
		.ngates  = ngates,
		.nqubits = nqubits,
	};

	// TODO: quantify error by spectral norm
	params->g_func = NULL;
	params->g_data = NULL;
	params->g_iter = NULL;

	// perform optimization
	int rdata = ngates;
	riemannian_trust_region_optimize(f, f_deriv, f_hvp, &fdata, retract_unitary_list, &rdata,
		ngates * num_tangent_params, (const double*)gates_start, ngates * 16 * (sizeof(numeric)/sizeof(double)), params, niter, f_iter, (double*)gates_opt);
}
