#include <memory.h>
#include <cblas.h>
#include <assert.h>
#include "target.h"
#include "numerical_gradient.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT
#define CDATA_LABEL "_cplx"
#else
#define CDATA_LABEL "_real"
#endif


static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	// suppress "unused parameter" warning
	(void)udata;

	assert(psi->nqubits == psi_out->nqubits);

	const intqs n = (intqs)1 << psi->nqubits;
	for (intqs i = 0; i < n; i++)
	{
		#ifndef COMPLEX_CIRCUIT
		psi_out->data[i] = -1.1 * psi->data[((i + 3) * 113) % n] - 0.7 * psi->data[((i + 9) * 173) % n] + 0.5 * psi->data[i] + 0.3 * psi->data[((i + 4) * 199) % n];
		#else
		psi_out->data[i] = (-1.1 + 0.8*I) * psi->data[((i + 3) * 113) % n] + (0.4 - 0.7*I) * psi->data[((i + 9) * 173) % n] + (0.5 + 0.1*I) * psi->data[i] + (-0.3 + 0.2*I) * psi->data[((i + 4) * 199) % n];
		#endif
	}

	return 0;
}


char* test_circuit_unitary_target()
{
	const int nqubits = 7;
	const int ngates  = 5;

	hid_t file = H5Fopen("../test/data/test_circuit_unitary_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_circuit_unitary_target failed";
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

	numeric f;
	if (circuit_unitary_target(ufunc, NULL, gates, ngates, wires, nqubits, &f) < 0) {
		return "'circuit_unitary_target' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	aligned_free(wires);
	aligned_free(gates);

	H5Fclose(file);

	return 0;
}


struct circuit_unitary_target_params
{
	linear_func ufunc;
	const int* wires;
	int nqubits;
	int ngates;
};


// wrapper of circuit unitary target function
static void circuit_unitary_target_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct circuit_unitary_target_params* params = p;

	numeric f;
	circuit_unitary_target(params->ufunc, NULL, (struct mat4x4*)x, params->ngates, params->wires, params->nqubits, &f);
	*y = f;
}


char* test_circuit_unitary_target_and_gradient()
{
	const int nqubits = 8;
	const int ngates  = 5;

	hid_t file = H5Fopen("../test/data/test_circuit_unitary_target_and_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_circuit_unitary_target_and_gradient failed";
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

	numeric f;
	struct mat4x4* dgates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	if (circuit_unitary_target_and_gradient(ufunc, NULL, gates, ngates, wires, nqubits, &f, dgates) < 0) {
		return "'circuit_unitary_target_and_gradient' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	// numerical gradient

	const double h = 1e-5;

	struct circuit_unitary_target_params params = {
		.ufunc   = ufunc,
		.wires   = wires,
		.nqubits = nqubits,
		.ngates  = ngates,
	};

	struct mat4x4* dgates_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));

	numeric dy = 1;
	#ifdef COMPLEX_CIRCUIT
	numerical_gradient_backward_wirtinger(circuit_unitary_target_wrapper, &params, ngates * 16, (numeric*)gates, 1, &dy, h, (numeric*)dgates_num);
	#else
	numerical_gradient_backward(circuit_unitary_target_wrapper, &params, ngates * 16, (numeric*)gates, 1, &dy, h, (numeric*)dgates_num);
	#endif

	// compare
	for (int i = 0; i < ngates; i++) {
		if (uniform_distance(16, dgates[i].data, dgates_num[i].data) > 1e-8) {
			return "target function gradient with respect to gates does not match finite difference approximation";
		}
	}

	aligned_free(dgates_num);
	aligned_free(dgates);
	aligned_free(wires);
	aligned_free(gates);

	H5Fclose(file);

	return 0;
}


struct circuit_unitary_target_directed_gradient_params
{
	linear_func ufunc;
	const struct mat4x4* gatedirs;
	const int* wires;
	int ngates;
	int nqubits;
};

// directed gradient computation with respect to gates for the quantum circuit target function
static void circuit_unitary_target_directed_gradient(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct circuit_unitary_target_directed_gradient_params* params = p;

	numeric f;
	struct mat4x4* dgates = aligned_alloc(MEM_DATA_ALIGN, params->ngates * sizeof(struct mat4x4));

	circuit_unitary_target_and_gradient(params->ufunc, NULL, (struct mat4x4*)x, params->ngates, params->wires, params->nqubits, &f, dgates);

	// compute inner products with gate gradient directions
	numeric dg = 0;
	for (int i = 0; i < params->ngates; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			dg += dgates[i].data[j] * params->gatedirs[i].data[j];
		}
	}
	*y = dg;

	aligned_free(dgates);
}

// wrapper of 'circuit_unitary_target_and_gradient' function
static void circuit_unitary_target_and_gradient_wapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct circuit_unitary_target_params* params = p;

	numeric f;
	circuit_unitary_target_and_gradient(params->ufunc, NULL, (const struct mat4x4*)x, params->ngates, params->wires, params->nqubits, &f, (struct mat4x4*)y);
}

char* test_circuit_unitary_target_hessian_vector_product()
{
	const int nqubits = 7;
	const int ngates  = 6;

	hid_t file = H5Fopen("../test/data/test_circuit_unitary_target_hessian_vector_product" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_circuit_unitary_target_hessian_vector_product failed";
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

	struct mat4x4* gatedirs = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	for (int i = 0; i < ngates; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, gatedirs[i].data) < 0) {
			return "reading derivative directions for two-qubit quantum gates from disk failed";
		}
	}

	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		return "reading wire indices from disk failed";
	}

	// quantum circuit unitary target function, its gate gradients and second derivatives of gates (Hessian-vector product output vector)
	numeric f;
	struct mat4x4* dgates = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	struct mat4x4* hess_gatedirs = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	if (circuit_unitary_target_hessian_vector_product(ufunc, NULL, gates, gatedirs, ngates, wires, nqubits, &f, dgates, hess_gatedirs) < 0) {
		return "'circuit_unitary_target_hessian_vector_product' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	const double h = 1e-5;

	// numerical gradient of target function
	{
		struct circuit_unitary_target_params params = {
			.ufunc   = ufunc,
			.wires   = wires,
			.nqubits = nqubits,
			.ngates  = ngates,
		};

		struct mat4x4* dgates_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));

		numeric dy = 1;

		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(circuit_unitary_target_wrapper, &params, ngates * 16, (numeric*)gates, 1, &dy, h, (numeric*)dgates_num);
		#else
		numerical_gradient_backward(circuit_unitary_target_wrapper, &params, ngates * 16, (numeric*)gates, 1, &dy, h, (numeric*)dgates_num);
		#endif

		// compare
		for (int i = 0; i < ngates; i++) {
			if (uniform_distance(16, dgates[i].data, dgates_num[i].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		aligned_free(dgates_num);
	}

	// numerical gradient of the directed gradient
	{
		struct circuit_unitary_target_directed_gradient_params params = {
			.ufunc    = ufunc,
			.wires    = wires,
			.gatedirs = gatedirs,
			.ngates   = ngates,
			.nqubits  = nqubits,
		};

		struct mat4x4* hess_gatedirs_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));

		numeric dy = 1;

		numerical_gradient_backward(circuit_unitary_target_directed_gradient, &params, ngates * 16, (numeric*)gates, 1, &dy, h, (numeric*)hess_gatedirs_num);

		// compare
		if (uniform_distance(ngates * 16, (numeric*)hess_gatedirs, (numeric*)hess_gatedirs_num) > 1e-8) {
			return "Hessian-vector product with respect to gates computed by 'circuit_unitary_target_hessian_vector_product' does not match finite difference approximation";
		}

		aligned_free(hess_gatedirs_num);
	}

	// alternative calculation of the Hessian-vector product using forward-mode differentiation
	{
		struct circuit_unitary_target_params params = {
			.ufunc   = ufunc,
			.wires   = wires,
			.ngates  = ngates,
			.nqubits = nqubits,
		};

		struct mat4x4* hess_gatedirs_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));

		numerical_gradient_forward(circuit_unitary_target_and_gradient_wapper, &params, ngates * 16, (numeric*)gates, (numeric*)gatedirs, ngates * 16, h, (numeric*)hess_gatedirs_num);

		// compare
		if (uniform_distance(ngates * 16, (numeric*)hess_gatedirs, (numeric*)hess_gatedirs_num) > 1e-8) {
			return "Hessian-vector product with respect to gates computed by 'circuit_unitary_target_hessian_vector_product' does not match finite difference approximation";
		}

		aligned_free(hess_gatedirs_num);
	}

	aligned_free(hess_gatedirs);
	aligned_free(dgates);
	aligned_free(wires);
	aligned_free(gatedirs);
	aligned_free(gates);

	H5Fclose(file);

	return 0;
}


// projected gradient computation with respect to gates for the quantum circuit target function
static void circuit_unitary_target_and_projected_gradient_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct circuit_unitary_target_params* params = p;

	const struct mat4x4* gates = (const struct mat4x4*)x;
	struct mat4x4* dgates_proj = (struct mat4x4*)y;

	double* grad_vec = aligned_alloc(MEM_DATA_ALIGN, params->ngates * num_tangent_params * sizeof(double));

	numeric f;
	circuit_unitary_target_and_projected_gradient(ufunc, NULL, gates, params->ngates, params->wires, params->nqubits, &f, grad_vec);

	// convert gradient vectors back to matrices in tangent plane
	for (int i = 0; i < params->ngates; i++)
	{
		real_to_tangent(&grad_vec[i * num_tangent_params], &gates[i], &dgates_proj[i]);
	}

	aligned_free(grad_vec);
}

char* test_circuit_unitary_target_projected_hessian_vector_product()
{
	const int nqubits = 7;
	const int ngates  = 6;

	hid_t file = H5Fopen("../test/data/test_circuit_unitary_target_projected_hessian_vector_product" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_circuit_unitary_target_projected_hessian_vector_product failed";
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

	struct mat4x4* gatedirs = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	for (int i = 0; i < ngates; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, gatedirs[i].data) < 0) {
			return "reading derivative directions for two-qubit quantum gates from disk failed";
		}
	}

	int* wires = aligned_alloc(MEM_DATA_ALIGN, 2 * ngates * sizeof(int));
	if (read_hdf5_dataset(file, "wires", H5T_NATIVE_INT, wires) < 0) {
		return "reading wire indices from disk failed";
	}

	// quantum circuit unitary target function, its projected gate gradients and second derivatives of gates (Hessian-vector product output vector)
	numeric f;
	double* grad_vec = aligned_alloc(MEM_DATA_ALIGN, ngates * num_tangent_params * sizeof(double));
	double* hvp_vec  = aligned_alloc(MEM_DATA_ALIGN, ngates * num_tangent_params * sizeof(double));
	if (circuit_unitary_target_projected_hessian_vector_product(ufunc, NULL, gates, gatedirs, ngates, wires, nqubits, &f, grad_vec, hvp_vec) < 0) {
		return "'circuit_unitary_target_projected_hessian_vector_product' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}
	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	// numerical (forward-mode) gradient of the projected gradient

	const double h = 1e-5;

	struct circuit_unitary_target_params params = {
		.ufunc   = ufunc,
		.wires   = wires,
		.ngates  = ngates,
		.nqubits = nqubits,
	};
	struct mat4x4* hess_gatedirs_proj_num = aligned_alloc(MEM_DATA_ALIGN, ngates * sizeof(struct mat4x4));
	numerical_gradient_forward(circuit_unitary_target_and_projected_gradient_wrapper, &params, ngates * 16, (numeric*)gates, (numeric*)gatedirs, ngates * 16/*ngates * num_tangent_params * sizeof(double) / sizeof(numeric)*/, h, (numeric*)hess_gatedirs_proj_num);

	// numerical gradient will (in general) not be within tangent plane, so need to project again
	double* hvp_vec_num = aligned_alloc(MEM_DATA_ALIGN, ngates * num_tangent_params * sizeof(double));
	for (int i = 0; i < ngates; i++)
	{
		// tangent_to_real implicitly also projects into tangent plane
		tangent_to_real(&gates[i], &hess_gatedirs_proj_num[i], &hvp_vec_num[i * num_tangent_params]);
	}

	// compare
	if (uniform_distance_real(ngates * num_tangent_params, hvp_vec, hvp_vec_num) > 1e-8) {
		return "Hessian-vector product with respect to gates computed by 'circuit_unitary_target_projected_hessian_vector_product' does not match finite difference approximation";
	}

	aligned_free(hess_gatedirs_proj_num);
	aligned_free(hvp_vec_num);
	aligned_free(hvp_vec);
	aligned_free(grad_vec);
	aligned_free(wires);
	aligned_free(gatedirs);
	aligned_free(gates);

	H5Fclose(file);

	return 0;
}


//________________________________________________________________________________________________________________________
//


char* test_brickwall_unitary_target()
{
	const int nqubits = 8;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		numeric f;
		if (brickwall_unitary_target(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f) < 0) {
			return "'brickwall_unitary_target' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}
	}

	H5Fclose(file);

	return 0;
}


struct brickwall_unitary_target_params
{
	linear_func ufunc;
	const int** perms;
	int nqubits;
	int nlayers;
};

// wrapper of brickwall unitary target function
static void brickwall_unitary_target_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	struct mat4x4* vlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < params->nlayers; i++) {
		memcpy(vlist[i].data, &x[i * 16], sizeof(vlist[i].data));
	}

	numeric f;
	brickwall_unitary_target(params->ufunc, NULL, vlist, params->nlayers, params->nqubits, params->perms, &f);
	*y = f;

	aligned_free(vlist);
}


char* test_brickwall_unitary_target_and_gradient()
{
	const int nqubits = 8;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_and_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_and_gradient failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		numeric f;
		struct mat4x4 dvlist[5];
		if (brickwall_unitary_target_and_gradient(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist) < 0) {
			return "'brickwall_unitary_target_and_gradient' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		// numerical gradient
		const double h = 1e-5;
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};
		struct mat4x4 dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(brickwall_unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#else
		numerical_gradient_backward(brickwall_unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_num[j].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dvlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dvlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_ref[j].data) > 1e-12) {
				return "computed target function gradient does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_target_and_projected_gradient()
{
	const int nqubits = 6;
	const int nlayers = 3;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_and_projected_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_and_projected_gradient failed";
	}

	struct mat4x4 vlist[3];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[3][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };

	const int m = nlayers * 16;

	numeric f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (brickwall_unitary_target_and_projected_gradient(ufunc, NULL, vlist, nlayers, nqubits, pperms, &f, grad) < 0) {
		return "'brickwall_unitary_target_and_gradient_vector' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m, grad, grad_ref) > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	aligned_free(grad_ref);
	aligned_free(grad);
	H5Fclose(file);

	return 0;
}

#endif


struct brickwall_unitary_target_directed_gradient_params
{
	linear_func ufunc;
	const struct mat4x4* vdirs;
	const int** perms;
	int nlayers;
	int nqubits;
};

// directed gradient computation with respect to gates for the brickwall circuit target function
static void brickwall_unitary_target_directed_gradient(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_directed_gradient_params* params = p;

	numeric f;
	struct mat4x4* dvlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct mat4x4));

	brickwall_unitary_target_and_gradient(params->ufunc, NULL, (struct mat4x4*)x, params->nlayers, params->nqubits, params->perms, &f, dvlist);

	// compute inner products with gate gradient directions
	numeric dg = 0;
	for (int i = 0; i < params->nlayers; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			dg += dvlist[i].data[j] * params->vdirs[i].data[j];
		}
	}
	*y = dg;

	aligned_free(dvlist);
}

// wrapper of 'brickwall_unitary_target_and_gradient' function
static void brickwall_unitary_target_and_gradient_wapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	numeric f;
	brickwall_unitary_target_and_gradient(params->ufunc, NULL, (const struct mat4x4*)x, params->nlayers, params->nqubits, params->perms, &f, (struct mat4x4*)y);
}

char* test_brickwall_unitary_target_hessian_vector_product()
{
	const int nqubits = 6;
	const int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_hessian_vector_product" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_hessian_vector_product failed";
	}

	struct mat4x4* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	struct mat4x4* vdirs = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vdirs[i].data) < 0) {
			return "reading derivative directions for two-qubit quantum gates from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	// brickwall circuit unitary target function, its gate gradients and second derivatives of gates (Hessian-vector product output vector)
	numeric f;
	struct mat4x4* dvlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	struct mat4x4* hess_vdirs = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	if (brickwall_unitary_target_hessian_vector_product(ufunc, NULL, vlist, vdirs, nlayers, pperms, nqubits, &f, dvlist, hess_vdirs) < 0) {
		return "'brickwall_unitary_target_hessian_vector_product' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	const double h = 1e-5;

	// numerical gradient of target function
	{
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.perms   = pperms,
			.nqubits = nqubits,
			.nlayers = nlayers,
		};

		struct mat4x4* dvlist_num = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

		numeric dy = 1;

		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(brickwall_unitary_target_wrapper, &params, nlayers * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#else
		numerical_gradient_backward(brickwall_unitary_target_wrapper, &params, nlayers * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif

		// compare
		for (int i = 0; i < nlayers; i++) {
			if (uniform_distance(16, dvlist[i].data, dvlist_num[i].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		aligned_free(dvlist_num);
	}

	// numerical gradient of the directed gradient
	{
		struct brickwall_unitary_target_directed_gradient_params params = {
			.ufunc   = ufunc,
			.perms   = pperms,
			.vdirs   = vdirs,
			.nlayers = nlayers,
			.nqubits = nqubits,
		};

		struct mat4x4* hess_vdirs_num = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

		numeric dy = 1;

		numerical_gradient_backward(brickwall_unitary_target_directed_gradient, &params, nlayers * 16, (numeric*)vlist, 1, &dy, h, (numeric*)hess_vdirs_num);

		// compare
		if (uniform_distance(nlayers * 16, (numeric*)hess_vdirs, (numeric*)hess_vdirs_num) > 1e-7) {
			return "Hessian-vector product with respect to gates computed by 'brickwall_unitary_target_hessian_vector_product' does not match finite difference approximation";
		}

		aligned_free(hess_vdirs_num);
	}

	// alternative calculation of the Hessian-vector product using forward-mode differentiation
	{
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.perms   = pperms,
			.nlayers = nlayers,
			.nqubits = nqubits,
		};

		struct mat4x4* hess_vdirs_num = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

		numerical_gradient_forward(brickwall_unitary_target_and_gradient_wapper, &params, nlayers * 16, (numeric*)vlist, (numeric*)vdirs, nlayers * 16, h, (numeric*)hess_vdirs_num);

		// compare
		if (uniform_distance(nlayers * 16, (numeric*)hess_vdirs, (numeric*)hess_vdirs_num) > 1e-7) {
			return "Hessian-vector product with respect to gates computed by 'brickwall_unitary_target_hessian_vector_product' does not match finite difference approximation";
		}

		aligned_free(hess_vdirs_num);
	}

	aligned_free(hess_vdirs);
	aligned_free(dvlist);
	aligned_free(vdirs);
	aligned_free(vlist);

	H5Fclose(file);

	return 0;
}


// projected gradient computation with respect to gates for the brickwall circuit target function
static void brickwall_unitary_target_and_projected_gradient_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	const struct mat4x4* vlist = (const struct mat4x4*)x;
	struct mat4x4* dvlist_proj = (struct mat4x4*)y;

	double* grad_vec = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * num_tangent_params * sizeof(double));

	numeric f;
	brickwall_unitary_target_and_projected_gradient(ufunc, NULL, vlist, params->nlayers, params->nqubits, params->perms, &f, grad_vec);

	// convert gradient vectors back to matrices in tangent plane
	for (int i = 0; i < params->nlayers; i++)
	{
		real_to_tangent(&grad_vec[i * num_tangent_params], &vlist[i], &dvlist_proj[i]);
	}

	aligned_free(grad_vec);
}

char* test_brickwall_unitary_target_projected_hessian_vector_product()
{
	const int nqubits = 6;
	const int nlayers = 4;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_projected_hessian_vector_product" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_projected_hessian_vector_product failed";
	}

	struct mat4x4* vlist = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	struct mat4x4* vdirs = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vdirs[i].data) < 0) {
			return "reading derivative directions for two-qubit quantum gates from disk failed";
		}
	}

	int perms[4][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3] };

	// brickwall circuit unitary target function, its projected gate gradients and second derivatives of gates (Hessian-vector product output vector)
	numeric f;
	double* grad_vec = aligned_alloc(MEM_DATA_ALIGN, nlayers * num_tangent_params * sizeof(double));
	double* hvp_vec  = aligned_alloc(MEM_DATA_ALIGN, nlayers * num_tangent_params * sizeof(double));
	if (brickwall_unitary_target_projected_hessian_vector_product(ufunc, NULL, vlist, vdirs, nlayers, pperms, nqubits, &f, grad_vec, hvp_vec) < 0) {
		return "'brickwall_unitary_target_projected_hessian_vector_product' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}
	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	// numerical (forward-mode) gradient of the projected gradient

	const double h = 1e-5;

	struct brickwall_unitary_target_params params = {
		.ufunc   = ufunc,
		.perms   = pperms,
		.nlayers = nlayers,
		.nqubits = nqubits,
	};
	struct mat4x4* hess_vdirs_proj_num = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
	numerical_gradient_forward(brickwall_unitary_target_and_projected_gradient_wrapper, &params, nlayers * 16, (numeric*)vlist, (numeric*)vdirs, nlayers * 16/*nlayers * num_tangent_params * sizeof(double) / sizeof(numeric)*/, h, (numeric*)hess_vdirs_proj_num);

	// numerical gradient will (in general) not be within tangent plane, so need to project again
	double* hvp_vec_num = aligned_alloc(MEM_DATA_ALIGN, nlayers * num_tangent_params * sizeof(double));
	for (int i = 0; i < nlayers; i++)
	{
		// tangent_to_real implicitly also projects into tangent plane
		tangent_to_real(&vlist[i], &hess_vdirs_proj_num[i], &hvp_vec_num[i * num_tangent_params]);
	}

	// compare
	if (uniform_distance_real(nlayers * num_tangent_params, hvp_vec, hvp_vec_num) > 1e-8) {
		return "Hessian-vector product with respect to gates computed by 'brickwall_unitary_target_projected_hessian_vector_product' does not match finite difference approximation";
	}

	aligned_free(hess_vdirs_proj_num);
	aligned_free(hvp_vec_num);
	aligned_free(hvp_vec);
	aligned_free(grad_vec);
	aligned_free(vdirs);
	aligned_free(vlist);

	H5Fclose(file);

	return 0;
}


// wrapper of brickwall unitary target gradient function
static void brickwall_unitary_target_gradient_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_unitary_target_params* params = p;

	numeric f;
	brickwall_unitary_target_and_gradient(params->ufunc, NULL, (struct mat4x4*)x, params->nlayers, params->nqubits, params->perms, &f, (struct mat4x4*)y);
}

char* test_brickwall_unitary_target_gradient_hessian()
{
	const int nqubits = 6;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_hessian failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	// gradient direction of quantum gates
	struct mat4x4 Zlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Zlist[i].data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
	}

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		const int m = nlayers[i] * 16;

		numeric f;
		struct mat4x4 dvlist[5];
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (brickwall_unitary_target_gradient_hessian(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist, hess) < 0) {
			return "'brickwall_unitary_target_gradient_hessian' failed internally";
		}

		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess[j*m + k] - hess[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		numeric f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}
		// compare with reference
		if (_abs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		const double h = 1e-5;
		struct brickwall_unitary_target_params params = {
			.ufunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};

		// numerical gradient
		struct mat4x4 dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(brickwall_unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#else
		numerical_gradient_backward(brickwall_unitary_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_num[j].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dvlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dvlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_ref[j].data) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}

		// numerical gradient
		struct mat4x4 dVZlist_num[5];
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, nlayers[i] * 16, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#else
		numerical_gradient_backward(brickwall_unitary_target_gradient_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, nlayers[i] * 16, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#endif
		// Hessian matrix times gradient direction
		struct mat4x4 dVZlist[5];
		#ifdef COMPLEX_CIRCUIT
		numeric alpha = 1;
		numeric beta  = 0;
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);
		#else
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, hess, m, (numeric*)Zlist, 1, 0, (numeric*)dVZlist, 1);
		#endif
		// compare
		if (uniform_distance(m, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-7) {
			return "second derivative with respect to gates computed by 'unitary_target_gradient_hessian' does not match finite difference approximation";
		}

		sprintf(varname, "hess%i", i);
		numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess_ref) < 0) {
			return "reading reference Hessian matrix from disk failed";
		}

		// compare with reference
		if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
			return "computed brickwall circuit Hessian matrix does not match reference";
		}

		aligned_free(hess_ref);
		aligned_free(hess);
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_unitary_target_gradient_vector_hessian_matrix()
{
	const int nqubits = 6;
	const int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_brickwall_unitary_target_gradient_vector_hessian_matrix" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_unitary_target_gradient_vector_hessian_matrix failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	const int m = nlayers * 16;

	numeric f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	double* H = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));

	if (brickwall_unitary_target_gradient_vector_hessian_matrix(ufunc, NULL, vlist, nlayers, nqubits, pperms, &f, grad, H) < 0) {
		return "'brickwall_unitary_target_gradient_vector_hessian_matrix' failed internally";
	}

	numeric f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (_abs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m, grad, grad_ref) > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	// check symmetry
	double es = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			es = fmax(es, fabs(H[i * m + j] - H[j * m + i]));
		}
	}
	if (es > 1e-14) {
		return "computed gate Hessian matrix is not symmetric";
	}

	double* H_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H_ref == NULL) {
		return "memory allocation for reference Hessian matrix failed";
	}
	if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H_ref) < 0) {
		return "reading reference Hessian matrix from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m * m, H, H_ref) > 1e-13) {
		return "computed unitary target Hessian matrix does not match reference";
	}

	aligned_free(H_ref);
	aligned_free(grad_ref);
	aligned_free(H);
	aligned_free(grad);

	H5Fclose(file);

	return 0;
}

#endif


//________________________________________________________________________________________________________________________
//


char* test_brickwall_blockenc_target()
{
	const int nqubits = 8;

	hid_t file = H5Fopen("../test/data/test_brickwall_blockenc_target" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_blockenc_target failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		double f;
		if (brickwall_blockenc_target(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f) < 0) {
			return "'brickwall_blockenc_target' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}
	}

	H5Fclose(file);

	return 0;
}


struct brickwall_blockenc_target_params
{
	linear_func hfunc;
	int nqubits;
	int nlayers;
	const int** perms;
};

// wrapper of block encoding target function
static void blockenc_target_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_blockenc_target_params* params = p;

	struct mat4x4* vlist = aligned_alloc(MEM_DATA_ALIGN, params->nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < params->nlayers; i++) {
		memcpy(vlist[i].data, &x[i * 16], sizeof(vlist[i].data));
	}

	double f;
	brickwall_blockenc_target(params->hfunc, NULL, vlist, params->nlayers, params->nqubits, params->perms, &f);
	*y = f;

	aligned_free(vlist);
}


char* test_brickwall_blockenc_target_and_gradient()
{
	const int nqubits = 8;

	hid_t file = H5Fopen("../test/data/test_brickwall_blockenc_target_and_gradient" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_blockenc_target_and_gradient failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		double f;
		struct mat4x4 dvlist[5];
		if (brickwall_blockenc_target_and_gradient(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist) < 0) {
			return "'brickwall_blockenc_target_and_gradient' failed internally";
		}

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}

		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		// numerical gradient
		const double h = 1e-5;
		struct brickwall_blockenc_target_params params = {
			.hfunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};
		struct mat4x4 dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(blockenc_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		// convert from Wirtinger convention
		for (int j = 0; j < nlayers[i]; j++) {
			for (int k = 0; k < 16; k++) {
				dvlist_num[j].data[k] = 2 * dvlist_num[j].data[k];
			}
		}
		#else
		numerical_gradient_backward(blockenc_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_num[j].data) > 1e-8) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dvlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dvlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_ref[j].data) > 1e-12) {
				return "computed target function gradient does not match reference";
			}
		}
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_blockenc_target_and_gradient_vector()
{
	const int nqubits = 6;
	const int nlayers = 3;

	hid_t file = H5Fopen("../test/data/test_brickwall_blockenc_target_and_gradient_vector" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_blockenc_target_and_gradient_vector failed";
	}

	struct mat4x4 vlist[3];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[3][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2] };

	const int m = nlayers * 16;

	double f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (brickwall_blockenc_target_and_gradient_vector(ufunc, NULL, vlist, nlayers, nqubits, pperms, &f, grad) < 0) {
		return "'brickwall_blockenc_target_and_gradient_vector' failed internally";
	}

	double f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (fabs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m, grad, grad_ref) > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	aligned_free(grad_ref);
	aligned_free(grad);
	H5Fclose(file);

	return 0;
}

#endif


// wrapper of block encoding target gradient function
static void brickwall_blockenc_target_gradient_wrapper(const numeric* restrict x, void* p, numeric* restrict y)
{
	const struct brickwall_blockenc_target_params* params = p;

	double f;
	brickwall_blockenc_target_and_gradient(params->hfunc, NULL, (struct mat4x4*)x, params->nlayers, params->nqubits, params->perms, &f, (struct mat4x4*)y);
}

char* test_brickwall_blockenc_target_gradient_hessian()
{
	const int nqubits = 6;

	hid_t file = H5Fopen("../test/data/test_brickwall_blockenc_target_gradient_hessian" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_blockenc_target_gradient_hessian failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][8];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	// gradient direction of quantum gates
	struct mat4x4 Zlist[5];
	for (int i = 0; i < 5; i++)
	{
		char varname[32];
		sprintf(varname, "Z%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, Zlist[i].data) < 0) {
			return "reading gradient direction of two-qubit quantum gates from disk failed";
		}
	}

	int nlayers[] = { 4, 5 };
	for (int i = 0; i < 2; i++)
	{
		const int m = nlayers[i] * 16;

		double f;
		struct mat4x4 dvlist[5];
		#ifdef COMPLEX_CIRCUIT
		numeric* hess1 = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		numeric* hess2 = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (brickwall_blockenc_target_gradient_hessian(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist, hess1, hess2) < 0) {
			return "'brickwall_blockenc_target_gradient_hessian' failed internally";
		}
		#else
		numeric* hess = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (brickwall_blockenc_target_gradient_hessian(ufunc, NULL, vlist, nlayers[i], nqubits, pperms, &f, dvlist, hess) < 0) {
			return "'brickwall_blockenc_target_gradient_hessian' failed internally";
		}
		#endif

		#ifdef COMPLEX_CIRCUIT
		// check symmetry of first Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess1[j*m + k] - hess1[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}
		// check symmetry of second Hessian matrix
		err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess2[j*m + k] - conj(hess2[k*m + j])));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}
		#else
		// check symmetry of Hessian matrix
		double err_symm = 0;
		for (int j = 0; j < m; j++) {
			for (int k = 0; k < m; k++) {
				err_symm = fmax(err_symm, _abs(hess[j*m + k] - hess[k*m + j]));
			}
		}
		if (err_symm > 1e-12) {
			return "Hessian matrix is not symmetric";
		}
		#endif

		char varname[32];
		sprintf(varname, "f%i", i);
		double f_ref;
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, &f_ref) < 0) {
			return "reading reference target function value from disk failed";
		}
		// compare with reference
		if (fabs(f - f_ref) > 1e-12) {
			return "computed target function value does not match reference";
		}

		const double h = 1e-5;
		struct brickwall_blockenc_target_params params = {
			.hfunc   = ufunc,
			.nqubits = nqubits,
			.nlayers = nlayers[i],
			.perms   = pperms,
		};

		// numerical gradient
		struct mat4x4 dvlist_num[5];
		numeric dy = 1;
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(blockenc_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		// convert from Wirtinger convention
		for (int j = 0; j < nlayers[i]; j++) {
			for (int k = 0; k < 16; k++) {
				dvlist_num[j].data[k] = 2 * dvlist_num[j].data[k];
			}
		}
		#else
		numerical_gradient_backward(blockenc_target_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, 1, &dy, h, (numeric*)dvlist_num);
		#endif
		// compare
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_num[j].data) > 1e-7) {
				return "target function gradient with respect to gates does not match finite difference approximation";
			}
		}

		sprintf(varname, "dVlist%i", i);
		struct mat4x4 dvlist_ref[5];
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, dvlist_ref) < 0) {
			return "reading reference gradient data from disk failed";
		}

		// compare with reference
		for (int j = 0; j < nlayers[i]; j++) {
			if (uniform_distance(16, dvlist[j].data, dvlist_ref[j].data) > 1e-12) {
				return "computed brickwall circuit gradient does not match reference";
			}
		}

		// numerical gradient
		struct mat4x4 dVZlist_num[5];
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_wirtinger(brickwall_blockenc_target_gradient_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, nlayers[i] * 16, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#else
		numerical_gradient_backward(brickwall_blockenc_target_gradient_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, nlayers[i] * 16, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		#endif
		// Hessian matrix times gradient direction
		struct mat4x4 dVZlist[5];
		#ifdef COMPLEX_CIRCUIT
		numeric alpha = 1;
		numeric beta  = 0;
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess1, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);
		#else
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, hess, m, (numeric*)Zlist, 1, 0, (numeric*)dVZlist, 1);
		#endif
		// compare
		if (uniform_distance(m, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-6) {
			return "second derivative with respect to gates computed by 'blockenc_target_gradient_hessian' does not match finite difference approximation";
		}
		#ifdef COMPLEX_CIRCUIT
		numerical_gradient_backward_conjugated_wirtinger(brickwall_blockenc_target_gradient_wrapper, &params, nlayers[i] * 16, (numeric*)vlist, nlayers[i] * 16, (numeric*)Zlist, h, (numeric*)dVZlist_num);
		// Hessian matrix times gradient direction
		cblas_zgemv(CblasRowMajor, CblasNoTrans, m, m, &alpha, hess2, m, (numeric*)Zlist, 1, &beta, (numeric*)dVZlist, 1);
		// compare
		if (uniform_distance(m, (numeric*)dVZlist, (numeric*)dVZlist_num) > 1e-6) {
			return "second conjugated Wirtinger derivative with respect to gates computed by 'blockenc_target_gradient_hessian' does not match finite difference approximation";
		}
		#endif

		#ifdef COMPLEX_CIRCUIT
		sprintf(varname, "hess1_%i", i);
		numeric* hess1_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess1_ref) < 0) {
			return "reading reference Hessian matrix from disk failed";
		}
		sprintf(varname, "hess2_%i", i);
		numeric* hess2_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess2_ref) < 0) {
			return "reading reference Hessian matrix from disk failed";
		}

		// compare with reference
		if (uniform_distance(m * m, hess1, hess1_ref) > 1e-12) {
			return "computed brickwall circuit Hessian matrix does not match reference";
		}
		if (uniform_distance(m * m, hess2, hess2_ref) > 1e-12) {
			return "computed brickwall circuit Hessian matrix does not match reference";
		}

		aligned_free(hess2_ref);
		aligned_free(hess1_ref);
		#else
		sprintf(varname, "hess%i", i);
		numeric* hess_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(numeric));
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, hess_ref) < 0) {
			return "reading reference Hessian matrix from disk failed";
		}

		// compare with reference
		if (uniform_distance(m * m, hess, hess_ref) > 1e-12) {
			return "computed brickwall circuit Hessian matrix does not match reference";
		}
		aligned_free(hess_ref);
		#endif

		#ifdef COMPLEX_CIRCUIT
		aligned_free(hess2);
		aligned_free(hess1);
		#else
		aligned_free(hess);
		#endif
	}

	H5Fclose(file);

	return 0;
}


#ifdef COMPLEX_CIRCUIT

char* test_brickwall_blockenc_target_gradient_vector_hessian_matrix()
{
	const int nqubits = 6;
	const int nlayers = 5;

	hid_t file = H5Fopen("../test/data/test_brickwall_blockenc_target_gradient_vector_hessian_matrix" CDATA_LABEL ".hdf5", H5F_ACC_RDONLY, H5P_DEFAULT);
	if (file < 0) {
		return "'H5Fopen' in test_brickwall_blockenc_target_gradient_vector_hessian_matrix failed";
	}

	struct mat4x4 vlist[5];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "V%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_DOUBLE, vlist[i].data) < 0) {
			return "reading two-qubit quantum gate entries from disk failed";
		}
	}

	int perms[5][6];
	for (int i = 0; i < nlayers; i++)
	{
		char varname[32];
		sprintf(varname, "perm%i", i);
		if (read_hdf5_dataset(file, varname, H5T_NATIVE_INT, perms[i]) < 0) {
			return "reading permutation data from disk failed";
		}
	}
	const int* pperms[] = { perms[0], perms[1], perms[2], perms[3], perms[4] };

	const int m = nlayers * 16;

	double* zvec = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "zvec", H5T_NATIVE_DOUBLE, zvec) < 0) {
		return "reading gradient direction of gradient vector from disk failed";
	}

	double f;
	double* grad = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	double* H = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));

	if (brickwall_blockenc_target_gradient_vector_hessian_matrix(ufunc, NULL, vlist, nlayers, nqubits, pperms, &f, grad, H) < 0) {
		return "'brickwall_blockenc_target_gradient_vector_hessian_matrix' failed internally";
	}

	double f_ref;
	if (read_hdf5_dataset(file, "f", H5T_NATIVE_DOUBLE, &f_ref) < 0) {
		return "reading reference target function value from disk failed";
	}

	// compare with reference
	if (fabs(f - f_ref) > 1e-12) {
		return "computed target function value does not match reference";
	}

	double* grad_ref = aligned_alloc(MEM_DATA_ALIGN, m * sizeof(double));
	if (read_hdf5_dataset(file, "grad", H5T_NATIVE_DOUBLE, grad_ref) < 0) {
		return "reading reference gradient vector from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m, grad, grad_ref) > 1e-14) {
		return "computed gate gradient vector does not match reference";
	}

	// check symmetry
	double es = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			es = fmax(es, fabs(H[i * m + j] - H[j * m + i]));
		}
	}
	if (es > 1e-14) {
		return "computed gate Hessian matrix is not symmetric";
	}

	double* H_ref = aligned_alloc(MEM_DATA_ALIGN, m * m * sizeof(double));
	if (H_ref == NULL) {
		return "memory allocation for reference Hessian matrix failed";
	}
	if (read_hdf5_dataset(file, "H", H5T_NATIVE_DOUBLE, H_ref) < 0) {
		return "reading reference Hessian matrix from disk failed";
	}

	// compare with reference
	if (uniform_distance_real(m * m, H, H_ref) > 1e-13) {
		return "computed block encoding target Hessian matrix does not match reference";
	}

	aligned_free(H_ref);
	aligned_free(grad_ref);
	aligned_free(H);
	aligned_free(grad);
	aligned_free(zvec);

	H5Fclose(file);

	return 0;
}

#endif
