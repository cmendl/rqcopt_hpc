#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cblas.h>
#include <memory.h>
#include <stdbool.h>
#include "circuit_opt.h"
#include "brickwall_opt.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT

// preliminary version: storing the full matrix and performing a matrix-vector multiplication, for simplicity
static int ufunc(const struct statevector* restrict psi, void* udata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* u = (numeric*)udata;

	// apply U
	numeric alpha = 1;
	numeric beta  = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, u, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
}

#endif


#ifdef COMPLEX_CIRCUIT

static PyObject* optimize_quantum_circuit_py(PyObject* self, PyObject* args)
{
	// suppress "unused parameter" warning
	(void)self;

	// number of qubits
	int nqubits;
	// target unitary
	PyArrayObject* u_obj;
	// initial to-be optimized quantum gates
	PyObject* gates_start_obj;
	// quantum wires the gates act on
	PyObject* wires_obj;
	// number of iterations
	int niter;

	if (!PyArg_ParseTuple(args, "iOOOi", &nqubits, &u_obj, &gates_start_obj, &wires_obj, &niter)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: optimize_quantum_circuit(nqubits, U, gates_start, wires, niter)");
		return NULL;
	}

	if (nqubits <= 0) {
		char msg[1024];
		sprintf(msg, "'nqubits' must be positive, received nqubits = %i", nqubits);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	const intqs n = (intqs)1 << nqubits;

	if (PyArray_NDIM(u_obj) != 2) {
		char msg[1024];
		sprintf(msg, "'U' must have degree 2, received %i", PyArray_NDIM(u_obj));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(u_obj, 0) != n || PyArray_DIM(u_obj, 1) != n) {
		char msg[1024];
		sprintf(msg, "'U' must have dimensions 2^nqubits x 2^nqubits, received a %li x %li matrix and nqubits = %i", PyArray_DIM(u_obj, 0), PyArray_DIM(u_obj, 1), nqubits);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	if (PyArray_TYPE(u_obj) != NPY_CDOUBLE) {
		PyErr_SetString(PyExc_TypeError, "'U' must have 'complex double' entries");
		return NULL;
	}

	if (!(PyArray_FLAGS(u_obj) & NPY_ARRAY_C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_SyntaxError, "'U' does not have contiguous C storage format");
		return NULL;
	}

	numeric* u = PyArray_DATA(u_obj);

	if (!PySequence_Check(gates_start_obj)) {
		PyErr_SetString(PyExc_SyntaxError, "'gates_start' must be a sequence");
		return NULL;
	}

	const int ngates = PySequence_Length(gates_start_obj);
	if (ngates <= 0) {
		PyErr_SetString(PyExc_ValueError, "sequence of initial quantum gates cannot be empty");
		return NULL;
	}

	if (!PySequence_Check(wires_obj)) {
		PyErr_SetString(PyExc_SyntaxError, "'wires' must be a sequence");
		return NULL;
	}

	if (PySequence_Length(wires_obj) != ngates) {
		PyErr_SetString(PyExc_ValueError, "'gates_start' and 'wires' sequences must have the same length");
		return NULL;
	}

	if (niter <= 0) {
		char msg[1024];
		sprintf(msg, "'niter' must be a positive integer, received %i", niter);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	// read initial to-be optimized quantum gates
	struct mat4x4* gates_start = aligned_malloc(ngates * sizeof(struct mat4x4));
	for (int i = 0; i < ngates; i++)
	{
		PyObject* gate_obj = PySequence_GetItem(gates_start_obj, i);
		PyArrayObject* gate_obj_arr = (PyArrayObject*)PyArray_ContiguousFromObject(gate_obj, NPY_CDOUBLE, 2, 2);
		if (gate_obj_arr == NULL)
		{
			char msg[1024];
			sprintf(msg, "cannot interpret gates_start[%i] as complex matrix", i);
			PyErr_SetString(PyExc_SyntaxError, msg);
			Py_DECREF(gate_obj);
			aligned_free(gates_start);
			return NULL;
		}

		if (PyArray_DIM(gate_obj_arr, 0) != 4 || PyArray_DIM(gate_obj_arr, 1) != 4)
		{
			char msg[1024];
			sprintf(msg, "gates_start[%i] must be a 4 x 4 matrix, received a %li x %li matrix", i, PyArray_DIM(gate_obj_arr, 0), PyArray_DIM(gate_obj_arr, 1));
			PyErr_SetString(PyExc_ValueError, msg);
			Py_DECREF(gate_obj_arr);
			Py_DECREF(gate_obj);
			aligned_free(gates_start);
			return NULL;
		}

		memcpy(&gates_start[i], PyArray_DATA(gate_obj_arr), sizeof(struct mat4x4));

		Py_DECREF(gate_obj_arr);
		Py_DECREF(gate_obj);
	}

	// read quantum wire permutations
	int* wires = aligned_malloc(2 * ngates * sizeof(int));
	for (int i = 0; i < ngates; i++)
	{
		PyObject* wire_obj = PySequence_GetItem(wires_obj, i);
		if (!PySequence_Check(wire_obj)) {
			char msg[1024];
			sprintf(msg, "'wires[%i]' must be a sequence", i);
			PyErr_SetString(PyExc_SyntaxError, msg);
			Py_DECREF(wire_obj);
			aligned_free(wires);
			aligned_free(gates_start);
			return NULL;
		}

		if (PySequence_Length(wire_obj) != 2) {
			char msg[1024];
			sprintf(msg, "'wires[%i]' must be a sequence of length 2", i);
			PyErr_SetString(PyExc_ValueError, msg);
			Py_DECREF(wire_obj);
			aligned_free(wires);
			aligned_free(gates_start);
			return NULL;
		}

		for (int j = 0; j < 2; j++)
		{
			PyObject* p_obj = PySequence_GetItem(wire_obj, j);

			wires[2 * i + j] = (int)PyLong_AsLong(p_obj);
			if (PyErr_Occurred()) {
				char msg[1024];
				sprintf(msg, "cannot interpret 'wires[%i][%i]' as integer", i, j);
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(p_obj);
				Py_DECREF(wire_obj);
				aligned_free(wires);
				aligned_free(gates_start);
				return NULL;
			}

			Py_DECREF(p_obj);

			if (wires[2 * i + j] < 0 || wires[2 * i + j] >= nqubits) {
				char msg[1024];
				sprintf(msg, "index 'wires[%i][%i] = %i' outside of [0, nqubits) index range", i, j, wires[2 * i + j]);
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(wire_obj);
				aligned_free(wires);
				aligned_free(gates_start);
				return NULL;
			}
		}

		Py_DECREF(wire_obj);

		if (wires[2 * i] == wires[2 * i + 1]) {
			char msg[1024];
			sprintf(msg, "wire indices for gate %i are not distinct, received index %i twice", i, wires[2 * i]);
			PyErr_SetString(PyExc_ValueError, msg);
			aligned_free(wires);
			aligned_free(gates_start);
			return NULL;
		}
	}

	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(ngates * 16, &params);

	double* f_iter = aligned_malloc((niter + 1) * sizeof(double));
	struct mat4x4* gates_opt = aligned_malloc(ngates * sizeof(struct mat4x4));

	// perform optimization
	optimize_quantum_circuit(ufunc, u, gates_start, ngates, nqubits, wires, &params, niter, f_iter, gates_opt);

	aligned_free(wires);
	aligned_free(gates_start);

	// construct return value
	// f_iter
	npy_intp dims_f_iter[1] = { niter + 1 };
	PyArrayObject* f_iter_obj = (PyArrayObject*)PyArray_SimpleNew(1, dims_f_iter, NPY_DOUBLE);
	if (f_iter_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating to-be-returned 'f_iter' array");
		aligned_free(gates_opt);
		aligned_free(f_iter);
		return NULL;
	}
	memcpy(PyArray_DATA(f_iter_obj), f_iter, (niter + 1) * sizeof(double));
	aligned_free(f_iter);
	// gates_opt
	npy_intp dims_gates[3] = { ngates, 4, 4 };
	PyArrayObject* gates_opt_obj = (PyArrayObject*)PyArray_SimpleNew(3, dims_gates, NPY_CDOUBLE);
	if (gates_opt_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating to-be-returned 'gates_opt' tensor");
		aligned_free(gates_opt);
		return NULL;
	}
	memcpy(PyArray_DATA(gates_opt_obj), gates_opt, ngates * sizeof(struct mat4x4));
	aligned_free(gates_opt);

	return PyTuple_Pack(2, gates_opt_obj, f_iter_obj);
}

#else

static PyObject* optimize_quantum_circuit_py(PyObject* self, PyObject* args)
{
	// suppress "unused parameter" warning
	(void)self;
	(void)args;

	PyErr_SetString(PyExc_RuntimeError, "cannot perform optimization - please re-build with support for complex numbers enabled");

	return NULL;
}

#endif


#ifdef COMPLEX_CIRCUIT

static PyObject* optimize_brickwall_circuit_py(PyObject* self, PyObject* args, PyObject* kwargs)
{
	// suppress "unused parameter" warning
	(void)self;

	// number of qubits
	int L;
	// target unitary
	PyArrayObject* U_obj;
	// initial to-be optimized quantum gates
	PyObject* Vlist_start_obj;
	// permutations of the quantum wires, for each layer
	PyObject* perms_obj;
	// number of iterations
	int niter;
	// number of samples (if using sampling)
	long nsamples = 0;
	// random number generator seed
	uint64_t seed = 42;

	// parse input arguments
	char* kwlist[] = { "", "", "", "", "", "nsamples", "seed", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iOOOi|lk", kwlist,
		&L, &U_obj, &Vlist_start_obj, &perms_obj, &niter, &nsamples, &seed)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: optimize_brickwall_circuit(L, U, Vlist_start, perms, niter, nsamples=0, seed=42)");
		return NULL;
	}

	if (L <= 0 || L % 2 != 0) {
		char msg[1024];
		sprintf(msg, "'L' must be positive and even, received L = %i", L);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	const intqs n = (intqs)1 << L;

	if (PyArray_NDIM(U_obj) != 2) {
		char msg[1024];
		sprintf(msg, "'U' must have degree 2, received %i", PyArray_NDIM(U_obj));
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}
	if (PyArray_DIM(U_obj, 0) != n || PyArray_DIM(U_obj, 1) != n) {
		char msg[1024];
		sprintf(msg, "'U' must have dimensions 2^L x 2^L, received a %li x %li matrix and L = %i", PyArray_DIM(U_obj, 0), PyArray_DIM(U_obj, 1), L);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	if (PyArray_TYPE(U_obj) != NPY_CDOUBLE) {
		PyErr_SetString(PyExc_TypeError, "'U' must have 'complex double' entries");
		return NULL;
	}

	if (!(PyArray_FLAGS(U_obj) & NPY_ARRAY_C_CONTIGUOUS)) {
		PyErr_SetString(PyExc_SyntaxError, "'U' does not have contiguous C storage format");
		return NULL;
	}

	numeric* U = PyArray_DATA(U_obj);

	if (!PySequence_Check(Vlist_start_obj)) {
		PyErr_SetString(PyExc_SyntaxError, "'Vlist_start' must be a sequence");
		return NULL;
	}

	int nlayers = PySequence_Length(Vlist_start_obj);
	if (nlayers <= 0) {
		PyErr_SetString(PyExc_ValueError, "sequence of initial quantum gates cannot be empty");
		return NULL;
	}

	if (!PySequence_Check(perms_obj)) {
		PyErr_SetString(PyExc_SyntaxError, "'perms' must be a sequence");
		return NULL;
	}

	if (PySequence_Length(perms_obj) != nlayers) {
		PyErr_SetString(PyExc_ValueError, "'Vlist_start' and 'perms' sequences must have the same length (equal to the number of layers)");
		return NULL;
	}

	if (niter <= 0) {
		char msg[1024];
		sprintf(msg, "'niter' must be a positive integer, received %i", niter);
		PyErr_SetString(PyExc_ValueError, msg);
		return NULL;
	}

	if (nsamples < 0) {
		PyErr_SetString(PyExc_ValueError, "'nsamples' cannot be negative");
		return NULL;
	}

	// read initial to-be optimized quantum gates
	struct mat4x4* Vlist_start = aligned_malloc(nlayers * sizeof(struct mat4x4));
	for (int i = 0; i < nlayers; i++)
	{
		PyObject* V_obj = PySequence_GetItem(Vlist_start_obj, i);
		PyArrayObject* V_obj_arr = (PyArrayObject*)PyArray_ContiguousFromObject(V_obj, NPY_CDOUBLE, 2, 2);
		if (V_obj_arr == NULL)
		{
			char msg[1024];
			sprintf(msg, "cannot interpret Vlist_start[%i] as complex matrix", i);
			PyErr_SetString(PyExc_SyntaxError, msg);
			Py_DECREF(V_obj);
			aligned_free(Vlist_start);
			return NULL;
		}

		if (PyArray_DIM(V_obj_arr, 0) != 4 || PyArray_DIM(V_obj_arr, 1) != 4)
		{
			char msg[1024];
			sprintf(msg, "Vlist_start[%i] must be a 4 x 4 matrix, received a %li x %li matrix", i, PyArray_DIM(V_obj_arr, 0), PyArray_DIM(V_obj_arr, 1));
			PyErr_SetString(PyExc_ValueError, msg);
			Py_DECREF(V_obj_arr);
			Py_DECREF(V_obj);
			aligned_free(Vlist_start);
			return NULL;
		}

		memcpy(&Vlist_start[i], PyArray_DATA(V_obj_arr), sizeof(struct mat4x4));

		Py_DECREF(V_obj_arr);
		Py_DECREF(V_obj);
	}

	// read quantum wire permutations
	int** perms = aligned_malloc(nlayers * sizeof(int*));
	for (int i = 0; i < nlayers; i++)
	{
		PyObject* perm_obj = PySequence_GetItem(perms_obj, i);
		if (!PySequence_Check(perm_obj)) {
			char msg[1024];
			sprintf(msg, "'perms[%i]' must be a sequence", i);
			PyErr_SetString(PyExc_SyntaxError, msg);
			Py_DECREF(perm_obj);
			for (int k = 0; k < i; k++) {
				aligned_free(perms[k]);
			}
			aligned_free(perms);
			aligned_free(Vlist_start);
			return NULL;
		}

		if (PySequence_Length(perm_obj) != L) {
			char msg[1024];
			sprintf(msg, "'perms[%i]' must be a sequence of length L = %i", i, L);
			PyErr_SetString(PyExc_ValueError, msg);
			Py_DECREF(perm_obj);
			for (int k = 0; k < i; k++) {
				aligned_free(perms[k]);
			}
			aligned_free(perms);
			aligned_free(Vlist_start);
			return NULL;
		}

		perms[i] = aligned_malloc(L * sizeof(int));
		for (int j = 0; j < L; j++)
		{
			PyObject* p_obj = PySequence_GetItem(perm_obj, j);

			perms[i][j] = (int)PyLong_AsLong(p_obj);
			if (PyErr_Occurred()) {
				char msg[1024];
				sprintf(msg, "cannot interpret 'perms[%i][%i]' as integer", i, j);
				PyErr_SetString(PyExc_ValueError, msg);
				Py_DECREF(p_obj);
				Py_DECREF(perm_obj);
				for (int k = 0; k <= i; k++) {
					aligned_free(perms[k]);
				}
				aligned_free(perms);
				aligned_free(Vlist_start);
				return NULL;
			}

			Py_DECREF(p_obj);
		}

		Py_DECREF(perm_obj);

		// ensure that 'perms[i]' is a valid permutation
		int* slots = aligned_malloc(L * sizeof(int));
		memset(slots, 0, L * sizeof(int));
		for (int j = 0; j < L; j++)
		{
			if (0 <= perms[i][j] && perms[i][j] < L) {
				slots[perms[i][j]] = 1;
			}
		}
		bool valid = true;
		for (int j = 0; j < L; j++)
		{
			if (slots[j] == 0) {
				valid = false;
			}
		}
		aligned_free(slots);
		if (!valid) {
			char msg[1024];
			sprintf(msg, "'perms[%i]' is not a valid permutation", i);
			PyErr_SetString(PyExc_ValueError, msg);
			for (int k = 0; k <= i; k++) {
				aligned_free(perms[k]);
			}
			aligned_free(perms);
			aligned_free(Vlist_start);
			return NULL;
		}
	}

	// parameters for optimization
	struct rtr_params params;
	set_rtr_default_params(nlayers * 16, &params);

	double* f_iter = aligned_malloc((niter + 1) * sizeof(double));
	struct mat4x4* Vlist_opt = aligned_malloc(nlayers * sizeof(struct mat4x4));

	// perform optimization
	if (nsamples == 0)
	{
		// without sampling
		optimize_brickwall_circuit_hmat(ufunc, U, Vlist_start, nlayers, L, (const int**)perms, &params, niter, f_iter, Vlist_opt);
	}
	else
	{
		struct rng_state rng;
		seed_rng_state(seed, &rng);
		optimize_brickwall_circuit_hmat_sampling(ufunc, U, Vlist_start, nlayers, L, (const int**)perms, nsamples, &rng, &params, niter, f_iter, Vlist_opt);
	}

	// construct return value
	// f_iter
	npy_intp dims_f_iter[1] = { niter + 1 };
	PyArrayObject* f_iter_obj = (PyArrayObject*)PyArray_SimpleNew(1, dims_f_iter, NPY_DOUBLE);
	if (f_iter_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating to-be-returned 'f_iter' array");
		aligned_free(Vlist_opt);
		aligned_free(f_iter);
		for (int i = 0; i < nlayers; i++) {
			aligned_free(perms[i]);
		}
		aligned_free(perms);
		aligned_free(Vlist_start);
		return NULL;
	}
	memcpy(PyArray_DATA(f_iter_obj), f_iter, (niter + 1) * sizeof(double));
	// Vlist_opt
	npy_intp dims_Vlist[3] = { nlayers, 4, 4 };
	PyArrayObject* Vlist_opt_obj = (PyArrayObject*)PyArray_SimpleNew(3, dims_Vlist, NPY_CDOUBLE);
	if (Vlist_opt_obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError, "error creating to-be-returned 'Vlist_opt' tensor");
		aligned_free(Vlist_opt);
		aligned_free(f_iter);
		for (int i = 0; i < nlayers; i++) {
			aligned_free(perms[i]);
		}
		aligned_free(perms);
		aligned_free(Vlist_start);
		return NULL;
	}
	memcpy(PyArray_DATA(Vlist_opt_obj), Vlist_opt, nlayers * sizeof(struct mat4x4));
	// return value
	PyObject* ret = PyTuple_Pack(2, Vlist_opt_obj, f_iter_obj);

	// clean up
	aligned_free(Vlist_opt);
	aligned_free(f_iter);
	for (int i = 0; i < nlayers; i++) {
		aligned_free(perms[i]);
	}
	aligned_free(perms);
	aligned_free(Vlist_start);

	return ret;
}

#else

static PyObject* optimize_brickwall_circuit_py(PyObject* self, PyObject* args, PyObject* kwargs)
{
	// suppress "unused parameter" warning
	(void)self;
	(void)args;
	(void)kwargs;

	PyErr_SetString(PyExc_RuntimeError, "cannot perform optimization - please re-build with support for complex numbers enabled");

	return NULL;
}

#endif


static PyMethodDef methods[] = {
	{
		.ml_name  = "optimize_quantum_circuit",
		.ml_meth  = optimize_quantum_circuit_py,
		.ml_flags = METH_VARARGS,
		.ml_doc   = "Optimize the two-qubit gates in a quantum circuit to approximate a unitary matrix using a trust-region method."
	},
	{
		.ml_name  = "optimize_brickwall_circuit",
		.ml_meth  = (PyCFunction)optimize_brickwall_circuit_py,
		.ml_flags = METH_VARARGS | METH_KEYWORDS,
		.ml_doc   = "Optimize the quantum gates in a brickwall layout to approximate a unitary matrix using a trust-region method."
	},
	{
		0  // sentinel
	},
};


static struct PyModuleDef module = {
	.m_base     = PyModuleDef_HEAD_INIT,
	.m_name     = "rqcopt_hpc",  // name of module
	.m_doc      = NULL,          // module documentation, may be NULL
	.m_size     = -1,            // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
	.m_methods  = methods,       // module methods
	.m_slots    = NULL,          // slot definitions for multi-phase initialization
	.m_traverse = NULL,          // traversal function to call during GC traversal of the module object, or NULL if not needed
	.m_clear    = NULL,          // a clear function to call during GC clearing of the module object, or NULL if not needed
	.m_free     = NULL,          // a function to call during deallocation of the module object, or NULL if not needed
};


PyMODINIT_FUNC PyInit_rqcopt_hpc(void)
{
	// import NumPy array module (required)
	import_array();

	PyObject* m = PyModule_Create(&module);
	if (m == NULL) {
		return NULL;
	}

	return m;
}
