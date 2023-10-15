#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <cblas.h>
#include <memory.h>
#include <stdbool.h>
#include "brickwall_opt.h"
#include "util.h"


#ifdef COMPLEX_CIRCUIT

static int ufunc(const struct statevector* restrict psi, void* fdata, struct statevector* restrict psi_out)
{
	const intqs n = (intqs)1 << psi->nqubits;
	const numeric* U = (numeric*)fdata;

	// apply U
	numeric alpha = 1;
	numeric beta  = 0;
	cblas_zgemv(CblasRowMajor, CblasNoTrans, n, n, &alpha, U, n, psi->data, 1, &beta, psi_out->data, 1);

	return 0;
}

static PyObject* optimize_brickwall_circuit_py(PyObject* self, PyObject* args)
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

	if (!PyArg_ParseTuple(args, "iOOOi", &L, &U_obj, &Vlist_start_obj, &perms_obj, &niter)) {
		PyErr_SetString(PyExc_SyntaxError, "error parsing input; syntax: optimize_brickwall_circuit(L, U, Vlist_start, perms, niter)");
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

	// read initial to-be optimized quantum gates
	struct mat4x4* Vlist_start = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));
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
	int** perms = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(int*));
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

		perms[i] = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
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
		int* slots = aligned_alloc(MEM_DATA_ALIGN, L * sizeof(int));
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

	double* f_iter = aligned_alloc(MEM_DATA_ALIGN, niter * sizeof(double));
	struct mat4x4* Vlist_opt = aligned_alloc(MEM_DATA_ALIGN, nlayers * sizeof(struct mat4x4));

	// perform optimization
	optimize_brickwall_circuit(ufunc, U, Vlist_start, nlayers, L, (const int**)perms, &params, niter, f_iter, Vlist_opt);

	// construct return value
	// f_iter
	npy_intp dims_f_iter[1] = { niter };
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
	memcpy(PyArray_DATA(f_iter_obj), f_iter, niter * sizeof(double));
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

static PyObject* optimize_brickwall_circuit_py(PyObject* self, PyObject* args)
{
	// suppress "unused parameter" warning
	(void)self;
	(void)args;

	PyErr_SetString(PyExc_RuntimeError, "cannot perform optimization - please re-build with support for complex numbers enabled");

	return NULL;
}

#endif


static PyMethodDef methods[] = {
	{ "optimize_brickwall_circuit", optimize_brickwall_circuit_py, METH_VARARGS, "Optimize the quantum gates in a brickwall layout to approximate a unitary matrix using a trust-region method." },
	{ NULL, NULL, 0, NULL }     // sentinel
};


static struct PyModuleDef module = {
	PyModuleDef_HEAD_INIT,
	"rqcopt_hpc",   // name of module
	NULL,           // module documentation, may be NULL
	-1,             // size of per-interpreter state of the module, or -1 if the module keeps state in global variables
	methods         // module methods
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
