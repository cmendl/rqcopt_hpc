#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_symm();
char* test_antisymm();
#ifdef COMPLEX_CIRCUIT
char* test_real_to_antisymm();
char* test_real_to_unitary_tangent();
#else
char* test_real_to_skew();
char* test_real_to_ortho_tangent();
#endif
char* test_multiply();
char* test_project_unitary_tangent();
char* test_inverse_matrix();
char* test_polar_factor();
char* test_transpose_statevector();
char* test_apply_gate();
char* test_apply_gate_backward();
char* test_apply_gate_to_array();
char* test_apply_gate_placeholder();
char* test_apply_brickwall_unitary();
char* test_brickwall_unitary_backward();
char* test_brickwall_unitary_backward_hessian();
char* test_apply_quantum_circuit();
char* test_unitary_target();
char* test_unitary_target_and_gradient();
#ifdef COMPLEX_CIRCUIT
char* test_unitary_target_and_gradient_vector();
#endif
char* test_unitary_target_gradient_hessian();
#ifdef COMPLEX_CIRCUIT
char* test_unitary_target_gradient_vector_hessian_matrix();
#endif
char* test_blockenc_target();
char* test_blockenc_target_and_gradient();
#ifdef COMPLEX_CIRCUIT
char* test_blockenc_target_and_gradient_vector();
#endif
char* test_blockenc_target_gradient_hessian();
#ifdef COMPLEX_CIRCUIT
char* test_blockenc_target_gradient_vector_hessian_matrix();
#endif
char* test_truncated_cg();


#define TEST_FUNCTION_ENTRY(fname) { .func = fname, .name = #fname }


int main()
{
	struct test tests[] = {
		TEST_FUNCTION_ENTRY(test_symm),
		TEST_FUNCTION_ENTRY(test_antisymm),
		#ifdef COMPLEX_CIRCUIT
		TEST_FUNCTION_ENTRY(test_real_to_antisymm),
		TEST_FUNCTION_ENTRY(test_real_to_unitary_tangent),
		#else
		TEST_FUNCTION_ENTRY(test_real_to_skew),
		TEST_FUNCTION_ENTRY(test_real_to_ortho_tangent),
		#endif
		TEST_FUNCTION_ENTRY(test_multiply),
		TEST_FUNCTION_ENTRY(test_project_unitary_tangent),
		TEST_FUNCTION_ENTRY(test_inverse_matrix),
		TEST_FUNCTION_ENTRY(test_polar_factor),
		TEST_FUNCTION_ENTRY(test_transpose_statevector),
		TEST_FUNCTION_ENTRY(test_apply_gate),
		TEST_FUNCTION_ENTRY(test_apply_gate_backward),
		TEST_FUNCTION_ENTRY(test_apply_gate_to_array),
		TEST_FUNCTION_ENTRY(test_apply_gate_placeholder),
		TEST_FUNCTION_ENTRY(test_apply_brickwall_unitary),
		TEST_FUNCTION_ENTRY(test_brickwall_unitary_backward),
		TEST_FUNCTION_ENTRY(test_brickwall_unitary_backward_hessian),
		TEST_FUNCTION_ENTRY(test_apply_quantum_circuit),
		TEST_FUNCTION_ENTRY(test_unitary_target),
		TEST_FUNCTION_ENTRY(test_unitary_target_and_gradient),
		#ifdef COMPLEX_CIRCUIT
		TEST_FUNCTION_ENTRY(test_unitary_target_and_gradient_vector),
		#endif
		TEST_FUNCTION_ENTRY(test_unitary_target_gradient_hessian),
		#ifdef COMPLEX_CIRCUIT
		TEST_FUNCTION_ENTRY(test_unitary_target_gradient_vector_hessian_matrix),
		#endif
		TEST_FUNCTION_ENTRY(test_blockenc_target),
		TEST_FUNCTION_ENTRY(test_blockenc_target_and_gradient),
		#ifdef COMPLEX_CIRCUIT
		TEST_FUNCTION_ENTRY(test_blockenc_target_and_gradient_vector),
		#endif
		TEST_FUNCTION_ENTRY(test_blockenc_target_gradient_hessian),
		#ifdef COMPLEX_CIRCUIT
		TEST_FUNCTION_ENTRY(test_blockenc_target_gradient_vector_hessian_matrix),
		#endif
		TEST_FUNCTION_ENTRY(test_truncated_cg),
	};
	int num_tests = sizeof(tests) / sizeof(struct test);

	int num_pass = 0;
	for (int i = 0; i < num_tests; i++)
	{
		printf(".");
		char* msg = tests[i].func();
		if (msg == 0) {
			num_pass++;
		}
		else {
			printf("\nTest '%s' failed: %s\n", tests[i].name, msg);
		}
	}
	printf("\nNumber of successful tests: %i / %i\n", num_pass, num_tests);

	if (num_pass < num_tests)
	{
		printf("At least one test failed!\n");
	}
	else
	{
		printf("All tests passed.\n");
	}

	return num_pass != num_tests;
}
