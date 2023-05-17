#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_symm();
char* test_antisymm();
char* test_multiply();
char* test_project_unitary_tangent();
char* test_transpose_statevector();
char* test_apply_gate();
char* test_apply_parallel_gates();
char* test_apply_parallel_gates_directed_grad();
char* test_parallel_gates_grad_matfree();
char* test_parallel_gates_hess_matfree();
char* test_apply_brickwall_unitary();
char* test_apply_adjoint_brickwall_unitary();
char* test_brickwall_unitary_grad_matfree();


int main()
{
	struct test tests[] = {
		{ .func = test_symm,                               .name = "test_symm" },
		{ .func = test_antisymm,                           .name = "test_antisymm" },
		{ .func = test_multiply,                           .name = "test_multiply" },
		{ .func = test_project_unitary_tangent,            .name = "test_project_unitary_tangent" },
		{ .func = test_transpose_statevector,              .name = "test_transpose_statevector" },
		{ .func = test_apply_gate,                         .name = "test_apply_gate" },
		{ .func = test_apply_parallel_gates,               .name = "test_apply_parallel_gates" },
		{ .func = test_apply_parallel_gates_directed_grad, .name = "test_apply_parallel_gates_directed_grad" },
		{ .func = test_parallel_gates_grad_matfree,        .name = "test_parallel_gates_grad_matfree" },
		{ .func = test_parallel_gates_hess_matfree,        .name = "test_parallel_gates_hess_matfree" },
		{ .func = test_apply_brickwall_unitary,            .name = "test_apply_brickwall_unitary" },
		{ .func = test_apply_adjoint_brickwall_unitary,    .name = "test_apply_adjoint_brickwall_unitary" },
		{ .func = test_brickwall_unitary_grad_matfree,     .name = "test_brickwall_unitary_grad_matfree" },
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
