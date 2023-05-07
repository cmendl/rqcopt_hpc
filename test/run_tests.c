#include <stdio.h>


typedef char* (*test_function)();


struct test
{
	test_function func;
	const char* name;
};


char* test_apply_gate();
char* test_apply_parallel_gates();
char* test_apply_parallel_gates_directed_grad();


int main()
{
	struct test tests[] = {
		{ .func = test_apply_gate,                         .name = "test_apply_gate" },
		{ .func = test_apply_parallel_gates,               .name = "test_apply_parallel_gates" },
		{ .func = test_apply_parallel_gates_directed_grad, .name = "test_apply_parallel_gates_directed_grad" },
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
