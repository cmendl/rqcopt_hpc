
cmake_minimum_required(VERSION 3.8)

# compile the executable for unit tests as for now
project(rqcopt_hpc)

# add sources to this project's executable
add_executable(rqcopt_hpc "src/matrix.c" "src/statevector.c" "src/parallel_gates.c" "src/brickwall_circuit.c" "src/util.c" "test/file_io.c" "test/test_matrix.c" "test/test_statevector.c" "test/test_parallel_gates.c" "test/test_brickwall_circuit.c" "test/run_tests.c")

target_include_directories(rqcopt_hpc PUBLIC "src")

# math library, required for complex number operations
target_link_libraries(rqcopt_hpc m)

add_compile_definitions(COMPLEX_CIRCUIT)
