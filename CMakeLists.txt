
cmake_minimum_required(VERSION 3.8)

# compile the executable for unit tests as for now
project(rqcopt_hpc)

# add sources to this project's executable
add_executable(rqcopt_hpc "src/matrix.c" "src/statevector.c" "src/parallel_gates.c" "src/brickwall_circuit.c" "src/trust_region.c" "src/util.c" "test/io_util.c" "test/test_matrix.c" "test/test_statevector.c" "test/test_parallel_gates.c" "test/test_brickwall_circuit.c" "test/test_trust_region.c" "test/run_tests.c")

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(BLAS REQUIRED)

target_include_directories(rqcopt_hpc PRIVATE "src" ${HDF5_INCLUDE_DIRS})

target_link_libraries(rqcopt_hpc PRIVATE ${HDF5_LIBRARIES} ${BLAS_LIBRARIES})

add_compile_definitions(COMPLEX_CIRCUIT)
