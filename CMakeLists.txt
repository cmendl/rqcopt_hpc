
cmake_minimum_required(VERSION 3.8)

project(rqcopt_hpc)

find_package(BLAS REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

set(CMAKE_C_FLAGS "-Wall -Wextra -O3")

add_compile_definitions(COMPLEX_CIRCUIT)

set(RQCOPT_SOURCES "src/matrix.c" "src/statevector.c" "src/gate.c" "src/parallel_gates.c" "src/brickwall_circuit.c" "src/target.c" "src/trust_region.c" "src/brickwall_opt.c" "src/util.c")
set(TEST_SOURCES "test/numerical_gradient.c" "test/test_matrix.c" "test/test_statevector.c" "test/test_gate.c" "test/test_parallel_gates.c" "test/test_brickwall_circuit.c" "test/test_target.c" "test/test_trust_region.c" "test/run_tests.c")

add_executable(rqcopt_ising1d ${RQCOPT_SOURCES} "examples/ising1d/ising1d_dynamics_opt.c")
add_executable(rqcopt_test    ${RQCOPT_SOURCES} ${TEST_SOURCES})

target_include_directories(rqcopt_ising1d PRIVATE "src" ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     rqcopt_ising1d PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})

target_include_directories(rqcopt_test PRIVATE "src" ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_link_libraries(     rqcopt_test PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
