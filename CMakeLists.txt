
cmake_minimum_required(VERSION 3.8)

project(rqcopt_hpc)

find_package(BLAS REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

set(CMAKE_C_FLAGS "-Wall -Wextra -O3")

add_compile_definitions(COMPLEX_CIRCUIT)

set(RQCOPT_SOURCES "src/matrix.c" "src/statevector.c" "src/gate.c" "src/quantum_circuit.c" "src/brickwall_circuit.c" "src/target.c" "src/trust_region.c" "src/brickwall_opt.c" "src/util.c")

add_executable(            rqcopt_ising1d ${RQCOPT_SOURCES} "examples/ising1d/ising1d_dynamics_opt.c" "examples/timing/timing.c")
target_include_directories(rqcopt_ising1d PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
target_link_libraries(     rqcopt_ising1d PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})

add_library(               rqcopt_pymodule SHARED ${RQCOPT_SOURCES} "pymodule/rqcopt_module.c")
target_include_directories(rqcopt_pymodule PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} "src")
target_link_libraries(     rqcopt_pymodule PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES} ${Python3_LIBRARIES})
set_target_properties(     rqcopt_pymodule PROPERTIES PREFIX "" OUTPUT_NAME "rqcopt_hpc" LINKER_LANGUAGE C)

set(TEST_SOURCES "test/numerical_gradient.c" "test/test_matrix.c" "test/test_statevector.c" "test/test_gate.c" "test/test_quantum_circuit.c" "test/test_brickwall_circuit.c" "test/test_target.c" "test/test_trust_region.c" "test/run_tests.c")
add_executable(            rqcopt_test ${RQCOPT_SOURCES} ${TEST_SOURCES})
target_include_directories(rqcopt_test PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src")
target_link_libraries(     rqcopt_test PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
