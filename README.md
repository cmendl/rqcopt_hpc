[![CI](https://github.com/cmendl/rqcopt_hpc/actions/workflows/ci.yml/badge.svg)](https://github.com/cmendl/rqcopt_hpc/actions/workflows/ci.yml)


Riemannian quantum circuit optimization (C implementation)
==========================================================

The code uses a matrix-free and cache optimized application of single- and two-qubit gates.


Building
--------
The code requires the BLAS, HDF5 and Python 3 development libraries. These can be installed via `sudo apt install libblas-dev libhdf5-dev python3-dev` (on Ubuntu Linux) or similar.

From the project directory, use `cmake` to build the project:
```
mkdir build && cd build
cmake ../
cmake --build .
````


References
----------
-  Fabian Putterer, Max M. Zumpe, Isabel Nha Minh Le, Qunsheng Huang, Christian B. Mendl  
   _High-performance contraction of quantum circuits for Riemannian optimization_  
   ([arXiv:2506.23775](https://arxiv.org/abs/2506.23775))
-  Ayse Kotil, Rahul Banerjee, Qunsheng Huang, Christian B. Mendl  
   _Riemannian quantum circuit optimization for Hamiltonian simulation_  
   [J. Phys. A: Math. Theor. 57, 135303 (2024)](https://doi.org/10.1088/1751-8121/ad2d6e)
   ([arXiv:2212.07556](https://arxiv.org/abs/2212.07556))
-  P.-A. Absil, R. Mahony, R. Sepulchre  
   _Optimization Algorithms on Matrix Manifolds_  
   Princeton University Press (2008)  
   ([press.princeton.edu/absil](https://press.princeton.edu/absil))
