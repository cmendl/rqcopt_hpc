Riemannian quantum circuit optimization (C implementation)
==========================================================

The code uses a matrix-free and cache optimized application of single- and two-qubit gates.


Building
--------
The code requires the BLAS and HDF5 development libraries. These can be installed via `sudo apt install libblas-dev` and `sudo apt install libhdf5-dev` (on Ubuntu Linux) or similar, respectively.

From the project directory, use `cmake` to build the project:
```
mkdir build && cd build
cmake ../
cmake --build .
````


References
----------
-  Ayse Kotil, Rahul Banerjee, Qunsheng Huang, Christian B. Mendl  
   _Riemannian quantum circuit optimization for Hamiltonian simulation_  
   ([arXiv:2212.07556](https://arxiv.org/abs/2212.07556))
-  P.-A. Absil, R. Mahony, R. Sepulchre  
   _Optimization Algorithms on Matrix Manifolds_  
   Princeton University Press (2008)  
   ([press.princeton.edu/absil](https://press.princeton.edu/absil))
