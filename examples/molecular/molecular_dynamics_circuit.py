"""
Simulation and optimization of the fermionic swap network circuit architecture.

Reference:
    Ian D. Kivlichan, Jarrod McClean, Nathan Wiebe, Craig Gidney, Alán Aspuru-Guzik, Garnet Kin-Lic Chan, Ryan Babbush
    Quantum simulation of electronic structure with linear depth and connectivity
    Phys. Rev. Lett. 120, 110501 (2018)
"""

import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import h5py
import qib
from qib.operator import FieldOperatorTerm, IFOType, IFODesc
from circuit import circuit_unitary
from io_util import interleave_complex, decode_complex
# compiled Python module; ensure that it is on the Python search path
import rqcopt_hpc as oc_hpc


def fermionic_simulation_gate(tkin: float, vint: float, t: float):
    """
    Construct the fermionic simulation gate.
    """
    c = np.cos(tkin * t)
    s = np.sin(tkin * t)
    v = np.exp(-1j * vint * t)
    return np.array([
        [1,   0,    0,    0],
        [0, -1j*s,  c,    0],
        [0,   c,  -1j*s,  0],
        [0,   0,    0,   -v],
    ])


def construct_fermionic_swapnet(tkin, vint, dt: float):
    """
    Construct the fermionic swap network approximation of the time evolution operator
    of a molecular Hamiltonian with diaognal interaction term.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    # currently only a fixed system size supported
    assert tkin.shape == (5, 5)
    assert vint.shape == (5, 5)
    gates = []
    wires = []
    gate = lambda i, j, t: fermionic_simulation_gate(tkin[i, j], vint[i, j], t)
    # layer 1
    gates.append(gate(0, 1, dt/2));  wires.append([0, 1])
    gates.append(gate(2, 3, dt/2));  wires.append([2, 3])
    # layer 2
    gates.append(gate(0, 3, dt/2));  wires.append([1, 2])
    gates.append(gate(2, 4, dt/2));  wires.append([3, 4])
    # layer 3
    gates.append(gate(1, 3, dt/2));  wires.append([0, 1])
    gates.append(gate(0, 4, dt/2));  wires.append([2, 3])
    # layer 4
    gates.append(gate(1, 4, dt/2));  wires.append([1, 2])
    gates.append(gate(0, 2, dt/2));  wires.append([3, 4])
    # layer 5
    # cannot simply use a full time step since swap needs to be applied twice
    gates5 = []
    wires5 = []
    gates5.append(gate(3, 4, dt/2) @ gate(3, 4, dt/2));  wires5.append([0, 1])
    gates5.append(gate(1, 2, dt/2) @ gate(1, 2, dt/2));  wires5.append([2, 3])
    # append same gates in reversed order to form a Strang step
    gates = gates + gates5 + list(reversed(gates))
    wires = wires + wires5 + list(reversed(wires))
    return gates, wires


def main(run_opt=False):

    # number of qubits (or orbitals)
    nqubits = 5
    print("nqubits:", nqubits)

    rng = np.random.default_rng(723)

    # construct Hamiltonian
    # underlying lattice
    latt = qib.lattice.FullyConnectedLattice((nqubits,))
    field = qib.field.Field(qib.field.ParticleType.FERMION, latt)
    # Hamiltonian coefficients
    tkin = rng.standard_normal(size=(nqubits, nqubits))
    tkin = 0.5 * (tkin + tkin.conj().T)
    for j in range(nqubits):
        tkin[j, j] = 0
    vint = np.triu(rng.standard_normal(size=(nqubits, nqubits)), 1)
    # kinetic hopping term
    T = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                           IFODesc(field, IFOType.FERMI_ANNIHIL)],
                           tkin)
    # interaction term
    delta = np.zeros((nqubits, nqubits, nqubits))
    for j in range(nqubits):
        delta[j, j, j] = 1
    vint_full = np.einsum(delta, (0, 1, 4), delta, (2, 3, 5), vint, (4, 5), (0, 1, 2, 3))
    V = FieldOperatorTerm([IFODesc(field, IFOType.FERMI_CREATE),
                           IFODesc(field, IFOType.FERMI_ANNIHIL),
                           IFODesc(field, IFOType.FERMI_CREATE),
                           IFODesc(field, IFOType.FERMI_ANNIHIL)],
                           vint_full)
    H = qib.FieldOperator([T, V]).as_matrix().todense()
    assert np.allclose(H, H.conj().T)
    # visualize spectrum
    λ = np.linalg.eigvalsh(H)
    plt.plot(λ, '.')
    plt.xlabel(r"$j$")
    plt.ylabel(r"$\lambda_j$")
    plt.title(f"Molecular Hamiltonian for diagonal interaction term and {nqubits} orbitals")
    plt.show()

    # reference global unitary
    t = 0.25
    print("t:", t)
    expiH = scipy.linalg.expm(-1j*H*t)

    # real-time evolution via Strang splitting for different time steps
    nsteps_stra = np.array([2**i for i in range(4)])
    err_stra = np.zeros(len(nsteps_stra))
    for i, nsteps in enumerate(nsteps_stra):
        print("nsteps:", nsteps)
        dt = t / nsteps
        print("dt:", dt)
        gates, wires = construct_fermionic_swapnet(tkin, vint, dt)
        C = circuit_unitary(nqubits, gates, wires)
        W = np.linalg.matrix_power(C, nsteps)
        err_stra[i] = np.linalg.norm(W - expiH, ord=2)
        print(f"err_stra[{i}]: {err_stra[i]}")
        fval = -np.trace(expiH.conj().T @ W).real
        print("target function value f(G):", fval)
        print("1 + f(G)/2^nqubits:", 1 + fval/2**nqubits)
        print(50 * "_")
    # convergence plot
    dt_list = t / nsteps_stra
    plt.loglog(dt_list, err_stra, '.-', label="Strang")
    plt.loglog(dt_list, np.array(dt_list)**2, '--', label="Δt^2")
    plt.xlabel("Δt")
    plt.ylabel("error")
    plt.legend()
    plt.title(f"Real-time evolution up to t = {t} using Strang splitting")
    plt.show()

    print("optimized circuit for nsteps = 1:")

    # gates and wires for a single step
    gates_start, wires = construct_fermionic_swapnet(tkin, vint, t)
    # save initial data to disk
    with h5py.File("molecular_dynamics_opt_init.hdf5", "w") as file:
        file["tkin"]  = tkin
        file["vint"]  = vint
        file["expiH"] = interleave_complex(expiH, "cplx")
        file["gates_start"] = interleave_complex(np.stack(gates_start), "cplx")
        file["wires"] = wires
        # store parameters
        file.attrs["nqubits"] = nqubits
        file.attrs["t"] = float(t)

    if run_opt:
        t_start = time.perf_counter()
        gates_opt, f_iter = oc_hpc.optimize_quantum_circuit(nqubits, expiH, gates_start, wires, 10)
        t_run = time.perf_counter() - t_start
        print(f"completed optimization in {t_run:g} seconds")
    else:
        # load optimized circuit gates from disk
        with h5py.File("molecular_dynamics_opt.hdf5", "r") as file:
            # parameters must agree
            assert np.array_equal(file["tkin"], tkin)
            assert np.array_equal(file["vint"], vint)
            assert np.array_equal(file["wires"], wires)
            assert file.attrs["nqubits"] == nqubits
            assert file.attrs["t"] == t
            gates_opt = decode_complex(file["gates"][:], "cplx")
            f_iter = file["f_iter"][:]

    # rescaled and shifted target function
    print("f_iter:", f_iter)
    plt.semilogy(range(len(f_iter)), 1 + np.array(f_iter) / 2**nqubits)
    plt.xlabel("iteration")
    plt.ylabel(r"$1 + f(G)/2^{\mathrm{nqubits}}$")
    plt.title(f"Optimization target function for a quantum circuit with {len(gates)} gates")
    plt.show()

    C_opt = circuit_unitary(nqubits, gates_opt, wires)
    err_opt = np.linalg.norm(C_opt - expiH, ord=2)
    print(f"err_opt: {err_opt} (after {len(f_iter) - 1} iterations)")
    print(f"before:  {err_stra[0]}")


if __name__ == "__main__":
    main(run_opt=True)
