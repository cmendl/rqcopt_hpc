import numpy as np
from scipy.sparse.linalg import LinearOperator, svds
from .brickwall_circuit_matfree import apply_brickwall_unitary, apply_adjoint_brickwall_unitary, brickwall_unitary_gradient_vector_matfree, brickwall_unitary_hessian_matrix_matfree
from .trust_region import riemannian_trust_region_optimize
from .util import polar_decomp, real_to_antisymm


def _f_unitary_target_matfree(Vlist, L: int, Ufunc, perms):
    """
    Evaluate target function -Re tr[U^{\dagger} W],
    using the provided matrix-free application of U to a state.
    """
    f = 0
    # implement trace via summation over unit vectors
    for b in range(2**L):
        psi = np.zeros(2**L)
        psi[b] = 1
        Upsi = Ufunc(psi)
        Vpsi = apply_brickwall_unitary(Vlist, L, psi, perms)
        f -= np.vdot(Vpsi, Upsi).real
    return f


def brickwall_quadratic_model_matfree(Vlist, L: int, Ufunc, perms, hlist, rng: np.random.Generator=None):
    """
    Compute target function along a random direction in tangent space,
    and the corresponding quadratic approximation.
    """
    n = len(Vlist)
    # target function
    f = lambda vlist: _f_unitary_target_matfree(vlist, L, Ufunc, perms)
    f0 = f(Vlist)
    # gradient
    grad = -brickwall_unitary_gradient_vector_matfree(Vlist, L, Ufunc, perms)
    # Hessian matrix; eigenvalues can be of either sign
    H = -brickwall_unitary_hessian_matrix_matfree(Vlist, L, Ufunc, perms)
    # random direction
    if rng is None: rng = np.random.default_rng()
    eta = rng.standard_normal(n * 16)
    eta /= np.linalg.norm(eta)
    # model function (Taylor approximation)
    q = lambda h: f0 + h*np.dot(grad, eta) + 0.5 * h**2 * np.dot(eta, H @ eta)
    # target function in direction 'eta'
    eta_mat = np.reshape(eta, (n, 4, 4))
    eta_mat = [Vlist[j] @ real_to_antisymm(eta_mat[j]) for j in range(n)]
    feta = lambda h: f([polar_decomp(Vlist[j] + h*eta_mat[j])[0] for j in range(n)])
    # return eta and the target function and quadratic model evaluated at 'hlist'
    return eta, np.array([feta(h) for h in hlist]), np.array([q(h) for h in hlist])


def optimize_brickwall_circuit_matfree(L: int, Ufunc, Uadjfunc, Vlist_start, perms, rng: np.random.Generator=None, **kwargs):
    """
    Optimize the quantum gates in a brickwall layout to approximate
    the unitary matrix `U` using a trust-region method.
    """
    # target function
    f = lambda vlist: _f_unitary_target_matfree(vlist, L, Ufunc, perms)
    gradfunc = lambda vlist: -brickwall_unitary_gradient_vector_matfree(vlist, L, Ufunc, perms)
    hessfunc = lambda vlist: -brickwall_unitary_hessian_matrix_matfree(vlist, L, Ufunc, perms)
    # quantify error by spectral norm
    if rng is None: rng = np.random.default_rng()
    errfunc = lambda vlist: spectral_norm_matfree((2**L, 2**L),
                lambda psi: apply_brickwall_unitary(vlist, L, psi, perms) - Ufunc(np.reshape(psi, -1)),
                lambda psi: apply_adjoint_brickwall_unitary(vlist, L, psi, perms) - Uadjfunc(np.reshape(psi, -1)), rng)
    kwargs["gfunc"] = errfunc
    # perform optimization
    Vlist, f_iter, err_iter = riemannian_trust_region_optimize(
        f, retract_unitary_list, gradfunc, hessfunc, np.stack(Vlist_start), **kwargs)
    return Vlist, f_iter, err_iter


def _f_blockenc_target_matfree(Vlist, L: int, Hfunc, perms, P):
    """
    Evaluate target function || P^{\dagger} W P - H ||_F^2 / 2,
    using the provided matrix-free application of H to a state.
    """
    f = 0
    # implement Frobenius norm via summation over unit vectors
    for b in range(P.shape[1]):
        psi = np.zeros(P.shape[1])
        psi[b] = 1
        Hpsi = Hfunc(psi)
        Vpsi = P.conj().T @ apply_brickwall_unitary(Vlist, L, P @ psi, perms)
        f += 0.5 * np.linalg.norm(Vpsi - Hpsi)**2
    return f


def retract_unitary_list(vlist, eta):
    """
    Retraction for unitary matrices, with tangent direction represented as anti-symmetric matrices.
    """
    n = len(vlist)
    eta = np.reshape(eta, (n, 4, 4))
    dvlist = [vlist[j] @ real_to_antisymm(eta[j]) for j in range(n)]
    return np.stack([polar_decomp(vlist[j] + dvlist[j])[0] for j in range(n)])


def spectral_norm_matfree(shape, Afunc, Aadjfunc, rng: np.random.Generator=None):
    """
    Compute spectral norm (largest singular value) of a linear operator A,
    using the provided matrix-free application of A and adjoint of A.
    """
    if rng is None:
        rng = np.random.default_rng()
    return max(svds(LinearOperator(shape, matvec=Afunc, rmatvec=Aadjfunc),
                    k=6, which="LM", return_singular_vectors=False, maxiter=100, solver="lobpcg", random_state=rng))
