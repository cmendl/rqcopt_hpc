import numpy as np
from .gate import apply_gate
from .util import project_unitary_tangent, antisymm, real_to_antisymm, antisymm_to_real


def apply_parallel_gates(V, L: int, psi, perm=None):
    """
    Apply a parallel sequence of two-qubit gates V to state psi,
    optionally using a permutation of quantum wires.
    """
    assert L % 2 == 0
    if perm is None:
        perm = range(L)
    for i in range(0, L, 2):
        psi = apply_gate(V, L, perm[i], perm[i + 1], psi)
    return psi


def parallel_gates_grad_matfree(V, L, Ufunc, perm=None):
    """
    Compute the gradient of Re tr[U† (V ⊗ ... ⊗ V)] with respect to V,
    using the provided matrix-free application of U to a state.
    """
    assert V.shape == (4, 4)
    assert L % 2 == 0
    if perm is not None:
        inv_perm = np.argsort(perm)
    G = np.zeros_like(V)
    # implement trace via summation over unit vectors
    for b in range(2**L):
        psi = np.zeros(2**L)
        psi[b] = 1
        if perm is not None:
            psi = np.reshape(np.transpose(np.reshape(psi, L * (2,)), inv_perm), -1)
        psi = Ufunc(psi)
        if perm is not None:
            psi = np.reshape(np.transpose(np.reshape(psi, L * (2,)), perm), -1)
        for i in range(0, L, 2):
            chi = psi.copy()
            for j in range(0, i, 2):
                chi = np.dot(np.reshape(chi, (-1, 4)), V[:, (b >> j) & 3].conj())
            for j in reversed(range(i + 2, L, 2)):
                chi = np.tensordot(V[:, (b >> j) & 3].conj(), np.reshape(chi, (4, -1)), axes=(0, 0))
            assert chi.shape == (4,)
            G[:, (b >> i) & 3] += chi
    return G


def apply_parallel_gates_directed_grad(V, L, Z, psi, perm=None):
    """
    Apply the gradient of V ⊗ ... ⊗ V in direction Z to state psi.
    """
    assert L % 2 == 0
    if perm is None:
        perm = range(L)
    Gpsi = 0
    for i in range(0, L, 2):
        chi = psi.copy()
        for j in range(0, L, 2):
            chi = apply_gate(Z if i == j else V, L, perm[j], perm[j + 1], chi)
        Gpsi += chi
    return Gpsi


def parallel_gates_hess_matfree(V, L, Z, Ufunc, perm=None, unitary_proj=False):
    """
    Compute the Hessian of V -> Re tr[U† (V ⊗ ... ⊗ V)] in direction Z,
    using the provided matrix-free application of U to a state.
    """
    assert V.shape == (4, 4)
    assert Z.shape == (4, 4)
    assert L % 2 == 0
    if perm is not None:
        inv_perm = np.argsort(perm)
    G = np.zeros_like(V)
    # implement trace via summation over unit vectors
    for b in range(2**L):
        psi = np.zeros(2**L)
        psi[b] = 1
        if perm is not None:
            psi = np.reshape(np.transpose(np.reshape(psi, L * (2,)), inv_perm), -1)
        psi = Ufunc(psi)
        if perm is not None:
            psi = np.reshape(np.transpose(np.reshape(psi, L * (2,)), perm), -1)
        for i in range(0, L, 2):
            for j in range(0, L, 2):
                if j == i:
                    continue
                chi = psi.copy()
                for k in range(0, i, 2):
                    x = Z[:, (b >> k) & 3] if k == j else V[:, (b >> k) & 3]
                    chi = np.dot(np.reshape(chi, (-1, 4)), x.conj())
                for k in reversed(range(i + 2, L, 2)):
                    x = Z[:, (b >> k) & 3] if k == j else V[:, (b >> k) & 3]
                    chi = np.tensordot(x.conj(), np.reshape(chi, (4, -1)), axes=(0, 0))
                assert chi.shape == (4,)
                G[:, (b >> i) & 3] += chi
    if unitary_proj:
        G = project_unitary_tangent(V, G)
        # additional terms resulting from the projection of the gradient
        # onto the Stiefel manifold (unitary matrices)
        grad = parallel_gates_grad_matfree(V, L, Ufunc, perm)
        G -= 0.5 * (Z @ grad.conj().T @ V + V @ grad.conj().T @ Z)
        if not np.allclose(Z, project_unitary_tangent(V, Z)):
            G -= 0.5 * (Z @ V.conj().T + V @ Z.conj().T) @ grad
    return G


def apply_brickwall_unitary(Vlist, L: int, psi, perms):
    """
    Apply the unitary matrix representation of a brickwall-type
    quantum circuit with periodic boundary conditions to state psi.
    """
    for V, perm in zip(Vlist, perms):
        psi = apply_parallel_gates(V, L, psi, perm)
    return psi


def apply_adjoint_brickwall_unitary(Vlist, L: int, psi, perms):
    """
    Apply the adjoint unitary matrix representation of a brickwall-type
    quantum circuit with periodic boundary conditions to state psi.
    """
    for V, perm in zip(reversed(Vlist), reversed(perms)):
        psi = apply_parallel_gates(V.conj().T, L, psi, perm)
    return psi


def brickwall_unitary_grad_matfree(Vlist, L, Ufunc, perms):
    """
    Compute the gradient of Re tr[U† W] with respect to Vlist,
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    return np.stack([
        parallel_gates_grad_matfree(Vlist[j], L,
            lambda psi: apply_adjoint_brickwall_unitary(Vlist[j+1:], L,
                        Ufunc(
                        apply_adjoint_brickwall_unitary(Vlist[:j], L, psi, perms[:j])),
                            perms[j+1:]),
                    perms[j])
            for j in range(len(Vlist))])


def brickwall_unitary_gradient_vector_matfree(Vlist, L, Ufunc, perms):
    """
    Represent the gradient of Re tr[U† W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    grad = brickwall_unitary_grad_matfree(Vlist, L, Ufunc, perms)
    # project gradient onto unitary manifold, represent as anti-symmetric matrix
    # and then convert to a vector
    return np.stack([antisymm_to_real(
        antisymm(Vlist[j].conj().T @ grad[j]))
        for j in range(len(grad))]).reshape(-1)


def apply_brickwall_unitary_directed_grad(Vlist, L, Z, k, psi, perms):
    """
    Apply the gradient of W in direction Z with respect to Vlist[k] to psi,
    where W is the brickwall circuit constructed from the gates in Vlist.
    """
    return apply_brickwall_unitary(Vlist[k+1:], L,
           apply_parallel_gates_directed_grad(Vlist[k], L, Z,
           apply_brickwall_unitary(Vlist[:k], L, psi, perms[:k]), perms[k]), perms[k+1:])


def brickwall_unitary_hess_matfree(Vlist, L, Z, k, Ufunc, perms, unitary_proj=False):
    """
    Compute the Hessian of Re tr[U† W] in direction Z with respect to Vlist[k],
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    n = len(Vlist)
    dVlist = np.stack([np.zeros_like(V) for V in Vlist])
    for j in range(k):
        # j < k
        # directed gradient with respect to Vlist[k] in direction Z
        UdZk = (lambda psi:
            apply_adjoint_brickwall_unitary(Vlist[j+1:k], L,
            apply_parallel_gates_directed_grad(Vlist[k].conj().T, L, Z.conj().T,
            apply_adjoint_brickwall_unitary(Vlist[k+1:], L,
            Ufunc(
            apply_adjoint_brickwall_unitary(Vlist[:j], L, psi, perms[:j])),
                perms[k+1:]), perms[k]), perms[j+1:k]))
        dVj = parallel_gates_grad_matfree(Vlist[j], L, UdZk, perms[j])
        if unitary_proj:
            dVlist[j] += project_unitary_tangent(Vlist[j], dVj)
        else:
            dVlist[j] += dVj

    # Hessian for layer k
    Ueff = (lambda psi:
        apply_adjoint_brickwall_unitary(Vlist[k+1:], L,
        Ufunc(
        apply_adjoint_brickwall_unitary(Vlist[:k], L, psi, perms[:k])), perms[k+1:]))
    dVlist[k] += parallel_gates_hess_matfree(Vlist[k], L, Z, Ueff, perms[k], unitary_proj=unitary_proj)

    for j in range(k + 1, n):
        # k < j
        # directed gradient with respect to Vlist[k] in direction Z
        UdZk = (lambda psi:
            apply_adjoint_brickwall_unitary(Vlist[j+1:], L,
            Ufunc(
            apply_adjoint_brickwall_unitary(Vlist[:k], L,
            apply_parallel_gates_directed_grad(Vlist[k].conj().T, L, Z.conj().T,
            apply_adjoint_brickwall_unitary(Vlist[k+1:j], L, psi, perms[k+1:j]),
                perms[k]), perms[:k])), perms[j+1:]))
        dVj = parallel_gates_grad_matfree(Vlist[j], L, UdZk, perms[j])
        if unitary_proj:
            dVlist[j] += project_unitary_tangent(Vlist[j], dVj)
        else:
            dVlist[j] += dVj

    return dVlist


def brickwall_unitary_hessian_matrix_matfree(Vlist, L, Ufunc, perms):
    """
    Construct the Hessian matrix of Re tr[U† W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    using the provided matrix-free application of U to a state.
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16))
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = real_to_antisymm(np.reshape(Z, (4, 4)))
            dVZj = brickwall_unitary_hess_matfree(Vlist, L, Vlist[j] @ Z, j, Ufunc, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = antisymm_to_real(antisymm(Vlist[i].conj().T @ dVZj[i])).reshape(-1)
    return H.reshape((n * 16, n * 16))


def squared_brickwall_unitary_grad_matfree(Vlist, L, Afunc, Bfunc, perms):
    """
    Compute the gradient of tr[A W† B W] with respect to Vlist,
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    return 2 * brickwall_unitary_grad_matfree(Vlist, L, lambda psi: Bfunc(apply_brickwall_unitary(Vlist, L, Afunc(psi), perms)), perms)


def squared_brickwall_unitary_gradient_vector_matfree(Vlist, L, Afunc, Bfunc, perms):
    """
    Represent the gradient of tr[A W† B W] as real vector,
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    return 2 * brickwall_unitary_gradient_vector_matfree(Vlist, L, lambda psi: Bfunc(apply_brickwall_unitary(Vlist, L, Afunc(psi), perms)), perms)


def squared_brickwall_unitary_hess_matfree(Vlist, L, Z, k, Afunc, Bfunc, perms, unitary_proj=False):
    """
    Compute the Hessian of tr[A W† B W] in direction Z with respect to Vlist[k],
    where W is the brickwall circuit constructed from the gates in Vlist
    and A and B are Hermitian.
    """
    H1 = brickwall_unitary_hess_matfree(Vlist, L, Z, k, lambda psi: Bfunc(apply_brickwall_unitary(Vlist, L, Afunc(psi), perms)), perms, unitary_proj)
    H2 = brickwall_unitary_grad_matfree(Vlist, L, lambda psi: Bfunc(apply_brickwall_unitary_directed_grad(Vlist, L, Z, k, Afunc(psi), perms)), perms)
    if unitary_proj:
        H2 = np.stack([project_unitary_tangent(Vlist[j], dVj) for j, dVj in enumerate(H2)])
    return 2 * (H1 + H2)


def squared_brickwall_unitary_hessian_matrix_matfree(Vlist, L, Afunc, Bfunc, perms):
    """
    Construct the Hessian matrix of tr[A W† B W] with respect to Vlist
    defining the layers of W,
    where W is the brickwall circuit constructed from the gates in Vlist,
    and A and B are Hermitian..
    """
    n = len(Vlist)
    H = np.zeros((n, 16, n, 16))
    for j in range(n):
        for k in range(16):
            # unit vector
            Z = np.zeros(16)
            Z[k] = 1
            Z = real_to_antisymm(np.reshape(Z, (4, 4)))
            dVZj = squared_brickwall_unitary_hess_matfree(Vlist, L, Vlist[j] @ Z, j, Afunc, Bfunc, perms, unitary_proj=True)
            for i in range(n):
                H[i, :, j, k] = antisymm_to_real(antisymm(Vlist[i].conj().T @ dVZj[i])).reshape(-1)
    return H.reshape((n * 16, n * 16))
