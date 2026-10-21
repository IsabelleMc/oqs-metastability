"""Density matrices, vectorisation and random quantum maps (thesis Chapters 1-2).

Vectorisation is row-major, so that  vec(U rho U^dagger) = (U kron conj(U)) vec(rho).
This is the "A-form" of a map (thesis Eq. 3.7).
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
SIGMA_Z = np.diag([1, -1]).astype(complex)
KET0 = np.diag([1, 0]).astype(complex)  # |0><0|


def random_density(n, rng=None):
    """Random n x n density matrix  rho = A A^dagger / Tr[A A^dagger]  (Eq. 1.2)."""
    rng = np.random.default_rng(rng)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def vectorise(rho):
    """Stack the rows of rho into a column vector |rho>>."""
    return rho.reshape(-1, 1)


def unvectorise(vec):
    """Inverse of :func:`vectorise`."""
    n = int(round(np.sqrt(vec.size)))
    return vec.reshape(n, n)


def a_form(u):
    """A-form of a unitary:  U kron U*  (Eq. 3.7). Acts on vectorised states."""
    return np.kron(u, u.conj())


def random_cptp(d, rng=None):
    """Random CPTP map in A-form (Eq. 2.8), acting on d x d density matrices.

    Draw d^2 random complex matrices F_k, set F_0 = (sum_k F_k^dag F_k)^(-1/2) and
    return  R = sum_k (F_k F_0) kron (F_k F_0)*.  The Kraus operators F_k F_0 satisfy
    sum_k (F_k F_0)^dag (F_k F_0) = 1, so R is trace preserving.
    """
    rng = np.random.default_rng(rng)
    f = rng.normal(size=(d * d, d, d)) + 1j * rng.normal(size=(d * d, d, d))
    s = np.einsum("kji,kjl->il", f.conj(), f)  # sum_k F_k^dag F_k
    w, v = np.linalg.eigh(s)
    f0 = (v * w ** -0.5) @ v.conj().T
    kraus = f @ f0
    return sum(np.kron(k, k.conj()) for k in kraus)
