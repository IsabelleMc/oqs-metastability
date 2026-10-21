"""Spectral theory of metastability (thesis Chapter 2 and Sections 4.1-4.3)."""

import numpy as np

from .states import random_cptp

DEFAULT_THRESHOLD = 1e-3  # Section 2.2.1: ratio below 10^-3 counts as a large gap


def generator_eigenvalues(dynamical_map):
    """Generator eigenvalues  lambda_L ~ (lambda_Lambda - 1) / t  at t = 1  (Eq. 4.2).

    Returned sorted by decreasing real part, so the stationary eigenvalue 0 comes first.
    """
    ev = np.linalg.eigvals(dynamical_map - np.eye(dynamical_map.shape[0]))
    return ev[np.argsort(-ev.real, kind="stable")]


def unique_real_parts(eigenvalues, rtol=1e-4, atol=1e-12):
    """Distinct values of Re[lambda], merging near-degenerate ones. Sorted, 0 first."""
    vals = np.sort(eigenvalues.real)[::-1]
    out = []
    for v in vals:
        if not out or abs(v - out[-1]) > max(atol, rtol * abs(out[-1])):
            out.append(v)
    return np.array(out)


def largest_gap(eigenvalues, rtol=1e-4):
    """Largest separation in |Re lambda| between neighbouring non-zero eigenvalues.

    Returns (lambda_m, lambda_m+1, ratio) with ratio = lambda_m / lambda_m+1  (Eq. 2.7),
    chosen as the *smallest* ratio, i.e. the biggest gap between distinct values.
    """
    re = np.abs(unique_real_parts(eigenvalues, rtol=rtol))
    re = re[re > 1e-14 + 0 * re.max()]
    re = np.sort(re)
    if len(re) < 2:
        return re[0], np.nan, np.nan
    ratios = re[:-1] / re[1:]
    m = int(np.argmin(ratios))
    return re[m], re[m + 1], ratios[m]


def is_metastable(eigenvalues, threshold=DEFAULT_THRESHOLD):
    """True if the largest spectral gap ratio is below ``threshold``."""
    return largest_gap(eigenvalues)[2] < threshold


def perturbed_map(dynamical_map, c, rng=None):
    """Mix with a random CPTP map:  G = (1 - c) Lambda + c R  (Eq. 4.20)."""
    d = int(round(np.sqrt(dynamical_map.shape[0])))
    return (1 - c) * dynamical_map + c * random_cptp(d, rng)


def minimal_metastable_liouvillian(gamma, kappa, omega):
    """Generator of the minimal metastable qubit of Section 2.2.1 (4 x 4, A-form).

    H = -omega S_z, dephasing J_z = sqrt(gamma) S_z and an infinite-temperature bath
    J_pm = sqrt(kappa / 2) S_pm, with spin-1/2 operators S = sigma / 2.
    Its eigenvalues are 0, -kappa and -(gamma + kappa)/2 -/+ i omega.
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
    sz = np.diag([1, -1]).astype(complex) / 2
    h = -omega * sz
    jumps = [np.sqrt(gamma) * sz, np.sqrt(kappa / 2) * (sx + 1j * sy), np.sqrt(kappa / 2) * (sx - 1j * sy)]
    eye = np.eye(2)
    # row-major vectorisation: vec(A rho B) = (A kron B^T) vec(rho)
    lv = -1j * (np.kron(h, eye) - np.kron(eye, h.T))
    for j in jumps:
        jdj = j.conj().T @ j
        lv += np.kron(j, j.conj()) - 0.5 * np.kron(jdj, eye) - 0.5 * np.kron(eye, jdj.T)
    return lv
