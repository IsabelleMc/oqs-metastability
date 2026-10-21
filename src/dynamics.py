"""System dynamics: survival probability and random / ordered collision sequences (Chapter 3)."""

import numpy as np
from scipy.signal import find_peaks

from . import models
from .states import I2, KET0, SIGMA_Z, random_density, unvectorise, vectorise

PZ = np.kron(np.kron(I2, SIGMA_Z), I2)  # sigma_z on the system qubit  (Section 3.1.1)
_PZ_VEC = PZ.T.reshape(-1)               # so that Tr[Pz rho] = _PZ_VEC . vec(rho)


def initial_state(system=None, rng=None):
    """rho = |0><0|_E1 (x) rho_S (x) |0><0|_E2 with a random system state by default."""
    if system is None:
        system = random_density(2, rng)
    return np.kron(np.kron(KET0, system), KET0)


def survival_probability(rho):
    """b_z = Tr[Pz rho]: z-polarisation of the system qubit (Eq. 3.11).

    Accepts a single 8x8 state or a stack of them, and returns real numbers.
    """
    return np.real(np.einsum("...ij,ji->...", rho, PZ))


def evolve_average(rho0, dynamical_map, steps, record_every=1):
    """Survival probability under repeated application of an average map  Lambda^t (Eq. 3.13).

    Returns (times, b_z). Only b_z is stored, so very long runs (10^5 steps) are cheap.
    """
    vec = vectorise(rho0).ravel()
    times = np.arange(0, steps + 1, record_every)
    out = np.empty(len(times))
    out[0] = np.real(_PZ_VEC @ vec)
    k = 1
    for t in range(1, steps + 1):
        vec = dynamical_map @ vec
        if t % record_every == 0:
            out[k] = np.real(_PZ_VEC @ vec)
            k += 1
    return times, out


def evolve_sequence(rho0, eta, steps, mode="random", rng=None):
    """Single trajectory of partial swaps: ``mode`` is "ordered" (U1 U2 U1 U2 ...) or "random".

    Returns b_z(t) for t = 0 .. steps  (Figure 3.2).
    """
    rng = np.random.default_rng(rng)
    gates = [models.u1(eta), models.u2(eta)]
    rho = rho0
    out = np.empty(steps + 1)
    out[0] = survival_probability(rho)
    for t in range(steps):
        u = gates[t % 2] if mode == "ordered" else gates[rng.integers(2)]
        rho = u @ rho @ u.conj().T
        out[t + 1] = survival_probability(rho)
    return out


def peaks(times, bz):
    """Times and values of the local maxima of the survival probability (Figures 3.5, 4.9)."""
    idx, _ = find_peaks(bz)
    return times[idx], bz[idx]


def time_averaged_deviation(bz, b_total):
    """|<Delta b_z>|(T) = | 1/(T+1) sum_{t<=T} b_z(t) - B/3 |  (Eq. 3.14, Benenti et al.).

    ``b_total`` is B, the sum of the three qubits' b_z at t = 0.
    """
    running = np.cumsum(bz) / np.arange(1, len(bz) + 1)
    return np.abs(running - b_total / 3)
