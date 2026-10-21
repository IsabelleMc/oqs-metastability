"""Check the code against numbers printed in the thesis."""

import numpy as np
import pytest

from src import metastability as ms
from src import models
from src.reference import THESIS_MODELS, THESIS_PERTURBATION

ETA = np.pi / 1000

THESIS = [(k, v[1], v[2], v[2] < 1e-3) for k, v in THESIS_MODELS.items()]


@pytest.mark.parametrize("k, values, ratio, metastable", THESIS)
def test_model_spectrum_matches_thesis(k, values, ratio, metastable):
    ev = ms.generator_eigenvalues(models.MODELS[k]())
    found = np.abs(ms.unique_real_parts(ev, rtol=1e-3))
    for v in values:
        assert np.min(np.abs(found - v) / v) < 1e-3, f"model {k}: {v} not in spectrum"
    lam_m, lam_m1, r = ms.largest_gap(ev, rtol=1e-3)
    assert r == pytest.approx(ratio, rel=0.05)
    assert ms.is_metastable(ev) == metastable


def test_stationary_eigenvalue_is_zero_for_all_models():
    for f in models.MODELS.values():
        assert abs(ms.generator_eigenvalues(f())[0]) < 1e-12


def test_smallest_nonzero_eigenvalue_is_eta_squared():
    # Section 4.1.1: for models 1, 3, 4, 6 the slowest rate is eta^2
    for k in (1, 3, 4, 6):
        ev = ms.generator_eigenvalues(models.MODELS[k]())
        slowest = np.abs(ms.unique_real_parts(ev, rtol=1e-3))[1]
        assert slowest == pytest.approx(ETA**2, rel=1e-3)


def test_perturbation_exposes_gap():
    # Section 4.3: c = 1e-8, eta = pi/100 -> lambda_m ~ -1.05e-8, lambda_m+1 ~ -4.9333e-4
    g = ms.perturbed_map(models.model0(np.pi / 100), 1e-8, rng=0)
    lam_m, lam_m1, ratio = ms.largest_gap(ms.generator_eigenvalues(g), rtol=0)
    assert lam_m == pytest.approx(THESIS_PERTURBATION["lambda_m"], rel=0.05)
    assert lam_m1 == pytest.approx(THESIS_PERTURBATION["lambda_m1"], rel=1e-3)
    assert ratio < 1e-3


def test_minimal_metastable_system_eigenvalues():
    gamma, kappa = 1.0, 0.005
    omega = gamma + kappa
    ev = np.sort_complex(np.linalg.eigvals(ms.minimal_metastable_liouvillian(gamma, kappa, omega)))
    expected = [0, -kappa, -(gamma + kappa) / 2 - 1j * omega, -(gamma + kappa) / 2 + 1j * omega]
    assert np.allclose(np.sort_complex(np.array(expected, dtype=complex)), ev, atol=1e-10)
