import numpy as np
import pytest

from src import dynamics as dy
from src import models
from src.states import a_form, random_cptp, random_density, unvectorise, vectorise

ETA = np.pi / 10


def test_gates_are_unitary():
    for u in (models.u1(ETA), models.u2(ETA), models.u3(ETA), models.swap_env()):
        assert np.allclose(u @ u.conj().T, np.eye(8))


def test_u3_swaps_the_two_environments():
    # eta = pi/2: U3 = i * S_E up to the identity part (cos = 0), so same action as S_E
    assert np.allclose(models.u3(np.pi / 2), 1j * models.swap_env())


def test_a_form_matches_conjugation():
    u, rho = models.u1(ETA), random_density(8, rng=1)
    assert np.allclose(u @ rho @ u.conj().T, unvectorise(a_form(u) @ vectorise(rho)))


@pytest.mark.parametrize("k", range(7))
def test_every_model_is_trace_preserving(k):
    # a trace preserving map has the vectorised identity as a left eigenvector with eigenvalue 1
    a = models.MODELS[k]()
    ident = vectorise(np.eye(8)).ravel()
    assert np.allclose(ident @ a, ident)


def test_random_cptp_is_trace_preserving_and_positive():
    r = random_cptp(8, rng=2)
    ident = vectorise(np.eye(8)).ravel()
    assert np.allclose(ident @ r, ident)
    out = unvectorise(r @ vectorise(random_density(8, rng=3)))
    assert np.isclose(np.trace(out), 1)
    assert np.all(np.linalg.eigvalsh((out + out.conj().T) / 2) > -1e-12)


def test_average_map_equals_mean_of_random_sequences():
    # Figure 3.3: the average map reproduces the mean over random swap sequences
    rho0 = dy.initial_state(rng=4)
    _, exact = dy.evolve_average(rho0, models.model0(ETA), 20)
    runs = [dy.evolve_sequence(rho0, ETA, 20, "random", rng=i) for i in range(3000)]
    assert np.allclose(exact, np.mean(runs, axis=0), atol=0.03)


def test_models_0_2_5_share_system_dynamics():
    # Section 4.2: acting on the environments only does not change the system's b_z
    rho0 = dy.initial_state(rng=5)
    eta = np.pi / 50
    _, b0 = dy.evolve_average(rho0, models.model0(eta), 400)
    _, b2 = dy.evolve_average(rho0, models.model2(eta), 400)
    _, b5 = dy.evolve_average(rho0, models.model5(eta, np.pi / 10), 400)
    assert np.allclose(b0, b2, atol=1e-9)
    assert np.allclose(b0, b5, atol=1e-9)


def test_ordered_swaps_oscillate_without_decay():
    # Figure 3.2 (left): the size of the oscillations stays constant for ordered swaps
    bz = dy.evolve_sequence(dy.initial_state(rng=6), ETA, 2000, "ordered")
    assert bz[-500:].std() == pytest.approx(bz[:500].std(), rel=0.02)
