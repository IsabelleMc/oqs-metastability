import numpy as np
import pytest

from src import correlations as co
from src import models
from src.states import vectorise


def test_mutual_information_matches_thesis_code():
    # Reference values were produced by running the original ncon implementation from the
    # thesis notebook (Mutual_Info2.ipynb) on the same maps; agreement was ~1e-15.
    a = models.model0(np.pi / 10)
    assert co.mutual_information(a, 5, 6) == pytest.approx(0.5421060661, abs=1e-9)
    assert co.mutual_information(a, 5, 12) == pytest.approx(0.1967147094, abs=1e-9)
    assert co.mutual_information(a, 3, 40) == pytest.approx(0.1708096373, abs=1e-9)
    b = models.model0(np.pi / 500)
    assert co.mutual_information(b, 5, 25) == pytest.approx(0.0009357272, abs=1e-9)


def test_memoryless_map_has_zero_mutual_information():
    # A map that resets all three qubits to a fixed state each step forgets everything.
    sigma = np.kron(np.kron(np.diag([1, 0]), np.diag([0.7, 0.3])), np.diag([1, 0]))
    reset = vectorise(sigma) @ vectorise(np.eye(8)).T
    assert co.mutual_information(reset, 2, 3) == pytest.approx(0, abs=1e-12)


def test_mutual_information_is_non_negative():
    a = models.model4(np.pi / 20, np.pi / 5)
    vals = co.mutual_information_vs_gap(a, 4, range(0, 30, 3))
    assert np.all(vals > -1e-12)


def test_choi_state_is_valid():
    a = models.model0(np.pi / 10)
    ups = co.choi_two_time(np.linalg.matrix_power(a, 3), np.linalg.matrix_power(a, 5))
    assert np.isclose(np.trace(ups), 1)
    assert np.allclose(ups, ups.conj().T)
    assert np.all(np.linalg.eigvalsh(ups) > -1e-12)
