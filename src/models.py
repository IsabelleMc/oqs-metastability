"""The partial swap collision model and its six modifications (thesis Chapters 3-4).

Qubit order is [E1, S, E2] -- environment 1, system, environment 2 -- exactly as in
the thesis (Eqs. 3.2-3.3), so Pz = 1 (x) sigma_z (x) 1 measures the system qubit.

All maps returned here are 64 x 64 matrices acting on vectorised 3-qubit states.
"""

import numpy as np

from .states import I2, a_form

I4 = np.eye(4, dtype=complex)
I8 = np.eye(8, dtype=complex)
SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)


def partial_swap_2q(eta):
    """Two-qubit partial swap  cos(eta) 1 + i sin(eta) S  (Eq. 3.1)."""
    return np.cos(eta) * I4 + 1j * np.sin(eta) * SWAP


# --- unitaries on three qubits -------------------------------------------------

def u1(eta):
    """Partial swap of E1 and S  (Eq. 3.2)."""
    return np.kron(partial_swap_2q(eta), I2)


def u2(eta):
    """Partial swap of S and E2  (Eq. 3.3)."""
    return np.kron(I2, partial_swap_2q(eta))


def u3(eta):
    """Partial swap of E1 and E2, an environment-environment interaction (Eq. 4.5)."""
    s23 = np.kron(I2, SWAP)
    return s23 @ u1(eta) @ s23


def swap_env():
    """Full swap S_E of the two environment qubits  (Eq. 4.4)."""
    s12 = np.kron(SWAP, I2)
    s23 = np.kron(I2, SWAP)
    return s12 @ s23 @ s12


# --- the seven models (A-form, 64 x 64) --------------------------------------------

def model0(eta=np.pi / 1000):
    """Original partial swap average map  (U1 + U2) / 2  (Eq. 3.12)."""
    return (a_form(u1(eta)) + a_form(u2(eta))) / 2


def model1(eta=np.pi / 1000):
    """(U1 + U2 + S_E) / 3  (Eq. 4.6)."""
    return (a_form(u1(eta)) + a_form(u2(eta)) + a_form(swap_env())) / 3


def model2(eta=np.pi / 1000):
    """((U1 + U2) / 2) . S_E  (Eq. 4.8): environment swap after each average step."""
    return model0(eta) @ a_form(swap_env())


def model3(eta=np.pi / 1000):
    """(U1 + U2 + U3) / 3, all with strength eta  (Eq. 4.10)."""
    return (a_form(u1(eta)) + a_form(u2(eta)) + a_form(u3(eta))) / 3


def model4(eta1=np.pi / 1000, eta2=np.pi / 10):
    """(U1(eta1) + U2(eta1) + U3(eta2)) / 3  (Eq. 4.12): stronger environment coupling."""
    return (a_form(u1(eta1)) + a_form(u2(eta1)) + a_form(u3(eta2))) / 3


def model5(eta1=np.pi / 1000, eta2=np.pi / 10):
    """((U1(eta1) + U2(eta1)) / 2) . U3(eta2)  (Eq. 4.14)."""
    return model0(eta1) @ a_form(u3(eta2))


def model6(eta=np.pi / 1000):
    """((U1 + U2 + U3) / 3) . S_E  (Eq. 4.16)."""
    return model3(eta) @ a_form(swap_env())


MODELS = {
    0: model0, 1: model1, 2: model2, 3: model3, 4: model4, 5: model5, 6: model6,
}
