"""Memory in the partial swap models: mutual information of the two-time Choi state (Chapter 5).

Following Eq. A.1 and Figure 5.1 of the thesis, the two-time Choi state is built from the
dynamical map applied for ``t1`` steps and then for ``t2`` steps:

* the environments start in |0><0|;
* the system legs of each block (input and output) are left open, giving a 4-qubit Choi
  state on (in_1, out_1, in_2, out_2);
* the environment output of block 1 is the environment input of block 2;
* the environment output of block 2 is traced out.

The mutual information is

    I(t1 : t2) = S(Upsilon_t1) + S(Upsilon_t2) - S(Upsilon_t1t2)         (Eq. 5.1)

with S the von Neumann entropy (natural log). Throughout, Delta t = t2 - t1.
"""

import numpy as np


def choi_two_time(map_t1, map_t2):
    """Trace-normalised two-time Choi state, a 16 x 16 matrix on (in_1, out_1, in_2, out_2).

    ``map_t1`` and ``map_t2`` are 64 x 64 A-form dynamical maps in qubit order [E1, S, E2].
    """
    # Each map as a tensor with legs  out: (e1, s, e2, e1*, s*, e2*)   in: (e1, s, e2, e1*, s*, e2*)
    a = map_t1.reshape((2,) * 12)
    b = map_t2.reshape((2,) * 12)
    # Block 1: environment inputs fixed to |0><0|; system in-legs (s, S) stay open.
    a = a[:, :, :, :, :, :, 0, :, 0, 0, :, 0]  # legs: a b c A B C s S
    # Contract block-1 environment outputs (a, c, A, C) into block-2 environment inputs and
    # trace block-2 environment outputs (x, y repeated). Output: ket (s,b,v,t) | bra (S,B,w,u).
    upsilon = np.einsum("abcABCsS,xtyxuyavcAwC->sbvtSBwu", a, b)
    upsilon = upsilon.reshape(16, 16)
    return upsilon / np.trace(upsilon)


def von_neumann_entropy(rho):
    """S(rho) = -sum_i p_i log p_i for a density matrix (natural log, 0 log 0 = 0)."""
    p = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def mutual_information_from_maps(map_t1, map_t2):
    """I(t1 : t2) for the two blocks given as 64 x 64 maps  (Eq. 5.1)."""
    ups = choi_two_time(map_t1, map_t2)
    t = ups.reshape((2,) * 8)  # ket (in1, out1, in2, out2) | bra (same)
    ups_1 = np.einsum("sbvtSBvt->sbSB", t).reshape(4, 4)  # trace out block 2
    ups_2 = np.einsum("sbvtsbwu->vtwu", t).reshape(4, 4)  # trace out block 1
    return von_neumann_entropy(ups_1) + von_neumann_entropy(ups_2) - von_neumann_entropy(ups)


def mutual_information(dynamical_map, t1, t2):
    """I(t1 : t2) for the dynamical map applied t1 then t2 times."""
    power = np.linalg.matrix_power
    return mutual_information_from_maps(power(dynamical_map, t1), power(dynamical_map, t2))


def mutual_information_vs_gap(dynamical_map, t1, delta_ts):
    """I(t1 : t1 + Delta t) for a fixed t1 and a list of gaps (Figure 5.2)."""
    return np.array([mutual_information(dynamical_map, t1, t1 + dt) for dt in delta_ts])


def mutual_information_heatmap(dynamical_map, t1s, delta_ts):
    """Matrix of I(t1 : t1 + Delta t): rows are t1, columns are Delta t (Figures 5.3, 5.4)."""
    return np.array([mutual_information_vs_gap(dynamical_map, t1, delta_ts) for t1 in t1s])


def four_point_line(dynamical_map, delta_ts, alpha=1):
    """Mutual information along the line t1 = alpha * Delta t  (Eq. 5.2, Figures 5.5, 5.6)."""
    return np.array(
        [mutual_information(dynamical_map, int(alpha * dt), int(alpha * dt) + dt) for dt in delta_ts]
    )
