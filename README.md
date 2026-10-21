# Slowly Decaying Correlation in a Simple Metastable Quantum System

Code from my Honours thesis in Physics at Monash University (2021), supervised
by Kavan Modi.

## The problem in one paragraph

An open quantum system usually relaxes straight to its final state. Sometimes
it doesn't: it first gets stuck in a long-lived state that looks stationary but
isn't, and only much later decays to the true steady state. This is called
*metastability*. The theory says it should show up as a large gap in the
spectrum of the system's generator, a few slow decay rates separated from the
rest by several orders of magnitude. What is missing is a simple model where
we can see exactly how and why that gap appears.

## What this code does

Simulates the **partial swap model**: one system qubit collides repeatedly with
two environment qubits, each collision a partial swap of strength `eta`
(`eta = pi/2` is a full swap). It is the smallest non-trivial environment, so
every object is a small matrix that can be inspected directly. Maps are written
in the A-form `U (x) U*`, which acts on vectorised states.

1. **Spectral test.** Compute the 64 generator eigenvalues of the average
   collision map, and test for a gap with `lambda_m / lambda_{m+1} < 1e-3`.
   Compare the plain model (model 0) with six variants (models 1-6) that add
   interactions between the two environment qubits.

2. **Random-map perturbation.** Mix in a small random CPTP map,
   `G = (1 - c) * Lambda + c * R`, to break the degeneracies in the spectrum.

3. **Memory.** Build the two-time Choi state of the process and compute the
   mutual information between two times, then take a line through it to get a
   four-time correlation.

## What the thesis found

- The plain model shows no gap (`lambda_m / lambda_{m+1} = 0.4`).
- Adding interactions between the *environment* qubits opens one (ratios from
  `1.5e-5` down to `4.9e-6` in models 1, 2, 4, 5, 6), even though these
  interactions never touch the system qubit. Models 0, 2 and 5 have identical
  system dynamics, yet only 2 and 5 pass the spectral test.
- Perturbing the plain model with a random map of strength `c = 1e-8` exposes a
  gap of four orders of magnitude (`1.05e-8` to `4.93e-4`). So the plain model
  was metastable all along, with the gap hidden by large degeneracies.
- Mutual information between two times is non-zero, so the model has memory.
  Four-time correlations decay slowly towards a common steady state, which is
  proposed as an experimentally measurable signature of metastability in
  non-Markovian systems.

This code reproduces every eigenvalue and gap ratio quoted in the thesis, and
checks them in `tests/`.

## Why this generalises beyond physics

The core idea is: a system can look stationary for a very long time while a
slow mode is still decaying, and the *spectrum* of its transition map tells you
which modes are slow. The same principle governs Markov chains and sampling
algorithms. The spectral gap of a transition matrix sets how fast a chain
mixes, and a gap that appears after the first few eigenvalues means the chain
gets stuck in metastable regions before it explores the rest.

## Repo layout

```
src/
  states.py          # density matrices, vectorisation, random CPTP map
  models.py          # partial swap gates and the seven models (0-6)
  dynamics.py        # survival probability, random / ordered collision sequences
  metastability.py   # generator spectrum, gap ratio, random-map perturbation
  correlations.py    # two-time Choi state, mutual information, four-time lines
  reference.py       # numbers quoted in the thesis, used to check the code
notebooks/
  demo.ipynb         # model (Fig. 3.3), spectral test (Eqs. 4.3-4.17),
                     # hidden metastability (Fig. 4.10), memory (Figs. 5.3, 5.6)
tests/               # checks against the thesis values and basic physics
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/demo.ipynb
pytest
```

The demo notebook runs end to end in well under a minute on a laptop. The
mutual-information code also agrees with the original `ncon` implementation
from the thesis to about 1e-15.

## References

- Macieszczak et al., *Phys. Rev. Lett.* 116, 240404 (2016), spectral theory of metastability
- Macieszczak et al., *Phys. Rev. Research* 3, 033047 (2021), classical metastability in open quantum systems
- Benenti & Palma, *Phys. Rev. A* 75, 052110 (2007), qubit with a small environment
- Ziman et al., *Phys. Rev. A* 65, 042105 (2002), partial swap and homogenisation
- Milz & Modi, *PRX Quantum* 2, 030201 (2021), quantum stochastic processes and non-Markovianity
