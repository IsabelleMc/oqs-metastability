"""Numbers quoted in the thesis, used to check the code (see tests/ and notebooks/02)."""

# model -> (description, distinct |Re lambda_L| values quoted, gap ratio quoted, equation)
THESIS_MODELS = {
    0: ("(U1 + U2) / 2", [4.9348e-6, 1.2337e-5, 1.4804e-5], 0.4, "4.3"),
    1: ("(U1 + U2 + S_E) / 3", [9.86960e-6, 6.66669e-1], 1.5e-5, "4.7"),
    2: ("(U1 + U2) / 2 . S_E", [1.48044e-5, 1.999987], 7.4e-6, "4.9"),
    3: ("(U1 + U2 + U3) / 3", [9.86960e-6, 1.97391e-5], 0.5, "4.11"),
    4: ("(U1 + U2 + U3(eta2)) / 3", [9.86960e-6, 6.36643e-2], 1.6e-4, "4.13"),
    5: ("(U1 + U2) / 2 . U3(eta2)", [1.48044e-5, 1.89146e-1], 7.8e-5, "4.15"),
    6: ("(U1 + U2 + U3) / 3 . S_E", [9.86960e-6, 1.99998], 4.9e-6, "4.17"),
}

# Section 4.3: perturbed model 0, c = 1e-8, eta = pi/100
THESIS_PERTURBATION = {"lambda_m": 1.0500e-8, "lambda_m1": 4.9333e-4}
