# -----------------------------------------------------------------------------
# This script will:
#   1) load the normalizing/scaling constants from scaling_consts.npz
#   2) convert the normalized model coefficients term by term
#      back to physical units.

# Note that the model terms and coefficients are hardcoded from the results
# and tables presentated in the paper.

# The physicsal units are as follows:
#   - P is in mTorr
#   - F is in MHz
#   - Te, Ti are in eV
#   - E is in V/m
#   - n is in m^-3
#   - x is in m
#   - dn/dx is in m^-4
#   - Flux is in m^-2 s^-1
# -----------------------------------------------------------------------------

from pathlib import Path
import numpy as np
from typing import Dict

# If this file lives in <repo>/physics/, then parents[1] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "rf" / "pic" / "ml_data" / "mean_bulk_ccp_dataset.npz"
SCALE_FILE = REPO_ROOT / "data" / "rf" / "pic" / "scaling_consts.npz"
OUTPUT_DIR = REPO_ROOT / "physics" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------- load the data --------------------------------
# The repo stores the mean bulk dataset under the key "profiles".
profiles = np.load(DATA_FILE)["profiles"]
scales = np.load(SCALE_FILE)

# These scaling constants convert the normalized dataset back to physical units.
# src/nondimensionalizing_profiles.ipynb has details on how these are computed and saved.
ni0 = float(scales["ni0"])              # ion density in m^-3 (ni0 = ne0)
ne0 = float(scales["ne0"])              # electron density in m^-3 (ne0 = ni0)
Te0 = float(scales["Te0"])              # electron temperature in eV
Ti0 = float(scales["Ti0"])              # ion temperature in eV
E0 = float(scales["E0"])                # electric field in V/m
flux0 = float(scales["flux0"])          # ion flux in m^-2 s^-1
P0 = float(scales["P0"])                # gas pressure in mTorr
F0 = float(scales["F0"])                # RF driving frequency in MHz
L0 = 0.05                               # inter - electrode distance in m, used to normalize lengths 


# ---------------------------- compute physical model coefficients ---------------
# THIS PART IS HARDCODED TO THE MODELS PRESENTED IN THE PAPER.
# IF THE MODELS CHANGE, THIS PART MUST BE UPDATED MANUALLY.

# There are four models presented in the paper
# Model a has only n*E/P and n*E*P terms
# Model b has n*E/P, n*sqrt(E/P) and sqrt(Te*Ti)*dn/dx terms
# Model c has only n*E/P, n*sqrt(E/P), Te*dn/dx, and dn/dx terms
# Model d has only n*sqrt(E/P), dn/dx, sqrt(Ti)*dn/dx, and n*P*sqrt(E*Te) terms.

models = {
    'a': {'n*E/P': 0.110, 'n*E*P': 0.087},
    'b': {'n*E/P': 0.006, 'n*sqrt(E/P)': 0.122, 'sqrt(Te*Ti)*dn/dx': -0.030},
    'c': {'n*E/P': 0.031, 'n*sqrt(E/P)': 0.084, 'Te*dn/dx': -0.035, 'dn/dx': -0.001},
    'd': {'n*sqrt(E/P)': 0.111, 'dn/dx': -0.048, 'sqrt(Ti)*dn/dx': -0.001, 'n*P*sqrt(E*Te)': -0.016}
}

# ------------------------------------------------------------
# Denormalization multipliers
#   Gamma_hat = Gamma / G0
#   n_hat = n / ni0
#   P_hat = P / P0   (P written in mTorr)
#   E_hat = E / E0
#   Te_hat = Te / Te0
#   Ti_hat = Ti / Ti0
#   (dn/dx)_hat = (L0/ni0) (dn/dx)
# ------------------------------------------------------------
multipliers = {
    'n*E': flux0 / (ni0 * E0),
    'n*E/P': flux0 * P0 / (ni0 * E0),
    'n*E*P': flux0 / (ni0 * E0 * P0),
    'n*sqrt(E/P)': flux0 / ni0 * np.sqrt(P0 / E0),
    'sqrt(Te*Ti)*dn/dx': flux0 * L0 / (ni0 * np.sqrt(Te0 * Ti0)),
    'Te*dn/dx': flux0 * L0 / (ni0 * Te0),
    'dn/dx': flux0 * L0 / ni0,
    'E/P dn/dx': flux0 * L0 * P0 / (ni0 * E0),
    'sqrt(Ti)*dn/dx': flux0 * L0 / (ni0 * np.sqrt(Ti0)),
    'n*P*sqrt(E*Te)': flux0 / (ni0 * P0 * np.sqrt(E0 * Te0)),
}

phys_models: Dict[str, Dict[str, float]] = {
    m: {term: coeff * multipliers[term] for term, coeff in terms.items() if abs(coeff) > 0.0}
    for m, terms in models.items()
}

def print_denormalized_models():
    print('Denormalized models (P in mTorr, E in V/m, T in eV, n in m^-3, dn/dx in m^-4)')
    print('--------------------------------------------------------------------------------')
    for m, d in phys_models.items():
        print(f'Model {m}: {d}')
    print()


print_denormalized_models()
