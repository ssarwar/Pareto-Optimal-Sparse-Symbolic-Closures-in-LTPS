from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# -----------------------------------------------------------------------------
# This script will:
#   1) load the bulk dataset + scaling constants,
#   2) convert the normalized data back to physical units,
#   3) compute plasma and collision frequencies,
#   4) place those values on a 2D pressure x RF-frequency grid and plot them.
#
# -----------------------------------------------------------------------------


# ----------------------------- parameters -------------------------
GAS_TEMPERATURE_K = 300.0        # neutral argon gas temperature
NEUTRAL_DRIFT_SPEED_M_PER_S = 0.0  # assumed mean neutral flow speed in the bulk
RELATIVE_SPEED_GRID_POINTS = 600   # quadrature points for <Q_m(g) g>
RELATIVE_SPEED_GRID_WIDTH = 8.0    # integrate out to U + N*sigma_g
SAVE_PLOTS = True
SAVE_ARRAYS = True

# If this file lives in <repo>/physics/, then parents[1] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = REPO_ROOT / "data" / "rf" / "pic" / "ml_data" / "mean_bulk_ccp_dataset.npz"
SCALE_FILE = REPO_ROOT / "data" / "rf" / "pic" / "scaling_consts.npz"
OUTPUT_DIR = REPO_ROOT / "physics" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------- physical constants ---------------------------
q = 1.602176634e-19         # elementary charge [C]
eps0 = 8.8541878128e-12     # vacuum permittivity [F/m]
m_i = 6.634e-26             # Ar+ mass [kg]
m_n = 6.634e-26             # Ar mass [kg]
k_B = 1.380649e-23          # Boltzmann constant [J/K]
mTorr_to_Pa = 133.322e-3    # 1 mTorr in Pa
MHz_to_Hz = 1.0e6


# ------------------------------- helper functions -----------------------------
def phelps_qm_ar_plus_ar(energy_eV):
    """
    Ar+ + Ar momentum-transfer cross section from Phelps (1994), Eq. (2).

    energy_eV is the *equivalent stationary-target ion energy* in eV.
    For a given ion-neutral relative speed g, that equivalent energy is

        E_eq = 0.5 * m_i * g^2 / q.

    The output is in m^2.
    """
    energy_eV = np.clip(np.asarray(energy_eV, dtype=float), 1.0e-4, None)
    return 1.15e-18 * energy_eV ** (-0.1) * (1.0 + 0.015 / energy_eV) ** 0.6


def centers_to_edges(x):
    """
    Convert cell centers to cell edges so pcolormesh draws a proper heatmap.

    Works for both uniform and non-uniform grids.
    """
    x = np.asarray(x, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])

    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - 0.5 * (x[1] - x[0])
    last = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate([[first], mid, [last]])


def make_grid(pressure_mTorr, rf_MHz, values):
    """
    Put one scalar value from each simulation case onto a 2D grid.

    Grid shape is (n_rf, n_pressure), so rows follow RF frequency and columns
    follow pressure. If the same (pressure, rf) pair appears more than once,
    we average those entries.
    """
    pressure_mTorr = np.round(np.asarray(pressure_mTorr, dtype=float), 10)
    rf_MHz = np.round(np.asarray(rf_MHz, dtype=float), 10)
    values = np.asarray(values, dtype=float)

    P_unique = np.unique(pressure_mTorr)
    F_unique = np.unique(rf_MHz)

    p_to_j = {p: j for j, p in enumerate(P_unique)}
    f_to_i = {f: i for i, f in enumerate(F_unique)}

    grid_sum = np.zeros((len(F_unique), len(P_unique)), dtype=float)
    grid_count = np.zeros((len(F_unique), len(P_unique)), dtype=int)

    for p, f, v in zip(pressure_mTorr, rf_MHz, values):
        i = f_to_i[f]
        j = p_to_j[p]
        grid_sum[i, j] += v
        grid_count[i, j] += 1

    grid = np.full_like(grid_sum, np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return P_unique, F_unique, grid


def relative_speed_pdf(g, drift_speed, sigma_g_1d):
    """
    Speed PDF of the relative velocity magnitude g = |v_i - v_n|.

    Assumptions:
      - ions are Maxwellian with temperature T_i and drift U_i,
      - neutrals are Maxwellian with temperature T_n and drift U_n,
      - the relative drift is U = |U_i - U_n|,
      - each Cartesian component of the relative velocity is Gaussian with
        standard deviation sigma_g_1d.

    For U = 0, this reduces to the ordinary Maxwell speed distribution.
    For U > 0, this is the shifted-Maxwellian / noncentral-Maxwell speed PDF.
    """
    g = np.asarray(g, dtype=float)
    U = float(abs(drift_speed))
    s = float(max(sigma_g_1d, 1.0e-30))

    # Zero-drift limit: standard Maxwell speed distribution.
    if U < 1.0e-12 * max(s, 1.0):
        return np.sqrt(2.0 / np.pi) * (g ** 2 / s ** 3) * np.exp(-g ** 2 / (2.0 * s ** 2))

    # Stable finite-drift form. This avoids the sinh(x)/x cancellation when U
    # is very small and is better behaved numerically for larger U as well.
    term1 = np.exp(-(g - U) ** 2 / (2.0 * s ** 2))
    term2 = np.exp(-(g + U) ** 2 / (2.0 * s ** 2))
    return g * (term1 - term2) / (np.sqrt(2.0 * np.pi) * s * U)


def average_sigma_g_for_argon(
    ion_drift_speed_m_per_s,
    ion_temperature_eV,
    neutral_temperature_K,
    neutral_drift_speed_m_per_s=0.0,
    n_points=600,
    width_sigma=8.0,
):
    """
    Compute the quantity <Q_m(g) g> for Ar+ + Ar.

    This is the correct quantity that appears in the momentum-transfer collision
    frequency:

        nu_in = n_n * <Q_m(g) g>

    where g = |v_i - v_n| is the ion-neutral relative speed.

    Why this is better than the older formula:
      - The older script used one characteristic ion speed and assumed the
        neutrals were stationary.
      - Here we explicitly include the neutral thermal motion.
      - This matters most when the ion drift is small and/or the ion
        temperature is low, because then the neutral thermal spread can be a
        large fraction of the total relative speed.

    Returns three scalars:
      mean_sigma_g : <Q_m(g) g>  [m^3/s]
      mean_g       : <g>         [m/s]
      qm_effective : <Q_m g>/<g> [m^2]
    """
    # Relative drift speed between the ion fluid and the neutral gas.
    U_rel = abs(float(ion_drift_speed_m_per_s) - float(neutral_drift_speed_m_per_s))

    # 1D thermal spread of the relative velocity distribution.
    # For a Maxwellian species, each velocity component has variance kT/m.
    # Here T_i is given in eV, so kT_i = q * T_i_eV.
    sigma_g_1d = np.sqrt(
        q * max(float(ion_temperature_eV), 0.0) / m_i
        + k_B * float(neutral_temperature_K) / m_n
    )

    # Integrate over relative speed. A finite interval U + N*sigma captures the
    # tail well for the present purpose.
    g_max = max(U_rel + width_sigma * sigma_g_1d, 10.0)
    g = np.linspace(0.0, g_max, int(n_points))

    pdf = relative_speed_pdf(g, U_rel, sigma_g_1d)
    norm = np.trapz(pdf, g)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Relative-speed PDF normalization failed.")
    pdf = pdf / norm

    # Evaluate the Phelps cross section at the equivalent stationary-target ion
    # energy associated with the same relative speed g.
    energy_equiv_eV = 0.5 * m_i * g ** 2 / q
    q_m = phelps_qm_ar_plus_ar(energy_equiv_eV)

    mean_sigma_g = np.trapz(q_m * g * pdf, g)
    mean_g = np.trapz(g * pdf, g)
    qm_effective = mean_sigma_g / max(mean_g, 1.0e-30)
    return mean_sigma_g, mean_g, qm_effective


def plot_heatmap(P_unique, F_unique, grid_Hz, case_pressure_mTorr, case_rf_MHz, title, filename):
    """
    Plot a publication-style frequency heatmap in MHz and overlay the
    simulation points.

    Design choices:
      - use a perceptually uniform colormap,
      - keep the colorbar compact,
      - use inward ticks on all four sides,
      - make the simulation points easy to see in print,
      - optionally add light contour lines to help the eye follow gradients.
    """
    x_edges = centers_to_edges(P_unique)
    y_edges = centers_to_edges(F_unique)

    Z = np.asarray(grid_Hz, dtype=float)
    Z_masked = np.ma.masked_invalid(Z)

    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError("grid_Hz does not contain any finite values to plot.")

    # Guard against the degenerate case where the whole map is a single value.
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        pad = 0.05 * abs(vmin) if vmin != 0.0 else 1.0
        vmin -= pad
        vmax += pad

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)

    pcm = ax.pcolormesh(
        x_edges,
        y_edges,
        Z_masked,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )

    # Overlay the actual simulated cases. White-filled circles with black
    # outlines stay visible on both dark and light parts of the colormap.
    ax.scatter(
        case_pressure_mTorr,
        case_rf_MHz,
        s=28,
        facecolors="white",
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
    )

    ax.set_xlabel("Pressure [mTorr]", fontsize=15)
    ax.set_ylabel("Frequency [MHz]", fontsize=15)
    ax.set_title(title, fontsize=14, pad=8)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    # A slightly cleaner axis style
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=5, width=0.9, labelsize=12)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label("Frequency [MHz]", fontsize=13)
    cbar.ax.tick_params(direction="in", labelsize=11, length=4, width=0.8)
    cbar.locator = MaxNLocator(nbins=6)
    cbar.update_ticks()
    cbar.outline.set_linewidth(0.8)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


# ------------------------------- load the data --------------------------------
# The repo stores the mean bulk dataset under the key "profiles".
profiles = np.load(DATA_FILE)["profiles"]
scales = np.load(SCALE_FILE)

# These scaling constants convert the normalized dataset back to physical units.
flux0 = float(scales["flux0"])
ni0 = float(scales["ni0"])
Ti0 = float(scales["Ti0"])
P0 = float(scales["P0"])
F0 = float(scales["F0"])

# Channel indices follow src/utils/library.py.
#   0 = ion flux
#   1 = ion density
#   6 = ion temperature
#   9 = pressure
#  10 = RF driving frequency
Gamma_i = profiles[..., 0] * flux0          # ion flux [m^-2 s^-1]
n_i = profiles[..., 1] * ni0                # ion density [m^-3]
T_i_eV = profiles[..., 6] * Ti0             # ion temperature [eV]
pressure_mTorr = profiles[..., 9] * P0      # gas pressure [mTorr]
rf_MHz = profiles[..., 10] * F0             # RF frequency [MHz]


# -------------------------- reduce each case to one number --------------------
# Each simulation case still contains many bulk points.
# For the maps, we collapse each case to a single mean value.
Gamma_case = np.nanmean(np.abs(Gamma_i), axis=1)
n_i_case = np.nanmean(n_i, axis=1)
T_i_case_eV = np.nanmean(T_i_eV, axis=1)
pressure_case_mTorr = np.nanmean(pressure_mTorr, axis=1)
rf_case_MHz = np.nanmean(rf_MHz, axis=1)


# -------------------------- compute the three frequencies ----------------------
# 1) Ion plasma frequency.
plasma_frequency_case_rad = np.sqrt(n_i_case * q ** 2 / (eps0 * m_i))

# 2) RF driving frequency.
rf_frequency_case_rad = rf_case_MHz * MHz_to_Hz * 2.0 * np.pi

# 3) Ion-neutral collision frequency.
#
# We now compute nu_in from the relative ion-neutral motion instead of using
# one characteristic ion speed against a stationary background gas.
#
# The collision frequency is
#
#     nu_in = n_n * <Q_m(g) g>
#
# where g = |v_i - v_n| is the relative speed. The average is taken over a
# drifting ion Maxwellian and a thermal neutral Maxwellian.

n_i_safe = np.clip(n_i_case, 1.0e-30, None)
ion_drift_speed_case = Gamma_case / n_i_safe
n_neutral_case = pressure_case_mTorr * mTorr_to_Pa / (k_B * GAS_TEMPERATURE_K)

mean_relative_speed_case = np.zeros_like(ion_drift_speed_case)
effective_qm_case = np.zeros_like(ion_drift_speed_case)
ion_neutral_collision_frequency_case_Hz = np.zeros_like(ion_drift_speed_case)

for k in range(len(ion_drift_speed_case)):
    mean_sigma_g, mean_g, qm_eff = average_sigma_g_for_argon(
        ion_drift_speed_m_per_s=ion_drift_speed_case[k],
        ion_temperature_eV=T_i_case_eV[k],
        neutral_temperature_K=GAS_TEMPERATURE_K,
        neutral_drift_speed_m_per_s=NEUTRAL_DRIFT_SPEED_M_PER_S,
        n_points=RELATIVE_SPEED_GRID_POINTS,
        width_sigma=RELATIVE_SPEED_GRID_WIDTH,
    )
    mean_relative_speed_case[k] = mean_g
    effective_qm_case[k] = qm_eff
    ion_neutral_collision_frequency_case_Hz[k] = n_neutral_case[k] * mean_sigma_g


# ----------------------------- place everything on grids ----------------------
P_unique, F_unique, plasma_frequency_rad = make_grid(
    pressure_case_mTorr, rf_case_MHz, plasma_frequency_case_rad
)
_, _, rf_frequency_rad = make_grid(
    pressure_case_mTorr, rf_case_MHz, rf_frequency_case_rad
)
_, _, ion_neutral_collision_frequency_Hz = make_grid(
    pressure_case_mTorr, rf_case_MHz, ion_neutral_collision_frequency_case_Hz
)

# Make ratio maps
plasma_over_rf = plasma_frequency_rad / rf_frequency_rad
collision_over_rf = ion_neutral_collision_frequency_Hz / rf_frequency_rad

rf_over_plasma = rf_frequency_rad / plasma_frequency_rad
plasma_over_collision = plasma_frequency_rad / ion_neutral_collision_frequency_Hz
rf_over_collisionality = rf_frequency_rad / ((plasma_frequency_rad**2) / ion_neutral_collision_frequency_Hz)
# ------------------------------- save the arrays -------------------------------
if SAVE_ARRAYS:
    np.savez(
        OUTPUT_DIR / "frequency_grids.npz",
        pressure_mTorr=P_unique,
        rf_MHz=F_unique,
        plasma_frequency_Hz=plasma_frequency_rad,
        rf_frequency_Hz=rf_frequency_rad,
        ion_neutral_collision_frequency_Hz=ion_neutral_collision_frequency_Hz,
        plasma_over_rf=plasma_over_rf,
        collision_over_rf=collision_over_rf,
        pressure_case_mTorr=pressure_case_mTorr,
        rf_case_MHz=rf_case_MHz,
        plasma_frequency_case_Hz=plasma_frequency_case_rad,
        rf_frequency_case_Hz=rf_frequency_case_rad,
        ion_neutral_collision_frequency_case_Hz=ion_neutral_collision_frequency_case_Hz,
        mean_relative_speed_case_m_per_s=mean_relative_speed_case,
        effective_momentum_transfer_cross_section_case_m2=effective_qm_case,
    )


# -------------------------------- make the plots ------------------------------
if SAVE_PLOTS:
    plot_heatmap(
        P_unique,
        F_unique,
        plasma_frequency_rad,
        pressure_case_mTorr,
        rf_case_MHz,
        "Ion plasma frequency",
        OUTPUT_DIR / "plasma_frequency_heatmap.png",
    )

    plot_heatmap(
        P_unique,
        F_unique,
        rf_frequency_rad,
        pressure_case_mTorr,
        rf_case_MHz,
        "RF driving frequency",
        OUTPUT_DIR / "rf_frequency_heatmap.png",
    )

    plot_heatmap(
        P_unique,
        F_unique,
        ion_neutral_collision_frequency_Hz,
        pressure_case_mTorr,
        rf_case_MHz,
        "Ion-neutral collision frequency",
        OUTPUT_DIR / "ion_neutral_collision_heatmap.png",
    )

    plot_heatmap(
        P_unique,
        F_unique,
        rf_over_plasma,
        pressure_case_mTorr,
        rf_case_MHz,
        "RF/plasma frequency ratio",
        OUTPUT_DIR / "rf_over_plasma_heatmap.png",
    )

    plot_heatmap(
        P_unique,
        F_unique,
        plasma_over_collision,
        pressure_case_mTorr,
        rf_case_MHz,
        "Plasma/collision frequency ratio",
        OUTPUT_DIR / "plasma_over_collision_heatmap.png",
    )

    plot_heatmap(
        P_unique,
        F_unique,
        rf_over_collisionality,
        pressure_case_mTorr,
        rf_case_MHz,
        "RF/collision frequency ratio",
        OUTPUT_DIR / "rf_over_collisionality_heatmap.png",
    )

# ------------------------------- small text summary ---------------------------
print("Saved outputs to:", OUTPUT_DIR)
print()
print("Grid shapes:")
print("  plasma_frequency_Hz                ", plasma_frequency_rad.shape)
print("  rf_frequency_Hz                    ", rf_frequency_rad.shape)
print("  ion_neutral_collision_frequency_Hz ", ion_neutral_collision_frequency_Hz.shape)
print()
print("Typical ranges:")
print("  f_pi  [MHz] :", np.nanmin(plasma_frequency_rad) / 1e6, "to", np.nanmax(plasma_frequency_rad) / 1e6)
print("  f_rf  [MHz] :", np.nanmin(rf_frequency_rad) / 1e6, "to", np.nanmax(rf_frequency_rad) / 1e6)
print("  nu_in [MHz] :", np.nanmin(ion_neutral_collision_frequency_Hz) / 1e6, "to", np.nanmax(ion_neutral_collision_frequency_Hz) / 1e6)
