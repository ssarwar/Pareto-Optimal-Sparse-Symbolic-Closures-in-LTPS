"""
Compute and plot K_omega for argon CCP PIC cases.

For each case, the plotted quantity is

    K_omega = sqrt[(wtilde2 - omega_F^2)^2 + nutilde^2 omega_F^2]
              / { wtilde2 sqrt[1 + omega_F^2 (G1/G0)^2] }

so that the cycle-averaging criterion is K_omega >> 1.

Definitions used here, with q = k * lambda_D supplied by the user:

    omega_F = 2 pi F,  where F is the case RF frequency in Hz.

    nutilde = <nu_i> + <nu_e> q^2/(1 + q^2)

    Omega_i^2 = omega_pi^2 q^2/(1 + q^2), because c_i = 0.

    wtilde2 = Omega_i^2 - (<nu_i> + <nu_e>) <nu_e>/(1 + q^2)

    G0 = (<nu_i> + <nu_e>) <nu_e>/(1 + q^2)
         - omega_pi^2 q^2/(1 + q^2)

    G1 = <nu_e>/(1 + q^2)

Here, tilde and check are intechangeable; tilde -> check.
See the original paper for any notational confusion.
The angle brackets denote spatial averages over [x_m - L_b, x_m + L_b].
The default x_m is 2.5 cm and the default L_b is 1.0 cm.

The angular ion plasma frequency is computed for each case from the averaged
ion density:

    omega_pi = sqrt(n_i_avg e^2 / (epsilon_0 m_Ar))

assuming singly charged argon ions with mass 39.948 u.  The script only uses
the user-supplied q = k lambda_D in the equations above; it does not compute
lambda_D or k separately.

Required files:

    ccp_nu_e_ion.pkl
    ccp_nu_i_cx.pkl
    ccp_nu_i_iso.pkl
    ccp_ni.pkl

By default, ccp_nu_* values are treated as volumetric rates [m^-3 s^-1].
Per-particle frequencies are computed by dividing by density:

    nu_e(x) = S_e_ion(x) / n_e(x)
    nu_i_cx(x)  = S_i_cx(x)  / n_i(x)
    nu_i_iso(x) = (m_g / (m_i + m_g)) * S_i_iso(x) / n_i(x)
    nu_i(x) =  nu_i_cx(x) + nu_i_iso(x)

IMPORTANT:
For this paper, we need to apply a mass factor of m_g / (m_i + m_g) to the elastic 
scattering (isotropic/iso) collision frequency to get the drag term in the
ion momentum equation to have the form R = - n_i m_i nu_i (v_rel).
Here, m_g is the neutral gas mass and m_i is the ion mass.
Since we are considering Ar+-Ar colliions, we hardcode to 0.5.
Note that this does not apply to the charge-exchange "switch" model.
THIS IS AROUND LINE 820.

The script first tries to read electron and ion density profiles from ccp_ne.pkl
and ccp_ni.pkl. If ccp_ne.pkl is absent, n_e is derived by integrating
ccp_evdf_2d.pkl(.zst) over velocity. If ccp_ni.pkl is absent, n_i is derived
by integrating ccp_ivdf_2d.pkl(.zst) over velocity, although ccp_ni.pkl is
preferred and expected for this diagnostic.

If the ccp_nu_* files already contain per-particle frequencies [s^-1], run with
--rate-files-are-frequencies. The ion density is still needed for omega_pi.

The CSV keeps all processed cases for auditability. The heatmap and scatter
points exclude every case whose final K_omega ratio is non-finite,
including NaN ratios.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - keep a fallback if scipy is unavailable.
    gaussian_filter1d = None


# SI constants used for the angular ion plasma frequency.
ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_0_F_PER_M = 8.8541878128e-12
ATOMIC_MASS_KG = 1.66053906660e-27
ARGON_ION_MASS_KG = 39.948 * ATOMIC_MASS_KG


@dataclass(frozen=True)
class ProfileCase:
    rf_MHz: float
    pressure_mTorr: float
    x_m: np.ndarray
    values: np.ndarray


def repo_root_from_this_file() -> Path:
    """Return the repository root when this script lives in /physics."""
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    """Use the repo ml_data directory; fall back to this script's directory for tests."""
    script_dir = Path(__file__).resolve().parent
    repo_data = repo_root_from_this_file() / "data" / "rf" / "pic" / "ml_data"
    if repo_data.exists():
        return repo_data
    if (script_dir / "ccp_nu_e_ion.pkl").exists():
        return script_dir
    return repo_data


def existing_data_file(data_dir: Path, stem: str) -> Path:
    """Find stem.pkl or stem.pkl.zst in data_dir."""
    candidates = [data_dir / f"{stem}.pkl", data_dir / f"{stem}.pkl.zst"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {stem}.pkl or {stem}.pkl.zst in {data_dir}."
    )


def has_data_file(data_dir: Path, stem: str) -> bool:
    try:
        existing_data_file(data_dir, stem)
        return True
    except FileNotFoundError:
        return False


@contextmanager
def pickle_stream(path: Path) -> Iterator[object]:
    """Yield a readable binary stream for a plain or zstd-compressed pickle."""
    if path.suffix != ".zst":
        with path.open("rb") as fh:
            yield fh
        return

    try:
        import zstandard as zstd  # type: ignore
    except Exception:
        zstd = None

    if zstd is not None:
        with path.open("rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            try:
                yield reader
            finally:
                reader.close()
        return

    zstd_exe = shutil.which("zstd")
    if zstd_exe is None:
        raise ImportError(
            f"{path} is zstd-compressed, but neither the Python package "
            "'zstandard' nor the 'zstd' command-line tool was found."
        )

    proc = subprocess.Popen(
        [zstd_exe, "-dc", "--long=31", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    try:
        yield proc.stdout
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"zstd failed while reading {path}: {stderr.strip()}")


def load_pickle(path: Path):
    with pickle_stream(path) as fh:
        return pickle.load(fh)


def finite_nanmax(a: np.ndarray) -> float:
    """Return nanmax without warnings when all values are non-finite."""
    arr = np.asarray(a, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.max(finite))


def trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Use numpy.trapezoid when available; fall back to numpy.trapz."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def case_key(rf_MHz: float, pressure_mTorr: float, ndigits: int = 10) -> Tuple[float, float]:
    """Stable key for matching cases across different pkl files."""
    return (round(float(rf_MHz), ndigits), round(float(pressure_mTorr), ndigits))


def load_1d_dataset(data_dir: Path, stem: str) -> Dict[Tuple[float, float], ProfileCase]:
    """Load one 1D CCP pkl dataset keyed by (rf_MHz, pressure_mTorr)."""
    path = existing_data_file(data_dir, stem)
    data_all = load_pickle(path)

    cases: Dict[Tuple[float, float], ProfileCase] = {}
    for idx, rec in enumerate(data_all):
        if len(rec) != 4:
            raise ValueError(
                f"{path.name} entry {idx} has length {len(rec)}; expected 4: "
                "[rf_MHz, pressure_mTorr, x_grid_m, values]."
            )
        rf_MHz, pressure_mTorr, x_m, values = rec
        x = np.asarray(x_m, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        if x.size != y.size:
            raise ValueError(
                f"{path.name} entry {idx} has x/value length mismatch: {x.size} vs {y.size}."
            )
        key = case_key(rf_MHz, pressure_mTorr)
        if key in cases:
            raise ValueError(
                f"Duplicate case in {path.name}: rf={rf_MHz}, pressure={pressure_mTorr}."
            )
        cases[key] = ProfileCase(float(rf_MHz), float(pressure_mTorr), x, y)

    if not cases:
        raise ValueError(f"{path.name} did not contain any cases.")
    return cases


def load_density_from_vdf_dataset(
    data_dir: Path,
    vdf_stem: str,
    species_label: str,
) -> Dict[Tuple[float, float], ProfileCase]:
    """Derive density profiles by integrating a 2D VDF pkl over velocity."""
    path = existing_data_file(data_dir, vdf_stem)
    print(f"Deriving {species_label} density from {path.name} by integrating over velocity.")
    data_all = load_pickle(path)

    cases: Dict[Tuple[float, float], ProfileCase] = {}
    skipped: list[tuple[int, float, float, tuple, tuple, tuple]] = []
    for idx, rec in enumerate(data_all):
        if len(rec) != 5:
            raise ValueError(
                f"{path.name} entry {idx} has length {len(rec)}; expected 5: "
                "[rf_MHz, pressure_mTorr, x_grid_m, v_grid, f_vx]."
            )
        rf_MHz, pressure_mTorr, x_m, v_grid, f_vx = rec
        x = np.asarray(x_m, dtype=float).reshape(-1)
        v = np.asarray(v_grid, dtype=float).reshape(-1)
        f = np.asarray(f_vx, dtype=float)
        if f.shape != (v.size, x.size) or x.size < 2 or v.size < 2:
            # Failed or placeholder entries are usually stored as shape (1,).
            # They are not useful for density reconstruction and are normally
            # absent from the collision-rate files.
            skipped.append((idx, float(rf_MHz), float(pressure_mTorr), x.shape, v.shape, f.shape))
            continue
        density = trapz(f, v, axis=0)
        key = case_key(rf_MHz, pressure_mTorr)
        if key in cases:
            raise ValueError(
                f"Duplicate case in {path.name}: rf={rf_MHz}, pressure={pressure_mTorr}."
            )
        cases[key] = ProfileCase(float(rf_MHz), float(pressure_mTorr), x, density)

    if skipped:
        shown = ", ".join(
            f"#{idx} ({rf:g} MHz, {p:g} mTorr, x={xs}, v={vs}, values={fs})"
            for idx, rf, p, xs, vs, fs in skipped[:5]
        )
        more = "" if len(skipped) <= 5 else f", ... +{len(skipped) - 5} more"
        print(f"Warning: skipped {len(skipped)} invalid/placeholder entries in {path.name}: {shown}{more}.")

    if not cases:
        raise ValueError(f"{path.name} did not contain any valid cases.")
    return cases


def load_density_dataset(
    data_dir: Path,
    density_stem: str,
    fallback_vdf_stem: str,
    species_label: str,
) -> Tuple[Dict[Tuple[float, float], ProfileCase], str]:
    """Load density from a 1D density pkl if available; otherwise derive from a VDF."""
    if has_data_file(data_dir, density_stem):
        return load_1d_dataset(data_dir, density_stem), density_stem
    return load_density_from_vdf_dataset(data_dir, fallback_vdf_stem, species_label), fallback_vdf_stem


def _moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware moving-average fallback if scipy is unavailable."""
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float)
    finite = np.isfinite(y)
    numerator = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    out = numerator / np.maximum(denominator, 1.0e-30)
    out[denominator <= 0] = np.nan
    return out


def nan_gaussian_smooth(y: np.ndarray, sigma_points: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs by smoothing numerator and weights."""
    y = np.asarray(y, dtype=float)
    if sigma_points <= 0.0 or y.size < 3:
        return y.copy()

    finite = np.isfinite(y)
    if finite.sum() == 0:
        return np.full_like(y, np.nan, dtype=float)

    if gaussian_filter1d is None:
        return _moving_average_nan(y, int(round(6.0 * sigma_points)))

    numerator = gaussian_filter1d(
        np.where(finite, y, 0.0), sigma=sigma_points, mode="nearest"
    )
    denominator = gaussian_filter1d(finite.astype(float), sigma=sigma_points, mode="nearest")
    out = numerator / np.maximum(denominator, 1.0e-30)
    out[denominator <= 1.0e-12] = np.nan
    return out


def smooth_on_x(
    x_m: np.ndarray,
    y: np.ndarray,
    smooth_sigma_m: float,
    clip_nonnegative: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort a profile by x, combine duplicate x's, and smooth over smooth_sigma_m."""
    x = np.asarray(x_m, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    if x.size != yy.size:
        raise ValueError(f"x/value length mismatch inside profile: {x.size} vs {yy.size}.")
    if x.size == 0:
        return x, yy

    order = np.argsort(x)
    x = x[order]
    yy = yy[order]

    x_unique, inverse = np.unique(x, return_inverse=True)
    if x_unique.size != x.size:
        yy_unique = np.empty_like(x_unique, dtype=float)
        for i in range(x_unique.size):
            yy_unique[i] = np.nanmean(yy[inverse == i])
        x, yy = x_unique, yy_unique

    if smooth_sigma_m > 0.0 and x.size > 2:
        dx = np.nanmedian(np.diff(x))
        if np.isfinite(dx) and dx > 0.0:
            yy = nan_gaussian_smooth(yy, smooth_sigma_m / dx)

    if clip_nonnegative:
        yy = np.where(np.isfinite(yy), np.maximum(yy, 0.0), np.nan)
    return x, yy


def interp_smoothed(
    case: ProfileCase,
    x_target_m: np.ndarray,
    smooth_sigma_m: float,
    clip_nonnegative: bool = False,
) -> np.ndarray:
    """Smooth a case on its own x grid, then interpolate to x_target_m."""
    x, y = smooth_on_x(case.x_m, case.values, smooth_sigma_m, clip_nonnegative)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(x_target_m, np.nan, dtype=float)
    return np.interp(x_target_m, x[finite], y[finite], left=np.nan, right=np.nan)


def spatial_average_interval(x_m: np.ndarray, y: np.ndarray, xmin_m: float, xmax_m: float) -> float:
    """Average y over the exact interval xmin_m <= x <= xmax_m using interpolation.

    Endpoints are included by linear interpolation so that the average is over the
    requested physical interval, not merely over whichever grid points happen to
    fall inside it.
    """
    if xmax_m <= xmin_m:
        raise ValueError(f"xmax_m must be greater than xmin_m, got {xmin_m} and {xmax_m}.")
    x = np.asarray(x_m, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(yy)
    if finite.sum() < 2:
        return np.nan
    xf = x[finite]
    yf = yy[finite]
    order = np.argsort(xf)
    xf = xf[order]
    yf = yf[order]
    if xmin_m < xf[0] or xmax_m > xf[-1]:
        return np.nan

    inside = (xf > xmin_m) & (xf < xmax_m)
    xs = np.concatenate([[xmin_m], xf[inside], [xmax_m]])
    ys = np.interp(xs, xf, yf)
    return float(trapz(ys, xs) / (xmax_m - xmin_m))


def ion_plasma_angular_frequency_s_inv(ni_m3: float) -> float:
    """Angular ion plasma frequency for singly charged argon in SI units."""
    if not np.isfinite(ni_m3) or ni_m3 <= 0.0:
        return np.nan
    omega2 = ni_m3 * ELEMENTARY_CHARGE_C**2 / (EPSILON_0_F_PER_M * ARGON_ION_MASS_KG)
    return float(np.sqrt(omega2)) if omega2 > 0.0 else np.nan


def compute_Komega_inverse_ratio(
    *,
    rf_MHz: float,
    nu_i_s_inv: float,
    nu_e_s_inv: float,
    ni_avg_m3: float,
    k_lambda_D: float,
    use_abs_wtilde2_in_prefactor: bool = False,
) -> dict[str, float]:
    """Compute Eq. (11), (13), (14), and K_omega quantities.

    The default follows the formula for K_omega literally, using wtilde2 in the prefactor.
    Because K_omega is a positive magnitude diagnostic, cases with a non-positive
    literal prefactor are returned as NaN. If use_abs_wtilde2_in_prefactor is
    true, abs(wtilde2) is used in that prefactor instead.
    """
    q = float(k_lambda_D)
    q2 = q * q
    one_plus_q2 = 1.0 + q2

    F_Hz = float(rf_MHz) * 1.0e6
    omega_F = 2.0 * np.pi * F_Hz
    omega_pi = ion_plasma_angular_frequency_s_inv(ni_avg_m3)
    omega_pi2 = omega_pi * omega_pi if np.isfinite(omega_pi) else np.nan

    if (
        not np.isfinite(nu_i_s_inv)
        or not np.isfinite(nu_e_s_inv)
        or not np.isfinite(omega_pi2)
        or not np.isfinite(omega_F)
        or not np.isfinite(q)
        or q < 0.0
    ):
        return {
            "F_Hz": F_Hz,
            "omega_F_rad_s": omega_F,
            "omega_pi_rad_s": omega_pi,
            "omega_pi2_rad2_s2": omega_pi2,
            "Omega_i2_rad2_s2": np.nan,
            "nutilde_s_inv": np.nan,
            "wtilde2_rad2_s2": np.nan,
            "G0_s^-2": np.nan,
            "G1_s^-1": np.nan,
            "Komega_numerator_s^-2": np.nan,
            "Komega_denominator_s^-2": np.nan,
            "Komega_direct_ratio": np.nan,
            "Komega_inverse_ratio": np.nan,
        }

    # Collisional damping frequency, with q = k lambda_D.
    nutilde = nu_i_s_inv + nu_e_s_inv * q2 / one_plus_q2

    # Ion acoustic frequency, with c_i = 0.
    Omega_i2 = omega_pi2 * q2 / one_plus_q2

    # Eq. (13), Eq. (14).
    ionization_product_term = (nu_i_s_inv + nu_e_s_inv) * nu_e_s_inv / one_plus_q2
    wtilde2 = Omega_i2 - ionization_product_term
    G0 = ionization_product_term - Omega_i2
    G1 = nu_e_s_inv / one_plus_q2

    Komega_denominator = np.sqrt((wtilde2 - omega_F * omega_F) ** 2 + (nutilde * omega_F) ** 2)

    if use_abs_wtilde2_in_prefactor:
        wtilde2_prefactor = abs(wtilde2)
    else:
        wtilde2_prefactor = wtilde2

    if (
        np.isfinite(G0)
        and G0 != 0.0
        and np.isfinite(wtilde2_prefactor)
        and wtilde2_prefactor > 0.0
    ):
        Komega_numerator = wtilde2_prefactor * np.sqrt(1.0 + (omega_F * G1 / G0) ** 2)
    else:
        Komega_numerator = np.nan

    if (
        np.isfinite(Komega_denominator)
        and np.isfinite(Komega_numerator)
        and Komega_numerator != 0.0
    ):
        Komega_direct = Komega_numerator / Komega_denominator
        Komega_inverse = Komega_denominator / Komega_numerator
    else:
        Komega_direct = np.nan
        Komega_inverse = np.nan

    return {
        "F_Hz": F_Hz,
        "omega_F_rad_s": float(omega_F),
        "omega_pi_rad_s": float(omega_pi),
        "omega_pi2_rad2_s2": float(omega_pi2),
        "Omega_i2_rad2_s2": float(Omega_i2),
        "nutilde_s_inv": float(nutilde),
        "wtilde2_rad2_s2": float(wtilde2),
        "G0_s^-2": float(G0),
        "G1_s^-1": float(G1),
        "Komega_numerator_s^-2": float(Komega_numerator),
        "Komega_denominator_s^-2": float(Komega_denominator),
        "Komega_direct_ratio": float(Komega_direct),
        "Komega_inverse_ratio": float(Komega_inverse),
    }


def centers_to_edges(x: Iterable[float]) -> np.ndarray:
    """Convert center coordinates to pcolormesh edges."""
    x = np.asarray(list(x), dtype=float)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    mid = 0.5 * (x[:-1] + x[1:])
    first = x[0] - 0.5 * (x[1] - x[0])
    last = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.concatenate([[first], mid, [last]])


def make_grid(
    pressure_mTorr: np.ndarray, rf_MHz: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average finite scalar case values onto a grid shaped (n_rf, n_pressure)."""
    pressure = np.round(np.asarray(pressure_mTorr, dtype=float), 10)
    rf = np.round(np.asarray(rf_MHz, dtype=float), 10)
    values = np.asarray(values, dtype=float)

    p_unique = np.unique(pressure)
    f_unique = np.unique(rf)
    p_to_j = {p: j for j, p in enumerate(p_unique)}
    f_to_i = {f: i for i, f in enumerate(f_unique)}

    grid_sum = np.zeros((f_unique.size, p_unique.size), dtype=float)
    grid_count = np.zeros_like(grid_sum, dtype=int)
    for p, f, v in zip(pressure, rf, values):
        if not np.isfinite(v):
            continue
        grid_sum[f_to_i[f], p_to_j[p]] += v
        grid_count[f_to_i[f], p_to_j[p]] += 1

    grid = np.full_like(grid_sum, np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return p_unique, f_unique, grid


def plot_ratio_heatmap(
    p_unique: np.ndarray,
    f_unique: np.ndarray,
    ratio_grid: np.ndarray,
    pressure_cases_plotted: np.ndarray,
    rf_cases_plotted: np.ndarray,
    output_svg: Path,
    title: str,
) -> None:
    """Plot inverse Eq. (15) ratio heatmap and overlay finite-ratio cases."""
    p_edges = centers_to_edges(p_unique)
    f_edges = centers_to_edges(f_unique)
    Z = np.asarray(ratio_grid, dtype=float)
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError("The inverse Eq. (15) ratio grid contains no finite values to plot.")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    print(vmin)

    # WARNING
    # Changing vmin and vmax for plot scaling purposes
    vmin = max(math.ceil(vmin), 1.0)
    vmax = min(vmax, 100.0)
    # END WARNING

    plt.style.use(["science", "nature", "no-latex"])

    if np.isclose(vmin, vmax):
        pad = 0.05 * abs(vmin) if vmin != 0.0 else 1.0
        vmin -= pad
        vmax += pad

    fig, ax = plt.subplots(figsize=(6.5, 5.1), constrained_layout=True)
    pcm = ax.pcolormesh(
        p_edges,
        f_edges,
        np.ma.masked_invalid(Z),
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    ax.scatter(
        pressure_cases_plotted,
        rf_cases_plotted,
        s=32,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.0,
        alpha=0.65,
        zorder=3,
        label="PIC cases",
    )

    ax.set_xlabel("Pressure [mTorr]", fontsize=12)
    ax.set_ylabel("Frequency [MHz]", fontsize=12)
    #ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlim(p_edges[0], p_edges[-1])
    ax.set_ylim(f_edges[0], f_edges[-1])
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=5, width=0.9, labelsize=11)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    #cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.055)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\left| \tilde{N}_{i1}(k, \omega \rightarrow 0)\right|  / \left| \tilde{N}_{i1}(k,\omega_F) \right|$", fontsize=12)

    # Set ticks manually to ensure a reasonable number of ticks even if the range is small or large.
    num_ticks = 5
    ticks = np.linspace(vmin, 95, num_ticks)
    cbar.set_ticks(ticks)
    print(cbar.get_ticks())
    # END Set ticks manually.

    cbar.ax.tick_params(direction="in", labelsize=11, length=4, width=0.8)
    # cbar.locator = MaxNLocator(nbins=10)
    # cbar.update_ticks()
    cbar.outline.set_linewidth(0.8)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(fig)


def write_case_summary_csv(output_csv: Path, rows: list[dict]) -> None:
    """Write one scalar summary row per case."""
    if not rows:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D heatmap of the inverse Eq. (15) ratio for CCP PIC cases."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Directory containing ccp_*.pkl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root_from_this_file() / "physics" / "output_frequency_condition",
        help="Directory for the SVG, CSV, and saved arrays.",
    )
    parser.add_argument(
        "--x-midpoint-m",
        type=float,
        default=0.025,
        help="Midpoint x_m [m]. Default: 0.025 m = 2.5 cm.",
    )
    parser.add_argument(
        "--L-b-m",
        type=float,
        default=0.010,
        help="Distance L_b from x_m [m]. Default: 0.010 m = 1 cm.",
    )
    parser.add_argument(
        "--k-lambda-D",
        type=float,
        default= 2 * np.pi,
        help="User-supplied q = k*lambda_D used in Eqs. (11), (13), and (14). Default: 2pi.",
    )
    parser.add_argument(
        "--smooth-sigma-m",
        type=float,
        default=1e-3,
        help="Gaussian smoothing length for collision/density profiles [m]. Use 0 to disable. Default: 1.0 mm.",
    )
    parser.add_argument(
        "--rate-files-are-frequencies",
        action="store_true",
        help="Use ccp_nu_* values directly as s^-1 instead of dividing by densities.",
    )
    parser.add_argument(
        "--density-floor-rel",
        type=float,
        default=1.0e-8,
        help="Relative density floor used when dividing volumetric rates by density.",
    )
    parser.add_argument(
        "--abs-wtilde2-prefactor",
        action="store_true",
        help=(
            "Use abs(wtilde2) in the Eq. (15) numerator prefactor. By default the "
            "literal wtilde2 is used, so cases with non-positive wtilde2 are treated as NaN."
        ),
    )
    parser.add_argument(
        "--fig-name",
        default="Komega_inverse_ratio_heatmap.svg",
        help="SVG filename written inside --out-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.L_b_m <= 0.0:
        raise ValueError("--L-b-m must be positive.")
    if args.k_lambda_D < 0.0 or not np.isfinite(args.k_lambda_D):
        raise ValueError("--k-lambda-D must be a finite nonnegative number.")
    if args.density_floor_rel < 0.0:
        raise ValueError("--density-floor-rel must be nonnegative.")

    x_left_m = args.x_midpoint_m - args.L_b_m
    x_right_m = args.x_midpoint_m + args.L_b_m
    if x_left_m < 0.0:
        raise ValueError(
            f"x_m - L_b is negative ({x_left_m:g} m). Check --x-midpoint-m and --L-b-m."
        )

    density_source_e = "not used"
    datasets: Dict[str, Dict[Tuple[float, float], ProfileCase]] = {
        "e_ion": load_1d_dataset(args.data_dir, "ccp_nu_e_ion"),
        "i_cx": load_1d_dataset(args.data_dir, "ccp_nu_i_cx"),
        "i_iso": load_1d_dataset(args.data_dir, "ccp_nu_i_iso"),
    }

    datasets["ni"], density_source_i = load_density_dataset(
        args.data_dir, "ccp_ni", "ccp_ivdf_2d", "ion"
    )
    if not args.rate_files_are_frequencies:
        datasets["ne"], density_source_e = load_density_dataset(
            args.data_dir, "ccp_ne", "ccp_evdf_2d", "electron"
        )

    common_keys = set(datasets["e_ion"].keys())
    for ds in datasets.values():
        common_keys &= set(ds.keys())
    if not common_keys:
        raise ValueError("No common (rf_MHz, pressure_mTorr) cases found across the pkl files.")

    for name, ds in datasets.items():
        missing = sorted(set(datasets["e_ion"].keys()) - set(ds.keys()))
        if missing:
            print(f"Warning: {name} is missing {len(missing)} cases found in ccp_nu_e_ion.pkl.")

    rows: list[dict] = []
    pressure_cases: list[float] = []
    rf_cases: list[float] = []
    ratio_cases: list[float] = []
    Komega_direct_cases: list[float] = []
    nu_e_cases: list[float] = []
    nu_i_cases: list[float] = []
    ni_avg_cases: list[float] = []
    omega_F_cases: list[float] = []
    omega_pi_cases: list[float] = []
    omega_pi2_cases: list[float] = []
    Omega_i2_cases: list[float] = []
    nutilde_cases: list[float] = []
    wtilde2_cases: list[float] = []
    G0_cases: list[float] = []
    G1_cases: list[float] = []
    numerator_cases: list[float] = []
    denominator_cases: list[float] = []

    sorted_keys = sorted(common_keys, key=lambda k: (k[1], k[0]))

    for key in sorted_keys:
        rf_MHz, pressure_mTorr = key
        e_case = datasets["e_ion"][key]

        # Use the electron-ionization profile's x grid as the common grid.
        x_m, e_ion_rate = smooth_on_x(
            e_case.x_m,
            e_case.values,
            args.smooth_sigma_m,
            clip_nonnegative=True,
        )
        i_cx_rate = interp_smoothed(
            datasets["i_cx"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
        )
        i_iso_rate = interp_smoothed(
            datasets["i_iso"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
        )
        ni = interp_smoothed(
            datasets["ni"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
        )

        i_iso_rate = 0.5 * i_iso_rate       # hardcode mass factor (m_g / (m_i + m_g)) as explained above

        ni_max = finite_nanmax(ni)
        ni_floor = args.density_floor_rel * ni_max if np.isfinite(ni_max) else np.nan
        ni_safe = np.where(ni > ni_floor, ni, np.nan)

        if args.rate_files_are_frequencies:
            nu_e = e_ion_rate
            nu_i = i_cx_rate + i_iso_rate
        else:
            ne = interp_smoothed(
                datasets["ne"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
            )
            ne_max = finite_nanmax(ne)
            ne_floor = args.density_floor_rel * ne_max if np.isfinite(ne_max) else np.nan
            ne_safe = np.where(ne > ne_floor, ne, np.nan)
            nu_e = e_ion_rate / ne_safe
            nu_i = (i_cx_rate + i_iso_rate) / ni_safe

        nu_e_avg = spatial_average_interval(x_m, nu_e, x_left_m, x_right_m)
        nu_i_avg = spatial_average_interval(x_m, nu_i, x_left_m, x_right_m)
        ni_avg = spatial_average_interval(x_m, ni_safe, x_left_m, x_right_m)

        q = compute_Komega_inverse_ratio(
            rf_MHz=rf_MHz,
            nu_i_s_inv=nu_i_avg,
            nu_e_s_inv=nu_e_avg,
            ni_avg_m3=ni_avg,
            k_lambda_D=args.k_lambda_D,
            use_abs_wtilde2_in_prefactor=args.abs_wtilde2_prefactor,
        )
        ratio = q["Komega_inverse_ratio"]

        ratio_for_plot = ratio

        pressure_cases.append(pressure_mTorr)
        rf_cases.append(rf_MHz)
        ratio_cases.append(ratio_for_plot)
        Komega_direct_cases.append(q["Komega_direct_ratio"])
        nu_e_cases.append(nu_e_avg)
        nu_i_cases.append(nu_i_avg)
        ni_avg_cases.append(ni_avg)
        omega_F_cases.append(q["omega_F_rad_s"])
        omega_pi_cases.append(q["omega_pi_rad_s"])
        omega_pi2_cases.append(q["omega_pi2_rad2_s2"])
        Omega_i2_cases.append(q["Omega_i2_rad2_s2"])
        nutilde_cases.append(q["nutilde_s_inv"])
        wtilde2_cases.append(q["wtilde2_rad2_s2"])
        G0_cases.append(q["G0_s^-2"])
        G1_cases.append(q["G1_s^-1"])
        numerator_cases.append(q["Komega_numerator_s^-2"])
        denominator_cases.append(q["Komega_denominator_s^-2"])

        rows.append(
            {
                "rf_MHz": rf_MHz,
                "pressure_mTorr": pressure_mTorr,
                "x_midpoint_m": args.x_midpoint_m,
                "L_b_m": args.L_b_m,
                "x_average_min_m": x_left_m,
                "x_average_max_m": x_right_m,
                "k_lambda_D": args.k_lambda_D,
                "q2": args.k_lambda_D**2,
                "F_Hz": q["F_Hz"],
                "omega_F_rad_s": q["omega_F_rad_s"],
                "nu_e_avg_s^-1": nu_e_avg,
                "nu_i_avg_s^-1": nu_i_avg,
                "ni_avg_m^-3": ni_avg,
                "omega_pi_rad_s": q["omega_pi_rad_s"],
                "omega_pi2_rad2_s^-2": q["omega_pi2_rad2_s2"],
                "Omega_i2_rad2_s^-2": q["Omega_i2_rad2_s2"],
                "nutilde_s^-1": q["nutilde_s_inv"],
                "wtilde2_rad2_s^-2": q["wtilde2_rad2_s2"],
                "G0_s^-2": q["G0_s^-2"],
                "G1_s^-1": q["G1_s^-1"],
                "Komega_numerator_s^-2": q["Komega_numerator_s^-2"],
                "Komega_denominator_s^-2": q["Komega_denominator_s^-2"],
                "Komega_direct_ratio": q["Komega_direct_ratio"],
                "Komega_inverse_ratio_raw": ratio,
                "Komega_inverse_ratio_plotted_value": ratio_for_plot,
                "plotted": bool(np.isfinite(ratio_for_plot)),
            }
        )

    pressure_cases_arr = np.asarray(pressure_cases, dtype=float)
    rf_cases_arr = np.asarray(rf_cases, dtype=float)
    ratio_cases_arr = np.asarray(ratio_cases, dtype=float)
    Komega_direct_cases_arr = np.asarray(Komega_direct_cases, dtype=float)
    nu_e_cases_arr = np.asarray(nu_e_cases, dtype=float)
    nu_i_cases_arr = np.asarray(nu_i_cases, dtype=float)
    ni_avg_cases_arr = np.asarray(ni_avg_cases, dtype=float)
    omega_F_cases_arr = np.asarray(omega_F_cases, dtype=float)
    omega_pi_cases_arr = np.asarray(omega_pi_cases, dtype=float)
    omega_pi2_cases_arr = np.asarray(omega_pi2_cases, dtype=float)
    Omega_i2_cases_arr = np.asarray(Omega_i2_cases, dtype=float)
    nutilde_cases_arr = np.asarray(nutilde_cases, dtype=float)
    wtilde2_cases_arr = np.asarray(wtilde2_cases, dtype=float)
    G0_cases_arr = np.asarray(G0_cases, dtype=float)
    G1_cases_arr = np.asarray(G1_cases, dtype=float)
    numerator_cases_arr = np.asarray(numerator_cases, dtype=float)
    denominator_cases_arr = np.asarray(denominator_cases, dtype=float)
    plotted_case_mask = np.isfinite(ratio_cases_arr)

    p_unique, f_unique, ratio_grid = make_grid(pressure_cases_arr, rf_cases_arr, ratio_cases_arr)
    _, _, direct_grid = make_grid(pressure_cases_arr, rf_cases_arr, Komega_direct_cases_arr)
    _, _, nu_e_grid = make_grid(pressure_cases_arr, rf_cases_arr, nu_e_cases_arr)
    _, _, nu_i_grid = make_grid(pressure_cases_arr, rf_cases_arr, nu_i_cases_arr)
    _, _, ni_avg_grid = make_grid(pressure_cases_arr, rf_cases_arr, ni_avg_cases_arr)
    _, _, omega_pi_grid = make_grid(pressure_cases_arr, rf_cases_arr, omega_pi_cases_arr)
    _, _, wtilde2_grid = make_grid(pressure_cases_arr, rf_cases_arr, wtilde2_cases_arr)
    _, _, nuttilde_grid = make_grid(pressure_cases_arr, rf_cases_arr, nutilde_cases_arr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_svg = args.out_dir / args.fig_name
    title = (
        r"Argon CCP inverse Eq. (15) ratio "
        + rf"$k\lambda_D={args.k_lambda_D:g}$, "
        + rf"$x_m={args.x_midpoint_m*100:.1f}$ cm, $L_b={args.L_b_m*100:.1f}$ cm"
    )
    plot_ratio_heatmap(
        p_unique,
        f_unique,
        ratio_grid,
        pressure_cases_arr[plotted_case_mask],
        rf_cases_arr[plotted_case_mask],
        output_svg,
        title,
    )

    np.savez(
        args.out_dir / "Komega_inverse_ratio_grid.npz",
        pressure_mTorr=p_unique,
        rf_MHz=f_unique,
        Komega_inverse_ratio=ratio_grid,
        Komega_direct_ratio=direct_grid,
        nu_e_avg_s_inv=nu_e_grid,
        nu_i_avg_s_inv=nu_i_grid,
        ni_avg_m_inv3=ni_avg_grid,
        omega_pi_rad_s=omega_pi_grid,
        wtilde2_rad2_s_inv2=wtilde2_grid,
        nuttilde_s_inv=nuttilde_grid,
        pressure_case_mTorr=pressure_cases_arr,
        rf_case_MHz=rf_cases_arr,
        Komega_inverse_ratio_case=ratio_cases_arr,
        Komega_direct_ratio_case=Komega_direct_cases_arr,
        nu_e_avg_case_s_inv=nu_e_cases_arr,
        nu_i_avg_case_s_inv=nu_i_cases_arr,
        ni_avg_case_m_inv3=ni_avg_cases_arr,
        omega_F_case_rad_s=omega_F_cases_arr,
        omega_pi_case_rad_s=omega_pi_cases_arr,
        omega_pi2_case_rad2_s_inv2=omega_pi2_cases_arr,
        Omega_i2_case_rad2_s_inv2=Omega_i2_cases_arr,
        nuttilde_case_s_inv=nutilde_cases_arr,
        wtilde2_case_rad2_s_inv2=wtilde2_cases_arr,
        G0_case_s_inv2=G0_cases_arr,
        G1_case_s_inv=G1_cases_arr,
        Komega_numerator_case_s_inv2=numerator_cases_arr,
        Komega_denominator_case_s_inv2=denominator_cases_arr,
        plotted_case_mask=plotted_case_mask,
        pressure_case_plotted_mTorr=pressure_cases_arr[plotted_case_mask],
        rf_case_plotted_MHz=rf_cases_arr[plotted_case_mask],
        Komega_inverse_ratio_case_plotted=ratio_cases_arr[plotted_case_mask],
        x_midpoint_m=args.x_midpoint_m,
        L_b_m=args.L_b_m,
        x_average_min_m=x_left_m,
        x_average_max_m=x_right_m,
        k_lambda_D=args.k_lambda_D,
        argon_ion_mass_kg=ARGON_ION_MASS_KG,
        elementary_charge_C=ELEMENTARY_CHARGE_C,
        epsilon_0_F_per_m=EPSILON_0_F_PER_M,
    )
    write_case_summary_csv(args.out_dir / "Komega_inverse_ratio_case_summary.csv", rows)

    finite = ratio_cases_arr[np.isfinite(ratio_cases_arr)]

    # Internal consistency checks that directly test the implemented equations.
    finite_for_checks = plotted_case_mask & np.isfinite(Komega_direct_cases_arr)
    if np.any(finite_for_checks):
        product = ratio_cases_arr[finite_for_checks] * Komega_direct_cases_arr[finite_for_checks]
        inverse_product_err = float(np.nanmax(np.abs(product - 1.0)))
    else:
        inverse_product_err = np.nan

    g0_check_mask = np.isfinite(G0_cases_arr) & np.isfinite(wtilde2_cases_arr)
    if np.any(g0_check_mask):
        g0_plus_wtilde2_max_abs = float(np.nanmax(np.abs(G0_cases_arr[g0_check_mask] + wtilde2_cases_arr[g0_check_mask])))
    else:
        g0_plus_wtilde2_max_abs = np.nan

    omegaF_expected = 2.0 * np.pi * rf_cases_arr * 1.0e6
    omegaF_check_mask = np.isfinite(omega_F_cases_arr) & np.isfinite(omegaF_expected)
    if np.any(omegaF_check_mask):
        omegaF_max_abs_err = float(np.nanmax(np.abs(omega_F_cases_arr[omegaF_check_mask] - omegaF_expected[omegaF_check_mask])))
    else:
        omegaF_max_abs_err = np.nan

    print(f"Processed {len(sorted_keys)} common cases.")
    print(f"Excluded {np.count_nonzero(~plotted_case_mask)} cases with non-finite ratios from the plot.")
    print(f"Plotted {np.count_nonzero(plotted_case_mask)} cases with finite ratios.")
    print(f"Density sources: electron={density_source_e}, ion={density_source_i}")
    print(f"x_m = {args.x_midpoint_m:g} m, L_b = {args.L_b_m:g} m")
    print(f"Averaged frequencies and n_i over [{x_left_m:g}, {x_right_m:g}] m")
    print(f"k*lambda_D = {args.k_lambda_D:g} (user supplied; lambda_D not computed)")
    print(f"Argon ion mass = {ARGON_ION_MASS_KG:.12e} kg; charge state Z=1")
    print(f"Self-check max |omega_F - 2*pi*rf_MHz*1e6| = {omegaF_max_abs_err:g}")
    print(f"Self-check max |G0 + wtilde2| = {g0_plus_wtilde2_max_abs:g}")
    print(f"Self-check max |inverse*direct - 1| over plotted finite cases = {inverse_product_err:g}")
    print(f"Self-check non-finite plotted ratios = {np.count_nonzero(~np.isfinite(ratio_cases_arr[plotted_case_mask]))}")
    print(f"Saved SVG: {output_svg}")
    print(f"Saved grid arrays: {args.out_dir / 'K_omega_ratio_grid.npz'}")
    print(f"Saved case summary: {args.out_dir / 'K_omega_ratio_case_summary.csv'}")
    if finite.size:
        print("K_omega ratio range:", float(np.nanmin(finite)), "to", float(np.nanmax(finite)))


if __name__ == "__main__":
    main()
