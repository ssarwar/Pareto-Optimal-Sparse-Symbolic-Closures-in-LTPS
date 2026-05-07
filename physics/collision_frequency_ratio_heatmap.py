#!/usr/bin/env python3
"""
Compute and plot per-particle ion/electron collision-frequency ratios for argon CCP PIC cases.

Place this file under the repository's /physics directory and run from the
repository root with

    python physics/collision_frequency_ratio_heatmap.py

The script expects these 1D collision-rate pkl files produced/loaded by
physics/load_all_ml_data.py:

    data/rf/pic/ml_data/ccp_nu_e_ion.pkl
    data/rf/pic/ml_data/ccp_nu_i_cx.pkl
    data/rf/pic/ml_data/ccp_nu_i_iso.pkl

Each 1D record is assumed to be

    [rf_frequency_MHz, pressure_mTorr, x_grid_m, values]

By default, the ccp_nu_* files are treated as volumetric source/collision
rates [m^-3 s^-1]. Per-particle frequencies are computed as

    nu_e_ion(x) = S_e_ion(x) / n_e(x)
    nu_i_cx(x)  = S_i_cx(x)  / n_i(x)
    nu_i_iso(x) = (m_g / (m_i + m_g)) * S_i_iso(x) / n_i(x)
    nu_i(x)     = nu_i_cx(x) + nu_i_iso(x)

where n_e and n_i are read from ccp_ne.pkl and ccp_ni.pkl when available.
If those density files are absent, densities are derived by integrating
ccp_evdf_2d.pkl(.zst) and ccp_ivdf_2d.pkl(.zst) over velocity.

IMPORTANT:
For this paper, we need to apply a mass factor of m_g / (m_i + m_g) to the elastic 
scattering (isotropic/iso) collision frequency to get the drag term in the
ion momentum equation to have the form R = - n_i m_i nu_i (v_rel).
Here, m_g is the neutral gas mass and m_i is the ion mass.
Since we are considering Ar+-Ar colliions, we hardcode to 0.5.
Note that this does not apply to the charge-exchange "switch" model.
THIS IS AROUND LINE 680.

If the ccp_nu_* values are already per-particle frequencies [s^-1], run with
--rate-files-are-frequencies.
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
import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None


@dataclass(frozen=True)
class ProfileCase:
    rf_MHz: float
    pressure_mTorr: float
    x_m: np.ndarray
    values: np.ndarray


def repo_root_from_this_file() -> Path:
    """Return the repository root when this script lives in /physics."""
    return Path(__file__).resolve().parents[1]


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


def trapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Use numpy.trapezoid when available; fall back to numpy.trapz."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    return np.trapz(y, x, axis=axis)


def case_key(rf_MHz: float, pressure_mTorr: float, ndigits: int = 10) -> Tuple[float, float]:
    """A stable key for matching cases across different pkl files."""
    return (round(float(rf_MHz), ndigits), round(float(pressure_mTorr), ndigits))


def load_1d_dataset(data_dir: Path, stem: str) -> Dict[Tuple[float, float], ProfileCase]:
    """Load one 1D CCP pkl dataset into a dict keyed by (rf_MHz, pressure_mTorr)."""
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
    """Load ccp_ne/ni if present, otherwise derive from evdf/ivdf."""
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
    """Sort a profile by x and smooth y over a Gaussian length smooth_sigma_m."""
    x = np.asarray(x_m, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
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


def safe_per_particle_frequency(
    volumetric_rate: np.ndarray,
    density: np.ndarray,
    density_floor_rel: float,
) -> np.ndarray:
    """Compute rate/density with a relative density floor."""
    density = np.asarray(density, dtype=float)
    rate = np.asarray(volumetric_rate, dtype=float)
    max_density = np.nanmax(density) if np.isfinite(density).any() else np.nan
    if not np.isfinite(max_density) or max_density <= 0.0:
        return np.full_like(rate, np.nan, dtype=float)
    floor = density_floor_rel * max_density
    density_safe = np.where(density > floor, density, np.nan)
    return rate / density_safe


def window_values(x_m: np.ndarray, y: np.ndarray, xmin_m: float, xmax_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """Values over [xmin_m, xmax_m], including interpolated endpoints."""
    x = np.asarray(x_m, dtype=float).reshape(-1)
    yy = np.asarray(y, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(yy)
    if finite.sum() < 2 or xmax_m <= xmin_m:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = x[finite]
    yy = yy[finite]
    order = np.argsort(x)
    x = x[order]
    yy = yy[order]

    lo = max(float(xmin_m), float(x[0]))
    hi = min(float(xmax_m), float(x[-1]))
    if hi <= lo:
        return np.array([], dtype=float), np.array([], dtype=float)

    interior = (x > lo) & (x < hi)
    xs = np.concatenate([[lo], x[interior], [hi]])
    ys = np.interp(xs, x, yy)
    return xs, ys


def spatial_average(x_m: np.ndarray, y: np.ndarray, xmin_m: float, xmax_m: float) -> float:
    """Average y over xmin_m <= x <= xmax_m using trapezoidal integration."""
    xs, ys = window_values(x_m, y, xmin_m, xmax_m)
    if xs.size == 0:
        return np.nan
    if xs.size == 1 or xs[-1] <= xs[0]:
        return float(ys[0])
    return float(trapz(ys, xs) / (xs[-1] - xs[0]))


def spatial_minmax(x_m: np.ndarray, y: np.ndarray, xmin_m: float, xmax_m: float) -> Tuple[float, float]:
    xs, ys = window_values(x_m, y, xmin_m, xmax_m)
    if ys.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(ys)), float(np.nanmax(ys))


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
    """Average scalar case values onto a grid shaped (n_rf, n_pressure)."""
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
        i = f_to_i[f]
        j = p_to_j[p]
        grid_sum[i, j] += v
        grid_count[i, j] += 1

    grid = np.full_like(grid_sum, np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return p_unique, f_unique, grid


def plot_ratio_heatmap(
    p_unique: np.ndarray,
    f_unique: np.ndarray,
    ratio_grid: np.ndarray,
    pressure_cases: np.ndarray,
    rf_cases: np.ndarray,
    output_svg: Path,
    title: str,
) -> None:
    """Plot the nu_i/nu_e heatmap and overlay only finite-ratio simulation points."""
    p_edges = centers_to_edges(p_unique)
    f_edges = centers_to_edges(f_unique)
    Z = np.asarray(ratio_grid, dtype=float)
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        raise ValueError("The ratio grid contains no finite values to plot.")

    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if np.isclose(vmin, vmax):
        pad = 0.05 * abs(vmin) if vmin != 0.0 else 1.0
        vmin -= pad
        vmax += pad

    print(vmin)

    # WARNING
    # Changing vmin and vmax for plot scaling, chosing somewhat arbitrarily for scaling
    vmin = 0
    vmax = min(30, 40.0)
    # END

    fig, ax = plt.subplots(figsize=(6.5, 5.1), constrained_layout=True)
    pcm = ax.pcolormesh(
        p_edges,
        f_edges,
        np.ma.masked_invalid(Z),
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        #vmax = 40.0,  # limit the color scale to better show variations at low ratios; values above this will be saturated
    )

    marker_mask = np.isfinite(pressure_cases) & np.isfinite(rf_cases)
    if marker_mask.any():
        ax.scatter(
            pressure_cases[marker_mask],
            rf_cases[marker_mask],
            s=32,
            facecolors="none",
            edgecolors="tab:red",
            linewidths=1.0,
            alpha=0.65,
            zorder=3,
            label="PIC cases with finite ratio",
        )

    ax.set_xlabel("Pressure [mTorr]", fontsize=15)
    ax.set_ylabel("Frequency [MHz]", fontsize=15)
    # ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlim(p_edges[0], p_edges[-1])
    ax.set_ylim(f_edges[0], f_edges[-1])
    ax.minorticks_on()
    ax.tick_params(which="major", direction="in", top=True, right=True, length=5, width=0.9, labelsize=12)
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=3, width=0.7)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    #cbar = fig.colorbar(pcm, ax=ax, pad=0.02, fraction=0.055)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\nu_i / \nu_e$", fontsize=13)
    cbar.ax.tick_params(direction="in", labelsize=11, length=4, width=0.8)

    # Set ticks manually to ensure a reasonable number of ticks even if the range is small or large.
    num_ticks = 7
    ticks = np.linspace(vmin, vmax, num_ticks)
    cbar.set_ticks(ticks)
    print(cbar.get_ticks())
    # END set ticks manually

    # cbar.locator = MaxNLocator(nbins=6)
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


def finite_range(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def print_range(label: str, values: np.ndarray) -> None:
    vmin, vmax = finite_range(values)
    print(f"  {label:<28s} min={vmin:.6e}  max={vmax:.6e}")


def parse_args() -> argparse.Namespace:
    default_root = repo_root_from_this_file()
    parser = argparse.ArgumentParser(
        description="Compute smoothed nu_i/nu_e maps from CCP collision-rate pkl files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_root / "data" / "rf" / "pic" / "ml_data",
        help="Directory containing ccp_*.pkl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_root / "physics" / "output_collision_frequency",
        help="Directory for the SVG and saved arrays.",
    )
    parser.add_argument(
        "--smooth-sigma-m",
        type=float,
        default=1.0e-3,
        help="Gaussian smoothing length in meters. Use 0 to disable smoothing. Default: 1 mm.",
    )
    parser.add_argument(
        "--x-average-min-m",
        type=float,
        default=0.015,
        help="Lower x bound for bulk averaging [m]. Default: 1.5 cm.",
    )
    parser.add_argument(
        "--x-average-max-m",
        type=float,
        default=0.035,
        help="Upper x bound for bulk averaging [m]. Default: 3.5 cm.",
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
        "--fig-name",
        default="nu_i_over_nu_e_heatmap.svg",
        help="SVG filename written inside --out-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.x_average_max_m <= args.x_average_min_m:
        raise ValueError("--x-average-max-m must be larger than --x-average-min-m.")

    datasets: dict[str, Dict[Tuple[float, float], ProfileCase]] = {
        "e_ion_rate": load_1d_dataset(args.data_dir, "ccp_nu_e_ion"),
        "i_cx_rate": load_1d_dataset(args.data_dir, "ccp_nu_i_cx"),
        "i_iso_rate": load_1d_dataset(args.data_dir, "ccp_nu_i_iso"),
    }

    density_sources = "not used; ccp_nu_* treated as frequencies"
    if not args.rate_files_are_frequencies:
        datasets["ne"], ne_source = load_density_dataset(
            args.data_dir, "ccp_ne", "ccp_evdf_2d", "electron"
        )
        datasets["ni"], ni_source = load_density_dataset(
            args.data_dir, "ccp_ni", "ccp_ivdf_2d", "ion"
        )
        density_sources = f"electron={ne_source}, ion={ni_source}"

    common_keys = set(datasets["e_ion_rate"].keys())
    for ds in datasets.values():
        common_keys &= set(ds.keys())
    if not common_keys:
        raise ValueError("No common (rf_MHz, pressure_mTorr) cases found across the pkl files.")

    for name, ds in datasets.items():
        missing = sorted(set(datasets["e_ion_rate"].keys()) - set(ds.keys()))
        if missing:
            print(f"Warning: {name} is missing {len(missing)} cases found in ccp_nu_e_ion.pkl.")

    rows: list[dict] = []
    pressure_cases: list[float] = []
    rf_cases: list[float] = []
    ratio_cases: list[float] = []
    nu_e_avg_cases: list[float] = []
    nu_i_cx_avg_cases: list[float] = []
    nu_i_iso_avg_cases: list[float] = []
    nu_i_total_avg_cases: list[float] = []

    bulk_pointwise_nu_e: list[np.ndarray] = []
    bulk_pointwise_nu_i_cx: list[np.ndarray] = []
    bulk_pointwise_nu_i_iso: list[np.ndarray] = []
    bulk_pointwise_nu_i_total: list[np.ndarray] = []

    sorted_keys = sorted(common_keys, key=lambda k: (k[1], k[0]))

    for key in sorted_keys:
        rf_MHz, pressure_mTorr = key
        e_case = datasets["e_ion_rate"][key]

        x_m, e_ion_rate = smooth_on_x(
            e_case.x_m,
            e_case.values,
            args.smooth_sigma_m,
            clip_nonnegative=True,
        )

        i_cx_rate = interp_smoothed(
            datasets["i_cx_rate"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
        )
        i_iso_rate = interp_smoothed(
            datasets["i_iso_rate"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
        )

        if args.rate_files_are_frequencies:
            nu_e_ion = e_ion_rate
            nu_i_cx = i_cx_rate
            nu_i_iso = i_iso_rate
        else:
            ne = interp_smoothed(
                datasets["ne"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
            )
            ni = interp_smoothed(
                datasets["ni"][key], x_m, args.smooth_sigma_m, clip_nonnegative=True
            )
            nu_e_ion = safe_per_particle_frequency(e_ion_rate, ne, args.density_floor_rel)
            nu_i_cx = safe_per_particle_frequency(i_cx_rate, ni, args.density_floor_rel)
            nu_i_iso = safe_per_particle_frequency(i_iso_rate, ni, args.density_floor_rel)

        nu_i_iso = 0.5 * nu_i_iso         # hardcode mass factor (m_g / (m_i + m_g)) as explained above
        nu_i_total = nu_i_cx + nu_i_iso

        nu_e_avg = spatial_average(x_m, nu_e_ion, args.x_average_min_m, args.x_average_max_m)
        nu_i_cx_avg = spatial_average(x_m, nu_i_cx, args.x_average_min_m, args.x_average_max_m)
        nu_i_iso_avg = spatial_average(x_m, nu_i_iso, args.x_average_min_m, args.x_average_max_m)
        nu_i_total_avg = spatial_average(x_m, nu_i_total, args.x_average_min_m, args.x_average_max_m)
        ratio = nu_i_total_avg / nu_e_avg if np.isfinite(nu_e_avg) and nu_e_avg > 0.0 else np.nan

        nu_e_min, nu_e_max = spatial_minmax(x_m, nu_e_ion, args.x_average_min_m, args.x_average_max_m)
        nu_i_cx_min, nu_i_cx_max = spatial_minmax(x_m, nu_i_cx, args.x_average_min_m, args.x_average_max_m)
        nu_i_iso_min, nu_i_iso_max = spatial_minmax(x_m, nu_i_iso, args.x_average_min_m, args.x_average_max_m)
        nu_i_total_min, nu_i_total_max = spatial_minmax(x_m, nu_i_total, args.x_average_min_m, args.x_average_max_m)

        _, ywin = window_values(x_m, nu_e_ion, args.x_average_min_m, args.x_average_max_m)
        if ywin.size:
            bulk_pointwise_nu_e.append(ywin)
        _, ywin = window_values(x_m, nu_i_cx, args.x_average_min_m, args.x_average_max_m)
        if ywin.size:
            bulk_pointwise_nu_i_cx.append(ywin)
        _, ywin = window_values(x_m, nu_i_iso, args.x_average_min_m, args.x_average_max_m)
        if ywin.size:
            bulk_pointwise_nu_i_iso.append(ywin)
        _, ywin = window_values(x_m, nu_i_total, args.x_average_min_m, args.x_average_max_m)
        if ywin.size:
            bulk_pointwise_nu_i_total.append(ywin)

        pressure_cases.append(pressure_mTorr)
        rf_cases.append(rf_MHz)
        ratio_cases.append(ratio)
        nu_e_avg_cases.append(nu_e_avg)
        nu_i_cx_avg_cases.append(nu_i_cx_avg)
        nu_i_iso_avg_cases.append(nu_i_iso_avg)
        nu_i_total_avg_cases.append(nu_i_total_avg)

        rows.append(
            {
                "rf_MHz": rf_MHz,
                "pressure_mTorr": pressure_mTorr,
                "x_average_min_m": args.x_average_min_m,
                "x_average_max_m": args.x_average_max_m,
                "nu_e_ion_avg_s^-1": nu_e_avg,
                "nu_i_cx_avg_s^-1": nu_i_cx_avg,
                "nu_i_iso_avg_s^-1": nu_i_iso_avg,
                "nu_i_total_avg_s^-1": nu_i_total_avg,
                "nu_i_total_over_nu_e_ion": ratio,
                "nu_e_ion_min_s^-1": nu_e_min,
                "nu_e_ion_max_s^-1": nu_e_max,
                "nu_i_cx_min_s^-1": nu_i_cx_min,
                "nu_i_cx_max_s^-1": nu_i_cx_max,
                "nu_i_iso_min_s^-1": nu_i_iso_min,
                "nu_i_iso_max_s^-1": nu_i_iso_max,
                "nu_i_total_min_s^-1": nu_i_total_min,
                "nu_i_total_max_s^-1": nu_i_total_max,
            }
        )

    pressure_cases_arr = np.asarray(pressure_cases, dtype=float)
    rf_cases_arr = np.asarray(rf_cases, dtype=float)
    ratio_cases_arr = np.asarray(ratio_cases, dtype=float)
    nu_e_avg_cases_arr = np.asarray(nu_e_avg_cases, dtype=float)
    nu_i_cx_avg_cases_arr = np.asarray(nu_i_cx_avg_cases, dtype=float)
    nu_i_iso_avg_cases_arr = np.asarray(nu_i_iso_avg_cases, dtype=float)
    nu_i_total_avg_cases_arr = np.asarray(nu_i_total_avg_cases, dtype=float)

    # Do not plot simulation cases whose ratio is NaN or otherwise non-finite.
    # Those cases remain in the CSV summary and in the full per-case NPZ arrays.
    plot_case_mask = np.isfinite(ratio_cases_arr)
    n_skipped_plot_cases = int((~plot_case_mask).sum())
    if not plot_case_mask.any():
        raise ValueError("No finite nu_i_total/nu_e_ion ratios are available to plot.")

    pressure_cases_plot_arr = pressure_cases_arr[plot_case_mask]
    rf_cases_plot_arr = rf_cases_arr[plot_case_mask]
    ratio_cases_plot_arr = ratio_cases_arr[plot_case_mask]
    nu_e_avg_cases_plot_arr = nu_e_avg_cases_arr[plot_case_mask]
    nu_i_cx_avg_cases_plot_arr = nu_i_cx_avg_cases_arr[plot_case_mask]
    nu_i_iso_avg_cases_plot_arr = nu_i_iso_avg_cases_arr[plot_case_mask]
    nu_i_total_avg_cases_plot_arr = nu_i_total_avg_cases_arr[plot_case_mask]

    p_unique, f_unique, ratio_grid = make_grid(
        pressure_cases_plot_arr, rf_cases_plot_arr, ratio_cases_plot_arr
    )
    _, _, nu_e_grid = make_grid(
        pressure_cases_plot_arr, rf_cases_plot_arr, nu_e_avg_cases_plot_arr
    )
    _, _, nu_i_cx_grid = make_grid(
        pressure_cases_plot_arr, rf_cases_plot_arr, nu_i_cx_avg_cases_plot_arr
    )
    _, _, nu_i_iso_grid = make_grid(
        pressure_cases_plot_arr, rf_cases_plot_arr, nu_i_iso_avg_cases_plot_arr
    )
    _, _, nu_i_total_grid = make_grid(
        pressure_cases_plot_arr, rf_cases_plot_arr, nu_i_total_avg_cases_plot_arr
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_svg = args.out_dir / args.fig_name
    title = (
        r"Argon CCP collision-frequency ratio "
        + rf"$x\in[{args.x_average_min_m*100:.1f},{args.x_average_max_m*100:.1f}]$ cm"
    )
    plot_ratio_heatmap(
        p_unique,
        f_unique,
        ratio_grid,
        pressure_cases_plot_arr,
        rf_cases_plot_arr,
        output_svg,
        title,
    )

    np.savez(
        args.out_dir / "nu_i_over_nu_e_grid.npz",
        pressure_mTorr=p_unique,
        rf_MHz=f_unique,
        nu_i_total_over_nu_e_ion=ratio_grid,
        nu_e_ion_avg_s_inv=nu_e_grid,
        nu_i_cx_avg_s_inv=nu_i_cx_grid,
        nu_i_iso_avg_s_inv=nu_i_iso_grid,
        nu_i_total_avg_s_inv=nu_i_total_grid,
        plotted_case_mask=plot_case_mask,
        pressure_case_mTorr=pressure_cases_arr,
        rf_case_MHz=rf_cases_arr,
        nu_i_total_over_nu_e_ion_case=ratio_cases_arr,
        nu_e_ion_avg_case_s_inv=nu_e_avg_cases_arr,
        nu_i_cx_avg_case_s_inv=nu_i_cx_avg_cases_arr,
        nu_i_iso_avg_case_s_inv=nu_i_iso_avg_cases_arr,
        nu_i_total_avg_case_s_inv=nu_i_total_avg_cases_arr,
        pressure_case_plotted_mTorr=pressure_cases_plot_arr,
        rf_case_plotted_MHz=rf_cases_plot_arr,
        nu_i_total_over_nu_e_ion_case_plotted=ratio_cases_plot_arr,
        nu_e_ion_avg_case_plotted_s_inv=nu_e_avg_cases_plot_arr,
        nu_i_cx_avg_case_plotted_s_inv=nu_i_cx_avg_cases_plot_arr,
        nu_i_iso_avg_case_plotted_s_inv=nu_i_iso_avg_cases_plot_arr,
        nu_i_total_avg_case_plotted_s_inv=nu_i_total_avg_cases_plot_arr,
    )
    output_csv = args.out_dir / "nu_i_over_nu_e_case_summary.csv"
    write_case_summary_csv(output_csv, rows)

    print(f"Processed {len(sorted_keys)} common cases.")
    if n_skipped_plot_cases:
        print(f"Excluded {n_skipped_plot_cases} cases with non-finite ratios from the plot.")
    print(f"Plotted {int(plot_case_mask.sum())} cases with finite ratios.")
    print(f"Density sources: {density_sources}")
    print(f"Saved SVG: {output_svg}")
    print(f"Saved grid arrays: {args.out_dir / 'nu_i_over_nu_e_grid.npz'}")
    print(f"Saved case summary: {output_csv}")
    print("\nBulk-window pointwise per-particle frequency ranges [s^-1]:")
    print_range("nu_e_ion(x)", np.concatenate(bulk_pointwise_nu_e))
    print_range("nu_i_cx(x)", np.concatenate(bulk_pointwise_nu_i_cx))
    print_range("nu_i_iso(x)", np.concatenate(bulk_pointwise_nu_i_iso))
    print_range("nu_i_total(x)", np.concatenate(bulk_pointwise_nu_i_total))
    print("\nBulk-averaged per-case frequency ranges [s^-1]:")
    print_range("<nu_e_ion>_x", nu_e_avg_cases_arr)
    print_range("<nu_i_cx>_x", nu_i_cx_avg_cases_arr)
    print_range("<nu_i_iso>_x", nu_i_iso_avg_cases_arr)
    print_range("<nu_i_total>_x", nu_i_total_avg_cases_arr)
    print("\nRatio range:")
    print_range("<nu_i_total>/<nu_e_ion>", ratio_cases_arr)


if __name__ == "__main__":
    main()
