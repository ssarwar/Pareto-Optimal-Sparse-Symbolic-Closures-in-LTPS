import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as ast
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import pandas as pd
import sympy as sp
import seaborn as sns
from matplotlib.ticker import ScalarFormatter 
import utils.constants as Consts
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
plt.style.use(["seaborn-v0_8-colorblind"])
plt.rcParams.update({
    "font.size":14})
p = Consts.PhysicalConstants() # physical constants

class Plotting():
    def __init__(self, PATH_TO_DATA, svg_figures=None, png_figures=False):
        self.png_figures = png_figures
        self.PATH_TO_DATA = PATH_TO_DATA
        self.svg_figures = svg_figures    

    @staticmethod
    def _ensure_math(s):
        s = str(s).strip()
        if s.startswith("$") and s.endswith("$"):
            return s
        return f"${s.strip('$')}$"

    @staticmethod
    def _hypervolume_2d_min(points, ref):
        pts = np.asarray(points, float)
        ref = np.asarray(ref, float)

        if pts.size == 0:
            return 0.0

        mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[mask]
        if pts.size == 0:
            return 0.0

        # transform to a 2D max problem with reference at (0, 0)
        u = ref - pts
        u = np.clip(u, 0.0, None)

        mask = (u[:, 0] > 0) & (u[:, 1] > 0)
        u = u[mask]
        if u.size == 0:
            return 0.0

        # sort by first coordinate
        order = np.argsort(u[:, 0])
        x = u[order, 0]
        y = u[order, 1]
        k = len(x)

        # keep non-dominated points for maximization
        pareto_mask = np.zeros(k, dtype=bool)
        best_y = 0.0
        for i in range(k - 1, -1, -1):
            if y[i] > best_y:
                best_y = y[i]
                pareto_mask[i] = True

        px = x[pareto_mask]
        py = y[pareto_mask]
        order2 = np.argsort(px)
        px = px[order2]
        py = py[order2]

        hv = 0.0
        x_prev = 0.0
        for xi, yi in zip(px, py):
            dx = max(xi - x_prev, 0.0)
            hv += dx * yi
            x_prev = xi

        return float(hv)

    def mobo_2d_pareto(
        self, df, x="nonzero_count", set_xlabel="Complexity",
        y="generalizability_error", set_ylabel="Generalizability",
        plot_save_name="pareto", xlim=None, ylim=None,
        nbins_x=None, x_bin_width=None, xtick_cnt=2,
        nbins_y=50, y_bin_width=None,
        xlim_ins=None, ylim_ins=None,
        nbins_x_ins=None, x_bin_width_ins=None,
        nbins_y_ins=19, y_bin_width_ins=None,
        inset_width="45%", inset_height="45%",
        inset_loc="upper right", inset_borderpad=1.0,
        log_counts=False, count_gamma=0.5, vmax_percentile=99.0, max_xticks=12,
        show=False,
        **_ignored_kwargs
    ):
        df_f = df[np.isfinite(df[x]) & np.isfinite(df[y])]
        if df_f.empty:
            return

        def _make_edges(lo, hi, nbins=None, bin_width=None, default_nbins=50):
            lo = float(lo)
            hi = float(hi)
            if hi <= lo:
                hi = lo + 1.0

            if bin_width is not None:
                bw = float(bin_width)
                if bw <= 0:
                    n_eff = max(int(default_nbins if nbins is None else nbins), 2)
                    bw = (hi - lo) / n_eff
                edges = np.arange(lo, hi + bw, bw, dtype=float)
                if edges.size < 2:
                    edges = np.array([lo, hi], dtype=float)
                edges[0] = lo
                edges[-1] = hi
            else:
                n_eff = max(int(default_nbins if nbins is None else nbins), 2)
                edges = np.linspace(lo, hi, n_eff + 1)

            return edges

        def _make_centered_integer_edges(lo, hi, nbins=None, bin_width=None):
            n_cols = float(hi - lo + 1)
            if n_cols <= 0:
                n_cols = 1.0

            if bin_width is not None:
                bw = float(bin_width)
                if bw <= 0:
                    n_eff = max(int(n_cols if nbins is None else nbins), 2)
                    bw = n_cols / n_eff
            elif nbins is not None:
                n_eff = max(int(nbins), 2)
                bw = n_cols / n_eff
            else:
                bw = 1.0

            x_start = float(lo) - 0.5 * bw
            x_stop = float(hi) + 0.5 * bw
            n_eff = max(int(np.round((x_stop - x_start) / bw)), 1)
            edges = x_start + bw * np.arange(n_eff + 1, dtype=float)
            edges[0] = x_start
            edges[-1] = x_stop
            return edges

        if x == "a":
            c_raw = df_f[x].to_numpy(dtype=float)
            c_int = np.rint(c_raw).astype(int)

            mvals = df_f[y].to_numpy(dtype=float)

            if xlim is None:
                x_lo = 0
                x_hi = int(max(np.max(c_int), 0))
            else:
                if np.ndim(xlim) == 0:
                    x_lo = 0
                    x_hi = int(np.ceil(float(xlim)))
                else:
                    x_lo = int(np.floor(float(xlim[0])))
                    x_hi = int(np.ceil(float(xlim[1])))

            x_lo = max(x_lo, 0)
            x_hi = max(x_hi, x_lo)

            ylim = _ignored_kwargs.get("ylim", None)
            if ylim is None:
                y_lo = 0.0
                y_hi = 0.5
            else:
                y_lo = float(ylim[0])
                y_hi = float(ylim[1])
                if y_hi <= y_lo:
                    y_lo, y_hi = 0.0, 0.5

            m = (
                np.isfinite(c_int) & np.isfinite(mvals)
                & (c_int >= x_lo) & (c_int <= x_hi)
                & (mvals >= y_lo) & (mvals <= y_hi)
            )
            c_int = c_int[m]
            mvals = mvals[m]
            if c_int.size == 0:
                return

            x_edges = _make_centered_integer_edges(
                x_lo,
                x_hi,
                nbins=nbins_x,
                bin_width=x_bin_width
            )

            if y_bin_width is not None:
                bw = float(y_bin_width)
                if bw <= 0:
                    bw = (y_hi - y_lo) / max(int(nbins_y), 2)
                y_edges = np.arange(y_lo, y_hi + bw, bw)
                if y_edges.size < 2:
                    y_edges = np.array([y_lo, y_hi], dtype=float)
                y_edges[0] = y_lo
                y_edges[-1] = y_hi
            else:
                nbins_y = max(int(nbins_y), 2)
                y_edges = np.linspace(y_lo, y_hi, nbins_y + 1)

            H, _, _ = np.histogram2d(c_int, mvals, bins=[x_edges, y_edges])
            H = H.T
            Hm = np.ma.masked_where(H == 0, H)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.set_facecolor("white")

            norm = None
            if Hm.count():
                pos = Hm.compressed()
                vmax = float(np.percentile(pos, float(vmax_percentile)))
                vmax = max(vmax, 1.0)
                if log_counts:
                    norm = LogNorm(vmin=1.0, vmax=vmax)
                else:
                    norm = PowerNorm(gamma=float(count_gamma), vmin=0.0, vmax=vmax)

            ax.pcolormesh(
                x_edges,
                y_edges,
                Hm,
                shading="auto",
                cmap="Blues",
                norm=norm
            )

            if plot_save_name == "CvV_lim":
                y_raw = df_f[y].to_numpy(dtype=float)

                y_zoom_hi = float(np.nanpercentile(y_raw, 95))
                y_zoom_hi = min(max(y_zoom_hi, 1e-3), 0.005)

                if xlim_ins is None:
                    x_lo_ins = float(x_lo)
                    x_hi_ins = float(x_hi)
                else:
                    x_lo_ins = float(xlim_ins[0])
                    x_hi_ins = float(xlim_ins[1])

                x_lo_ins = max(x_lo_ins, float(x_lo))
                x_hi_ins = min(x_hi_ins, float(x_hi))
                if x_hi_ins <= x_lo_ins:
                    x_hi_ins = x_lo_ins + 1.0

                if ylim_ins is None:
                    y_lo_ins = 0.0
                    y_hi_ins = float(y_zoom_hi)
                else:
                    y_lo_ins = float(ylim_ins[0])
                    y_hi_ins = float(ylim_ins[1])

                if y_hi_ins <= y_lo_ins:
                    y_lo_ins = 0.0
                    y_hi_ins = float(y_zoom_hi)

                axins = inset_axes(
                    ax,
                    width=inset_width,
                    height=inset_height,
                    loc=inset_loc,
                    borderpad=inset_borderpad,
                )
                axins.set_facecolor("white")
                axins.pcolormesh(
                    x_edges,
                    y_edges,
                    Hm,
                    shading="auto",
                    cmap="Blues",
                    norm=norm,
                )
                axins.set_xlim([x_lo_ins, x_hi_ins])
                axins.set_ylim([y_lo_ins, y_hi_ins])
                axins.tick_params(axis="both", labelsize=9)
                axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axins.yaxis.get_offset_text().set_size(9)

                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=1.0)

            ax.set_xlabel(set_xlabel)
            ax.set_ylabel(set_ylabel)

            ax.set_xlim([x_edges[0], x_edges[-1]])
            ax.set_ylim([y_lo, y_hi])

            span = max(x_hi - x_lo, 1)

            def _nice_step(span_val, max_ticks_val):
                raw = span_val / max(float(max_ticks_val), 1.0)
                if raw <= 1.0:
                    return 1
                p10 = 10 ** int(np.floor(np.log10(raw)))
                base = raw / p10
                if base <= 1.0:
                    step = 1 * p10
                elif base <= 2.0:
                    step = 2 * p10
                elif base <= 5.0:
                    step = 5 * p10
                else:
                    step = 10 * p10
                return int(max(step, 1))

            step = _nice_step(span, int(max_xticks))
            major = np.arange(x_lo, x_hi + 1, step, dtype=int)
            if major.size == 0 or major[0] != x_lo:
                major = np.r_[x_lo, major]
            if major[-1] != x_hi:
                major = np.r_[major, x_hi]
            ax.set_xticks(major)

        elif x == "nonzero_count":
            xh = df_f[x].to_numpy(dtype=float)
            yh = df_f[y].to_numpy(dtype=float)

            if xlim is None:
                x_lo = 0.0
                x_hi = float(np.nanmax(xh))
            else:
                if np.ndim(xlim) == 0:
                    x_lo = 0.0
                    x_hi = float(xlim)
                else:
                    x_lo = float(xlim[0])
                    x_hi = float(xlim[1])

            if not np.isfinite(x_lo):
                x_lo = 0.0
            if not np.isfinite(x_hi) or x_hi <= x_lo:
                x_hi = x_lo + 1.0

            ylim = _ignored_kwargs.get("ylim", None)
            if ylim is None:
                y_lo_disp = 0.0
                y_hi_disp = 0.5
            else:
                y_lo_disp = float(ylim[0])
                y_hi_disp = float(ylim[1])
                if y_hi_disp <= y_lo_disp:
                    y_lo_disp, y_hi_disp = 0.0, 0.5

            x_edges = _make_edges(
                x_lo,
                x_hi,
                nbins=nbins_x,
                bin_width=x_bin_width,
                default_nbins=nbins_y
            )

            if y_bin_width is not None:
                bw = float(y_bin_width)
                if bw <= 0:
                    bw = 1.0 / max(int(nbins_y), 2)
                y_edges = np.arange(0.0, 1.0 + bw, bw)
                if y_edges.size < 2:
                    y_edges = np.array([0.0, 1.0], dtype=float)
                y_edges[0] = 0.0
                y_edges[-1] = 1.0
            else:
                y_edges = np.linspace(0.0, 1.0, max(int(nbins_y), 2) + 1)

            H, _, _ = np.histogram2d(xh, yh, bins=[x_edges, y_edges])
            H = H.T

            n_total = float(xh.size)
            if n_total > 0:
                dx = np.diff(x_edges)
                dy = np.diff(y_edges)
                cell_area = dy[:, None] * dx[None, :]
                H = H / (n_total * cell_area)

            Hm = np.ma.masked_where(H == 0, H)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.set_facecolor("white")

            norm = None
            if Hm.count():
                pos = Hm.compressed()
                vmax = float(np.percentile(pos, float(vmax_percentile)))
                vmax = max(vmax, np.finfo(float).tiny)
                if log_counts:
                    norm = LogNorm(vmin=max(pos.min(), np.finfo(float).tiny), vmax=vmax)
                else:
                    norm = PowerNorm(gamma=float(count_gamma), vmin=0.0, vmax=vmax)

            ax.pcolormesh(
                x_edges,
                y_edges,
                Hm,
                shading="auto",
                cmap="Blues",
                norm=norm
            )

            ax.set_xlabel(set_xlabel, fontsize=14)
            ax.set_ylabel(set_ylabel, fontsize=14)
            ax.tick_params(axis="both", labelsize=14)

            ax.set_xlim([x_lo, x_hi])
            ax.set_ylim([y_lo_disp, y_hi_disp])

            major = np.arange(x_lo, x_hi + 1, xtick_cnt, dtype=int)
            if major.size == 0 or major[0] != x_lo:
                major = np.r_[x_lo, major]
            if major[-1] != x_hi:
                major = np.r_[major, x_hi]
            ax.set_xticks(major)

            if plot_save_name == "CvV_lim":
                x_raw = df_f[x].to_numpy(dtype=float)
                y_raw = df_f[y].to_numpy(dtype=float)

                y_zoom_hi = float(np.nanpercentile(y_raw, 95))
                y_zoom_hi = min(max(y_zoom_hi, 1e-3), 0.005)

                if xlim_ins is None:
                    x_lo_ins = float(x_lo)
                    x_hi_ins = float(x_hi)
                else:
                    x_lo_ins = float(xlim_ins[0])
                    x_hi_ins = float(xlim_ins[1])

                x_lo_ins = max(x_lo_ins, float(x_lo))
                x_hi_ins = min(x_hi_ins, float(x_hi))
                if x_hi_ins <= x_lo_ins:
                    x_hi_ins = x_lo_ins + 1.0

                if ylim_ins is None:
                    y_lo_ins = 0.0
                    y_hi_ins = float(y_zoom_hi)
                else:
                    y_lo_ins = float(ylim_ins[0])
                    y_hi_ins = float(ylim_ins[1])

                if y_hi_ins <= y_lo_ins:
                    y_lo_ins = 0.0
                    y_hi_ins = float(y_zoom_hi)

                x_edges_ins = _make_edges(
                    x_lo_ins,
                    x_hi_ins,
                    nbins=nbins_x_ins,
                    bin_width=x_bin_width_ins,
                    default_nbins=nbins_x if nbins_x is not None else 50
                )

                if y_bin_width_ins is not None:
                    bw_ins = float(y_bin_width_ins)
                    if bw_ins <= 0:
                        bw_ins = (y_hi_ins - y_lo_ins) / max(int(nbins_y_ins), 2)
                    y_edges_ins = np.arange(y_lo_ins, y_hi_ins + bw_ins, bw_ins, dtype=float)
                    if y_edges_ins.size < 2:
                        y_edges_ins = np.array([y_lo_ins, y_hi_ins], dtype=float)
                    y_edges_ins[0] = y_lo_ins
                    y_edges_ins[-1] = y_hi_ins
                else:
                    y_edges_ins = np.linspace(
                        y_lo_ins,
                        y_hi_ins,
                        max(int(nbins_y_ins), 2) + 1
                    )

                m_ins = (
                    np.isfinite(x_raw) & np.isfinite(y_raw)
                    & (x_raw >= x_lo_ins) & (x_raw <= x_hi_ins)
                    & (y_raw >= y_lo_ins) & (y_raw <= y_hi_ins)
                )
                x_ins = x_raw[m_ins]
                y_ins = y_raw[m_ins]

                H_ins, _, _ = np.histogram2d(
                    x_ins,
                    y_ins,
                    bins=[x_edges_ins, y_edges_ins]
                )
                H_ins = H_ins.T

                # Use the main-panel cell area as the reference area so that
                # inset colors are comparable to the full plot colors.
                dx_ref = np.diff(x_edges)
                dy_ref = np.diff(y_edges)
                ref_area = float(np.median(dx_ref) * np.median(dy_ref))

                if n_total > 0:
                    H_ins = H_ins / (n_total * ref_area)

                Hm_ins = np.ma.masked_where(H_ins == 0, H_ins)

                axins = inset_axes(
                    ax,
                    width=inset_width,
                    height=inset_height,
                    loc=inset_loc,
                    borderpad=inset_borderpad,
                )
                axins.set_facecolor("white")
                axins.pcolormesh(
                    x_edges_ins,
                    y_edges_ins,
                    Hm_ins,
                    shading="auto",
                    cmap="Blues",
                    norm=norm,
                )
                axins.set_xlim([x_lo_ins, x_hi_ins])
                axins.set_ylim([y_lo_ins, y_hi_ins])
                axins.tick_params(axis="both", labelsize=9)
                axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axins.yaxis.get_offset_text().set_size(9)

                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=1.0)
        else:
            xh = df_f[x].to_numpy(dtype=float)
            yh = df_f[y].to_numpy(dtype=float)

            x_edges = _make_edges(
                0.0,
                1.0,
                nbins=nbins_x,
                bin_width=x_bin_width,
                default_nbins=nbins_y
            )

            if y_bin_width is not None:
                bw = float(y_bin_width)
                if bw <= 0:
                    bw = 1.0 / max(int(nbins_y), 2)
                y_edges = np.arange(0.0, 1.0 + bw, bw)
                if y_edges.size < 2:
                    y_edges = np.array([0.0, 1.0], dtype=float)
                y_edges[0] = 0.0
                y_edges[-1] = 1.0
            else:
                y_edges = np.linspace(0.0, 1.0, max(int(nbins_y), 2) + 1)

            H, _, _ = np.histogram2d(xh, yh, bins=[x_edges, y_edges])
            H = H.T

            n_total = float(xh.size)
            if n_total > 0:
                H = H / n_total

            Hm = np.ma.masked_where(H == 0, H)

            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.set_facecolor("white")

            norm = None
            if Hm.count():
                pos = Hm.compressed()
                vmax = float(np.percentile(pos, float(vmax_percentile)))
                vmax = max(vmax, np.finfo(float).tiny)
                if log_counts:
                    norm = LogNorm(vmin=max(pos.min(), np.finfo(float).tiny), vmax=vmax)
                else:
                    norm = PowerNorm(gamma=float(count_gamma), vmin=0.0, vmax=vmax)

            ax.pcolormesh(
                x_edges,
                y_edges,
                Hm,
                shading="auto",
                cmap="Blues",
                norm=norm
            )

            ax.set_xlabel(set_xlabel, fontsize=14)
            ax.set_ylabel(set_ylabel, fontsize=14)
            ax.tick_params(axis="both", labelsize=14)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if plot_save_name == "CvV":
                y_zoom_hi = float(np.nanpercentile(yh, 95))
                y_zoom_hi = min(max(y_zoom_hi, 1e-3), 0.005)

                x_lo_ins = 0.0
                x_hi_ins = 0.1
                y_lo_ins = 0.0
                y_hi_ins = y_zoom_hi

                x_edges_ins = _make_edges(
                    x_lo_ins,
                    x_hi_ins,
                    nbins=nbins_x,
                    bin_width=x_bin_width,
                    default_nbins=19
                )
                y_edges_ins = np.linspace(y_lo_ins, y_hi_ins, 19 + 1)

                H_ins, _, _ = np.histogram2d(xh, yh, bins=[x_edges_ins, y_edges_ins])
                H_ins = H_ins.T
                if n_total > 0:
                    H_ins = H_ins / n_total

                Hm_ins = np.ma.masked_where(H_ins == 0, H_ins)

                axins = inset_axes(
                    ax,
                    width="45%",
                    height="45%",
                    loc="upper right",
                    borderpad=1.0,
                )
                axins.set_facecolor("white")
                axins.pcolormesh(
                    x_edges_ins,
                    y_edges_ins,
                    Hm_ins,
                    shading="auto",
                    cmap="Blues",
                    norm=norm,
                )
                axins.set_xlim([x_lo_ins, x_hi_ins])
                axins.set_ylim([y_lo_ins, y_hi_ins])
                axins.tick_params(axis="both", labelsize=9)
                axins.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                axins.yaxis.get_offset_text().set_size(9)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.35", lw=1.0)

        fig.tight_layout()

        if show:
            plt.show()
        else:
            if self.png_figures:
                fig.savefig(os.path.join(self.PATH_TO_DATA, plot_save_name + ".png"))
            if self.svg_figures:
                fig.savefig(os.path.join(self.PATH_TO_DATA, plot_save_name + ".svg"))
            plt.close(fig)

    def mobo_3d_pareto(
        self, df, save_name="mobo_3d_gen_comp_val",
        gen_col="generalizability_error", comp_col="nonzero_count", val_col="validation_error", 
        show=False
    ):
        if df is None or len(df) == 0:
            return

        x = df[gen_col].to_numpy(dtype=float) if gen_col in df.columns else None
        y = df[comp_col].to_numpy(dtype=float) if comp_col in df.columns else None
        z = df[val_col].to_numpy(dtype=float) if val_col in df.columns else None
        if x is None or y is None or z is None:
            return

        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(m):
            return
        x, y, z = x[m], y[m], z[m]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(x, y, z, s=18, alpha=0.45)

        ax.set_xlabel("Generalizability")
        ax.set_ylabel("Complexity (normalized)")
        ax.set_zlabel("Validation")
        ax.set_xlim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".png"))
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".svg"))
        elif show:
            plt.show()
        plt.close(fig)

    def mobo_hypervolume_trace_3obj(self, df, obj_cols, ref, save_name="hypervolume_trace_3obj", 
                                    sort_by_iteration=True, normalize=False, show=False):

        if df is None or len(df) == 0:
            return

        df2 = df
        if sort_by_iteration and "iteration" in df2.columns:
            df2 = df2.sort_values("iteration").reset_index(drop=True)

        Y = df2[list(obj_cols)].to_numpy(dtype=float)
        m = np.all(np.isfinite(Y), axis=1)
        Y = Y[m]
        if Y.shape[0] == 0:
            return

        if "iteration" in df2.columns:
            x_iter = df2["iteration"].to_numpy(dtype=int)[m]
        else:
            x_iter = np.arange(1, Y.shape[0] + 1, dtype=int)

        ref = np.asarray(ref, dtype=float).reshape(-1)
        if ref.size != Y.shape[1]:
            raise ValueError("ref dimension must match number of objectives")

        # Minimization hypervolume relative to ref via max(0, ref - y)
        U = ref[None, :] - Y
        U = np.clip(U, 0.0, None)

        hv = Hypervolume(ref_point=torch.zeros(U.shape[1], dtype=torch.double))
        hv_trace = np.empty(U.shape[0], dtype=float)

        for k in range(1, U.shape[0] + 1):
            Uk = torch.as_tensor(U[:k], dtype=torch.double)
            nd = is_non_dominated(Uk)
            pareto_Uk = Uk[nd]
            hv_k = hv.compute(pareto_Uk) if pareto_Uk.numel() > 0 else torch.tensor(0.0)
            hv_trace[k - 1] = float(hv_k)

        if normalize:
            denom = float(np.prod(ref))
            if denom > 0.0:
                hv_trace = hv_trace / denom

        x_line = np.concatenate(([0], x_iter))
        hv_line = np.concatenate(([0.0], hv_trace))

        import matplotlib.pyplot as plt
        import os

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(x_line, hv_line, linewidth=3.0)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Hypervolume")
        fig.tight_layout()

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".png"))
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".svg"))
        elif show:
            plt.show()
        plt.close(fig)

    def mobo_hypervolume_trace_2obj(self, df, obj1_col, obj2_col, ref=None, save_name="hypervolume", show=False):
        if df is None or len(df) == 0:
            return

        vals1 = df[obj1_col].to_numpy(dtype=float)
        vals2 = df[obj2_col].to_numpy(dtype=float)

        mask = np.isfinite(vals1) & np.isfinite(vals2)
        vals1 = vals1[mask]
        vals2 = vals2[mask]
        if vals1.size == 0:
            return

        if "iteration" in df.columns:
            x_iter_full = df["iteration"].to_numpy(dtype=int)
            x_iter = x_iter_full[mask]
        else:
            x_iter = np.arange(1, len(df) + 1, dtype=int)

        if ref is None:
            eps1 = 0.05 * (np.max(vals1) - np.min(vals1) + 1e-12)
            eps2 = 0.05 * (np.max(vals2) - np.min(vals2) + 1e-12)
            ref = np.array(
                [np.max(vals1) + eps1, np.max(vals2) + eps2],
                dtype=float,
            )
        else:
            ref = np.asarray(ref, float)

        hv_trace = []
        for k in range(1, len(vals1) + 1):
            points_k = np.c_[vals1[:k], vals2[:k]]
            hv_k = self._hypervolume_2d_min(points_k, ref)
            hv_trace.append(hv_k)
        hv_trace = np.asarray(hv_trace, dtype=float)

        hv_improv = np.empty_like(hv_trace)
        hv_improv[0] = hv_trace[0]
        hv_improv[1:] = hv_trace[1:] - hv_trace[:-1]

        x_line = np.concatenate(([0], x_iter))
        hv_line = np.concatenate(([0.0], hv_trace))

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(x_line, hv_line, linewidth=3.0, label="hypervolume")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Hypervolume")
        # ax.legend(frameon=False)
        fig.tight_layout()

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, f"{save_name}.png"), dpi=300)
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, f"{save_name}.svg"))
        elif show:
            plt.show()
        plt.close(fig)

    def parity_CI(
        self, y_true, y_pred,
        nbins=15, show_scatter=False,
        scatter_s=3, scatter_alpha=0.2,
        q_axis=(0.01, 0.99),
        save_name="train_parity",
        vmin=None, vmax=None,
        axis_source="true",
        equal_aspect=True,
        nticks=6, show=False
    ):
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        m = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[m]
        y_pred = y_pred[m]
        if y_true.size == 0:
            return

        if (vmin is None) or (vmax is None):
            lo_q, hi_q = q_axis
            y_lim_src = np.concatenate([y_true, y_pred]) if axis_source == "both" else y_true
            vmin, vmax = np.quantile(y_lim_src, [lo_q, hi_q])
            pad = 0.02 * (vmax - vmin + 1e-12)
            vmin -= pad
            vmax += pad

        q = np.quantile(y_true, np.linspace(0, 1, nbins + 1))
        q = np.unique(q)
        if q.size < 2:
            q = np.array([float(np.min(y_true)), float(np.max(y_true)) + 1e-12], dtype=float)

        xc, med, q25, q75 = [], [], [], []
        for a, b in zip(q[:-1], q[1:]):
            sel = (y_true >= a) & (y_true < b) if b < q[-1] else (y_true >= a) & (y_true <= b)
            if not sel.any():
                continue
            xc.append(0.5 * (a + b))
            yp = y_pred[sel]
            med.append(np.median(yp))
            q25.append(np.percentile(yp, 25))
            q75.append(np.percentile(yp, 75))

        if len(xc) == 0:
            return

        xc = np.asarray(xc, float)
        med = np.asarray(med, float)
        q25 = np.asarray(q25, float)
        q75 = np.asarray(q75, float)

        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)

        ax.plot([vmin, vmax], [vmin, vmax], "--", lw=1, color="0.3")

        if show_scatter:
            ax.scatter(y_true, y_pred, s=scatter_s, alpha=scatter_alpha)

        yerr = [med - q25, q75 - med]
        ax.errorbar(xc, med, yerr=yerr, fmt="o-", ms=3, lw=1)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")

        locator = MaxNLocator(nbins=int(nticks))
        ticks = locator.tick_values(vmin, vmax)
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.tick_params(axis="both", which="major", pad=2)

        ax.set_xlabel(r"PIC ion flux $\widehat{\Gamma}_i$ [m$^{-2}$ s$^{-1}$]", labelpad=8)
        ax.set_ylabel(r"Predicted ion flux $\Gamma_i$ [m$^{-2}$ s$^{-1}$]", labelpad=10)

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, f"{save_name}.png"), dpi=300)
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, f"{save_name}.svg"))
        elif show:
            plt.show()
        plt.close(fig)

    def _compute_vif_values(self, tr_Theta, std_tol=1e-12, cond_thresh=1e12,
                            ridge_fallback=1e-10, pinv_rcond=1e-12):
        N, L, d = tr_Theta.shape
        Xraw = tr_Theta.detach().cpu().numpy().reshape((N * L, d)).astype(np.float64)

        row_ok = np.all(np.isfinite(Xraw), axis=1)
        Xraw = Xraw[row_ok]
        if Xraw.shape[0] < 3:
            raise ValueError("Not enough finite rows for VIF calculation.")

        Xc = Xraw - np.mean(Xraw, axis=0)

        col_std = np.std(Xc, axis=0, ddof=0)
        const_mask = col_std <= std_tol
        var_mask = ~const_mask
        if np.sum(var_mask) < 2:
            raise ValueError("Not enough non-constant columns for VIF calculation.")

        X = Xc[:, var_mask] / col_std[var_mask]
        n = X.shape[0]
        C = (X.T @ X) / max(n - 1, 1)

        ridge = 0.0
        try:
            cond = np.linalg.cond(C)
            if not np.isfinite(cond) or cond > cond_thresh:
                raise np.linalg.LinAlgError
            P = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            ridge = ridge_fallback
            P = np.linalg.pinv(C + ridge * np.eye(C.shape[0]), rcond=pinv_rcond)

        vif_nc = np.diag(np.asarray(P)).astype(np.float64, copy=True)
        vif_nc[~np.isfinite(vif_nc)] = np.inf
        vif_nc[vif_nc < 1.0] = 1.0

        vif_vals = np.empty(d, dtype=float)
        vif_vals[const_mask] = np.inf
        vif_vals[var_mask] = vif_nc

        return dict(
            vif_vals=vif_vals,
            const_mask=const_mask,
            var_mask=var_mask,
            ridge=ridge,
            C=C,
            valid_idx=np.where(var_mask)[0],
        )

    def vif(self, tr_Theta, syms, high=False, n_keep=10, show=False,
            plot_vif=False, plot_corr=False, stage="", lollipop_plot=False):

        out = self._compute_vif_values(tr_Theta)

        vif_vals = out["vif_vals"]
        C = out["C"]
        valid_idx = out["valid_idx"]

        order = np.argsort(vif_vals[valid_idx])
        if high:
            order = order[::-1]
        idx_sorted = valid_idx[order]

        if (n_keep is None) or (n_keep <= 0) or (n_keep >= idx_sorted.size):
            keep_indices = idx_sorted
        else:
            keep_indices = idx_sorted[:n_keep]

        vif_kept = vif_vals[keep_indices]
        symbols_kept = [fr"${sp.latex(syms[i])}$" for i in keep_indices]
        if lollipop_plot:
            return dict(
                keep_indices=keep_indices,
                vif_kept=vif_kept,
                symbols_kept=symbols_kept,
                stage=stage,
                high=high,
                n_keep=n_keep,
                out=out,
            )
        if plot_vif:
            if stage=="Physics-based": color = "#0072B2"
            else: color = "#D55E00"

            plt.figure(figsize=(4, 6))
            ax = plt.gca()
            ax.barh(symbols_kept, vif_kept, color=color)
            ax.invert_yaxis()
            formatter = ScalarFormatter()
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            ax.xaxis.set_major_formatter(formatter)
            plt.xlabel("VIF")
            # plt.title(f"{stage}: Top {len(keep_indices)} {vif_type} VIF predictors{title_ridge}")
            plt.tight_layout()

            if self.png_figures:
                plt.savefig(os.path.join(self.PATH_TO_DATA, f"{stage}_VIF.png"))
                plt.close()
            if self.svg_figures:
                plt.savefig(os.path.join(self.PATH_TO_DATA,  f"{stage}_VIF.svg"))
                plt.close()
            else:
                plt.show()

        if plot_corr:
            pos_map = {j: k for k, j in enumerate(valid_idx)}
            keep_pos = [pos_map[j] for j in keep_indices if j in pos_map]
            if len(keep_pos) > 1:
                C_top = C[np.ix_(keep_pos, keep_pos)]
                top_labels = [fr"${sp.latex(syms[j])}$" for j in keep_indices if j in pos_map]
                df_corr = pd.DataFrame(C_top, index=top_labels, columns=top_labels)

                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(
                    df_corr, annot=True, fmt=".2f",
                    cmap="RdBu_r", vmin=-1, vmax=1,
                    cbar_kws={"label": "corr"}
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
                # plt.title(f"{stage}: Correlation Matrix for Top {len(keep_pos)} {vif_type} VIF predictors")
                plt.tight_layout()

                if self.png_figures:
                    plt.savefig(os.path.join(self.PATH_TO_DATA, f"{stage}_corr_matrix.png"))
                    plt.close()
                if self.svg_figures:
                    plt.savefig(os.path.join(self.PATH_TO_DATA, f"{stage}_corr_matrix.svg"))
                    plt.close()
                else:
                    if show:
                        plt.show()

        return out

    def vif_rank_lollipop(self, adhoc_Theta, adhoc_syms, phys_Theta, phys_syms,
                        n_keep=10, stage_adhoc="Ad-hoc", stage_phys="Physics-based",
                        pair_offset=0.18, rank_spacing=1.0,
                        label_xmult=1.06, label_fontsize=9,
                        save_name="VIF_rank_lollipop", show=True,
                        alpha=0.45):

        ra = self.vif(
            adhoc_Theta, adhoc_syms,
            high=False, n_keep=n_keep,
            plot_vif=False, plot_corr=False,
            stage=stage_adhoc,
            lollipop_plot=True,
        )
        rp = self.vif(
            phys_Theta, phys_syms,
            high=False, n_keep=n_keep,
            plot_vif=False, plot_corr=False,
            stage=stage_phys,
            lollipop_plot=True,
        )

        ta = list(ra["symbols_kept"])
        va = np.asarray(ra["vif_kept"], dtype=float)
        tp = list(rp["symbols_kept"])
        vp = np.asarray(rp["vif_kept"], dtype=float)

        ma = np.isfinite(va) & (va > 0.0)
        mp = np.isfinite(vp) & (vp > 0.0)
        ta = [t for t, m in zip(ta, ma) if m]
        va = va[ma]
        tp = [t for t, m in zip(tp, mp) if m]
        vp = vp[mp]

        k = int(min(n_keep, len(va), len(vp)))
        if k <= 0:
            return None

        ta = ta[:k]
        va = va[:k]
        tp = tp[:k]
        vp = vp[:k]

        all_v = np.concatenate([va, vp])
        vmin = float(np.min(all_v))
        vmax = float(np.max(all_v))

        x0 = max(vmin / 2.0, 1e-12)
        x1 = vmax * max(1.4, label_xmult * 1.15)

        y_base = np.arange(k, dtype=float)[::-1] * rank_spacing
        y_phys = y_base + pair_offset
        y_adhoc = y_base - pair_offset

        c_phys = "#0072B2"
        c_adhoc = "#D55E00"

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xscale("log")
        ax.set_xlim(x0, x1)

        ax.hlines(y_phys, x0, vp, linewidth=4.0, color=c_phys, alpha=alpha)
        ax.scatter(vp, y_phys, s=75, color=c_phys, zorder=3)

        ax.hlines(y_adhoc, x0, va, linewidth=4.0, color=c_adhoc, alpha=alpha)
        ax.scatter(va, y_adhoc, s=75, color=c_adhoc, zorder=3)

        for i in range(k):
            ax.text(
                vp[i] * label_xmult, y_phys[i], tp[i],
                ha="left", va="center", fontsize=label_fontsize, color="0.15"
            )
            ax.text(
                va[i] * label_xmult, y_adhoc[i], ta[i],
                ha="left", va="center", fontsize=label_fontsize, color="0.15"
            )

        ax.set_yticks([])
        ax.set_xlabel(r"$\log_{10}(\mathrm{VIF})$")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # ax.axvline(x0, linewidth=1.0, color="0.2")

        y_lo = float(np.min(y_adhoc) - 0.6 * rank_spacing)
        y_hi = float(np.max(y_phys) + 0.6 * rank_spacing)
        ax.set_ylim(y_lo, y_hi)
        fig.tight_layout()

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".png"))
            plt.close(fig)
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".svg"))
            plt.close(fig)
        else:
            if show:
                plt.show()
            else:
                plt.close(fig)

        return dict(adhoc=ra, physics=rp, k=k, xlim=(x0, x1))

    def plot_vif_heatmap(
        self, df_full=None, df_opt=None, mode="both", tag_to_model=None,
        model_order=None, save_name="vif_heatmap", show=False
    ):
        mode = str(mode).lower()
        pv_parts = []

        if mode in ["full", "both"]:
            df_f = df_full.copy()
            df_f["term"] = df_f["term"].map(self._ensure_math)
            df_f["model"] = "full"

            pv_f = df_f.pivot_table(
                index="term",
                columns="model",
                values="vif",
                aggfunc="first",
            )
            pv_parts.append(pv_f)

        if mode in ["optimized", "both"]:
            df_s = df_opt.copy()
            df_s["term"] = df_s["term"].map(self._ensure_math)
            df_s["model"] = df_s["tag"].map(tag_to_model).fillna(df_s["tag"])

            pv_s = df_s.pivot_table(
                index="term",
                columns="model",
                values="vif",
                aggfunc="first",
            )
            pv_parts.append(pv_s)

        if len(pv_parts) == 1:
            pv = pv_parts[0]
        else:
            pv = pv_parts[0].join(pv_parts[1], how="outer")

        col_order = [c for c in model_order if c in pv.columns]
        pv = pv[col_order]

        logv = np.log10(pv)

        freq = pv.notna().sum(axis=1)
        minlog = logv.min(axis=1)
        order = np.lexsort((minlog.to_numpy(), (-freq).to_numpy()))
        logv = logv.iloc[order]

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="0.9")

        data = np.ma.masked_invalid(logv.to_numpy())
        n_terms, n_models = data.shape

        cell_in = 0.28
        left_in = 2.6
        right_in = 1.1
        top_in = 0.6
        bottom_in = 0.9

        fig_w = cell_in * n_models + left_in + right_in
        fig_h = cell_in * n_terms + top_in + bottom_in
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        x_edges = np.arange(n_models + 1, dtype=float)
        y_edges = np.arange(n_terms + 1, dtype=float)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            data,
            cmap=cmap,
            shading="flat",
            edgecolors="1.0",
            linewidth=0.6,
            antialiased=False,
            rasterized=True,
        )

        ax.set_aspect("equal")
        ax.set_xlim(0.0, float(n_models))
        ax.set_ylim(float(n_terms), 0.0)

        ax.set_xlabel("Model")
        ax.set_ylabel("Term")

        ax.set_xticks(np.arange(n_models) + 0.5)
        ax.set_xticklabels(logv.columns)

        ax.set_yticks(np.arange(n_terms) + 0.5)
        ax.set_yticklabels(logv.index)

        if mode == "both" and "full" in logv.columns and n_models > 1:
            full_idx = list(logv.columns).index("full")
            ax.vlines(float(full_idx + 1), 0.0, float(n_terms), colors="0.2", linewidth=1.2)

        fig.colorbar(mesh, ax=ax, pad=0.02, label=r"$\log_{10}(\mathrm{VIF})$")
        fig.tight_layout()

        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".png"), dpi=300)
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, save_name + ".svg"))

        if show:
            plt.show()

        plt.close(fig)
        return fig, ax, logv

    def knn(
        self, pp, ff, Perf, pf_ranges, p_all, f_all, window="window",
        vmin=None, vmax=None, train_mask=None, test_mask=None, val_mask=None, show=False
    ):
        p_all = np.asarray(p_all, float)
        f_all = np.asarray(f_all, float)
        pp = np.asarray(pp, float)
        ff = np.asarray(ff, float)
        Perf_arr = np.asarray(Perf, float)

        pp_plot = pp * p.mean_pressure
        ff_plot = ff * p.mean_frequency
        p_all_plot = p_all * p.mean_pressure
        f_all_plot = f_all * p.mean_frequency

        finite = np.isfinite(Perf_arr)
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
            if np.any(finite):
                vmax = max(vmax, float(np.nanmax(Perf_arr[finite])))

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.pcolormesh(
            pp_plot, ff_plot, Perf_arr,
            shading="auto",
            cmap="viridis_r",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )

        cbar_label = "Normalized Squared Error"
        fig.colorbar(im, label=cbar_label)

        ax.set_xlabel("pressure [mTorr]")
        ax.set_ylabel("frequency [MHz]")

        if window in pf_ranges:
            pmin, pmax, fmin, fmax = pf_ranges[window]
            xs = np.array([pmin, pmax, pmax, pmin, pmin], float) * p.mean_pressure
            ys = np.array([fmin, fmin, fmax, fmax, fmin], float) * p.mean_frequency
            ax.plot(xs, ys, "k-", lw=1, alpha=0.6)

        # faint background PF lattice
        ax.scatter(p_all_plot, f_all_plot, s=10, alpha=0.15)

        n_pf = p_all.shape[0]

        train_mask_arr = None
        if train_mask is not None:
            train_mask_arr = np.asarray(train_mask, bool)
            if train_mask_arr.shape[0] != n_pf:
                train_mask_arr = None

        val_mask_arr = None
        if val_mask is not None:
            val_mask_arr = np.asarray(val_mask, bool)
            if val_mask_arr.shape[0] != n_pf:
                val_mask_arr = None

        test_mask_arr = None
        if test_mask is not None:
            test_mask_arr = np.asarray(test_mask, bool)
            if test_mask_arr.shape[0] != n_pf:
                test_mask_arr = None

        pad = 0.05
        x0, x1 = float(np.nanmin(p_all_plot)), float(np.nanmax(p_all_plot))
        y0, y1 = float(np.nanmin(f_all_plot)), float(np.nanmax(f_all_plot))
        dx = max(x1 - x0, 1e-12)
        dy = max(y1 - y0, 1e-12)
        ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
        ax.set_ylim(y0 - pad * dy, y1 + pad * dy)

        fig.tight_layout()
        if self.png_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, "performance.png"))
        if self.svg_figures:
            fig.savefig(os.path.join(self.PATH_TO_DATA, "performance.svg"))
        elif show:
            plt.show()
        plt.close(fig)
