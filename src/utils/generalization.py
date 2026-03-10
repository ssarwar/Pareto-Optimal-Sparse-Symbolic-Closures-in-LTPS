import numpy as np
import torch
from sympy import srepr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import utils.library as Lib
import utils.constants as Consts

p = Consts.PhysicalConstants()
o = Consts.OtherConstants()
tkwargs = {"dtype": o.dtype, "device": o.device}
lib = Lib.Library(**tkwargs)


class KNN_Generalizability:
    def __init__(self):
        pass
    def _sample_metrics(self, y_hat_t, y_true_t, eps_abs=1e-10, eps_rel=1e-3):
        y_hat = y_hat_t.detach().cpu().numpy()
        y_true = y_true_t.detach().cpu().numpy()

        rmse = np.sqrt(np.nanmean((y_hat - y_true) ** 2, axis=1))

        std_y = np.nanstd(y_true, axis=1)
        if np.any(np.isfinite(std_y)):
            ref = float(np.nanmedian(std_y[np.isfinite(std_y)]))
        else:
            ref = 0.0
        var_ok = std_y > max(eps_abs, eps_rel * max(ref, 1e-12))

        mu = np.nanmean(y_true, axis=1, keepdims=True)
        sst = np.nansum((y_true - mu) ** 2, axis=1)
        sse = np.nansum((y_true - y_hat) ** 2, axis=1)

        r2 = np.full_like(sst, np.nan, dtype=float)
        good = var_ok & np.isfinite(sse) & np.isfinite(sst)
        denom = np.maximum(sst[good], 1e-12)
        r2[good] = 1.0 - sse[good] / denom

        y_min = np.nanmin(y_true, axis=1)
        y_max = np.nanmax(y_true, axis=1)
        range_y = y_max - y_min
        if np.any(np.isfinite(range_y)):
            ref_range = float(np.nanmedian(range_y[np.isfinite(range_y)]))
        else:
            ref_range = 0.0
        denom_range = np.maximum(
            range_y,
            max(eps_abs, eps_rel * max(ref_range, 1e-12)),
        )
        nrmse = rmse / (denom_range + 1e-12)
        nrmse = np.clip(nrmse, 0.0, 1.0)

        return r2, nrmse, rmse, sse, sst, var_ok
    
    def performance_map_continuous(self, p_all, f_all, performance_all, in_mask,
                                   grid_res=401, k=64, k_train=32, alpha=1.0, stat="mean"):
        p_all = np.asarray(p_all, float)
        f_all = np.asarray(f_all, float)
        performance_all = np.asarray(performance_all, float)
        in_mask = np.asarray(in_mask, bool)

        PF = np.c_[p_all, f_all].astype(np.float64, copy=False)
        
        pmin, pmax = float(np.min(p_all)), float(np.max(p_all))
        fmin, fmax = float(np.min(f_all)), float(np.max(f_all))
        pp, ff = np.meshgrid(
            np.linspace(pmin, pmax, grid_res),
            np.linspace(fmin, fmax, grid_res)
        )

        if not np.any(in_mask):
            Perf = np.full_like(pp, np.nan, dtype=float)
            return Perf, None, (pp, ff)

        scaler = StandardScaler().fit(PF[in_mask])

        finite = np.isfinite(performance_all)
        if not np.any(finite):
            raise ValueError("No finite R² values to smooth.")

        Z_all = scaler.transform(PF[finite])
        performance_fin = performance_all[finite]

        Zg = scaler.transform(
            np.c_[pp.ravel(), ff.ravel()].astype(np.float64, copy=False)
            )

        Z_train = scaler.transform(PF[in_mask])
        k_train_eff = int(min(max(1, k_train), Z_train.shape[0]))
        nn_train = NearestNeighbors(n_neighbors=k_train_eff).fit(Z_train)
        d_tr, _ = nn_train.kneighbors(Zg, return_distance=True)
        bw = alpha * np.maximum(d_tr[:, -1], 1e-12)

        k_eff = int(min(max(1, k), Z_all.shape[0]))
        nn_all = NearestNeighbors(n_neighbors=k_eff).fit(Z_all)
        d_all, idxs = nn_all.kneighbors(Zg, return_distance=True)

        w = np.exp(-0.5 * (d_all / bw[:, None]) ** 2)
        w_sum = np.sum(w, axis=1, keepdims=True) + 1e-12

        if stat == "mean":
            perf_vec = np.sum(w * performance_fin[idxs], axis=1) / w_sum.ravel()
        elif stat == "median":
            perf_vec = np.array(
                [self._weighted_quantile(performance_fin[idxs[i]], w[i], 0.5) 
                 for i in range(Zg.shape[0])],
                dtype=np.float64
            )
        else:
            raise ValueError("stat must be 'mean' or 'median'")

        Perf = perf_vec.reshape(pp.shape)
        return Perf, scaler, (pp, ff)

    def _weighted_quantile(self, x, w, q):
        ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if not np.any(ok):
            return np.nan
        x = x[ok]
        w = w[ok]
        order = np.argsort(x)
        x = x[order]
        w = w[order]
        cw = np.cumsum(w)
        t = q * cw[-1]
        idx = np.searchsorted(cw, t, side="right")
        idx = np.clip(idx, 0, len(x) - 1)
        return x[idx]

