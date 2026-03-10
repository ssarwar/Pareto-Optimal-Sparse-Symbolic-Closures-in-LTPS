import argparse
import csv
import hashlib
import json as _json
import json
import os

import numpy as np
import torch

import utils.constants as Consts
import utils.plotting as Pl
import utils.library as Lib
from utils.prune import StatPrune
from utils.generalization import KNN_Generalizability
from utils.post_processing import post_processing
from utils.run_opt import run_opt


def _to_numpy_int(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(int)
    return np.asarray(x, dtype=int)


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _select_columns(X, cols, device):
    if torch.is_tensor(X):
        cols_t = torch.as_tensor(cols, device=device, dtype=torch.long)
        return X.index_select(1, cols_t)
    return X[:, cols]


def _read_summary_trials(summary_csv_path):
    with open(summary_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        tags = []
        for row in reader:
            r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in row.items()}
            tag = r.get("tag", "")
            if tag != "":
                tags.append(tag)
                continue

            trial_num = r.get("trial num", "")
            tag_hash = r.get("tag hash", "")
            if trial_num == "" or tag_hash == "":
                continue
            trial_name = f"{str(int(float(trial_num)))}_{tag_hash}"
            tags.append(trial_name)
    return tags


def _find_trial_dir(exp_dir, trial_name):
    cand = os.path.join(exp_dir, trial_name)
    if os.path.isdir(cand):
        return cand

    if "_" in trial_name:
        a, b = trial_name.split("_", 1)
        if a.isdigit():
            cand2 = os.path.join(exp_dir, f"{int(a)}_{b}")
            if os.path.isdir(cand2):
                return cand2

    raise FileNotFoundError(f"Could not find trial directory for {trial_name} under {exp_dir}")


def _sst_floor_from_flux(flux_t, mask, q=0.05, eps=1e-12):
    y = flux_t.detach().cpu().numpy().astype(np.float64)
    mask = np.asarray(mask, bool)
    y = y[mask]
    if y.size == 0:
        return eps
    mu = np.mean(y, axis=1, keepdims=True)
    sst = np.sum((y - mu) ** 2, axis=1)
    sst = sst[np.isfinite(sst) & (sst > 0.0)]
    if sst.size == 0:
        return eps
    return float(max(np.quantile(sst, q), eps))


def scale_err_array(x, invalid_value=1.0):
    x = np.asarray(x, float)
    out = np.full_like(x, invalid_value, dtype=float)
    mask = np.isfinite(x) & (x >= 0.0)
    out[mask] = x[mask] / (1.0 + x[mask])
    return out


def masked_mean_and_sem(perf_field, mask, empty_value=1.0, sem_default=0.5, sem_floor=1e-3):
    v = np.asarray(perf_field, float)[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    k = v.size
    if k == 0:
        return float(empty_value), float(sem_default)
    mu = float(np.mean(v))
    if k == 1:
        return mu, float(sem_default)
    sem = float(np.std(v, ddof=1) / np.sqrt(k))
    return mu, float(max(sem, sem_floor))


def _clip01(x, fallback=1.0):
    x = float(x)
    if not np.isfinite(x):
        return float(fallback)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _safe_sem(x, fallback=0.5, floor=1e-3):
    x = float(x)
    if not np.isfinite(x):
        return float(fallback)
    return float(max(x, floor))


def complexity_norm_from_nonzero(raw_nonzero, d_full):
    if int(raw_nonzero) <= 0:
        return 1.0
    return float(raw_nonzero) / float(d_full)


def _window_seed(p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff):
    split_hash = hashlib.sha1(
        _json.dumps(
            {
                "p_lo": float(p_lo_eff),
                "p_hi": float(p_hi_eff),
                "f_lo": float(f_lo_eff),
                "f_hi": float(f_hi_eff),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:8]
    return int(split_hash, 16) % (2**32), split_hash


def select_samples_by_window(
    P_all, F_all,
    p_min, p_max, f_min, f_max,
    p_half_min, f_half_min,
    min_train, min_val,
    p_lo, p_hi, f_lo, f_hi,
):
    M_total = int(P_all.size)
    min_window = max(min_train + min_val, 6)

    p_lo_eff = max(float(p_lo), float(p_min))
    p_hi_eff = min(float(p_hi), float(p_max))
    f_lo_eff = max(float(f_lo), float(f_min))
    f_hi_eff = min(float(f_hi), float(f_max))

    in_window = (
        (P_all >= p_lo_eff) & (P_all <= p_hi_eff)
        & (F_all >= f_lo_eff) & (F_all <= f_hi_eff)
    )
    cand_idx = np.where(in_window)[0]

    if cand_idx.size < min_window:
        p_c = 0.5 * (p_lo_eff + p_hi_eff)
        f_c = 0.5 * (f_lo_eff + f_hi_eff)

        p_lo2 = p_lo_eff
        p_hi2 = p_hi_eff
        f_lo2 = f_lo_eff
        f_hi2 = f_hi_eff

        for _ in range(M_total):
            if cand_idx.size >= min_window:
                break

            p_lo2 = max(p_min, p_lo2 - p_half_min)
            p_hi2 = min(p_max, p_hi2 + p_half_min)
            f_lo2 = max(f_min, f_lo2 - f_half_min)
            f_hi2 = min(f_max, f_hi2 + f_half_min)

            in_window = (
                (P_all >= p_lo2) & (P_all <= p_hi2)
                & (F_all >= f_lo2) & (F_all <= f_hi2)
            )
            cand_idx = np.where(in_window)[0]

            if (p_lo2 == p_min) and (p_hi2 == p_max) and (f_lo2 == f_min) and (f_hi2 == f_max):
                break

        p_lo_eff = float(p_lo2)
        p_hi_eff = float(p_hi2)
        f_lo_eff = float(f_lo2)
        f_hi_eff = float(f_hi2)

        if cand_idx.size < min_window:
            dp = (P_all - p_c) / (p_max - p_min + 1e-12)
            df = (F_all - f_c) / (f_max - f_min + 1e-12)
            dist2 = dp * dp + df * df
            k_need = int(min(min_window, M_total))
            cand_idx = np.argsort(dist2)[:k_need].astype(int)

        if cand_idx.size < min_window:
            raise RuntimeError(
                f"Window selection produced {cand_idx.size} samples, need at least {min_window}."
            )

    return np.unique(cand_idx.astype(int)), p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff


def _fallback_split_indices_from_metrics(
    m, P_all, F_all,
    p_min, p_max, f_min, f_max,
    p_half_min, f_half_min,
    min_train, min_val,
):
    if all(k in m for k in ["p_lo_eff", "p_hi_eff", "f_lo_eff", "f_hi_eff"]):
        p_lo_eff = float(m["p_lo_eff"])
        p_hi_eff = float(m["p_hi_eff"])
        f_lo_eff = float(m["f_lo_eff"])
        f_hi_eff = float(m["f_hi_eff"])
    else:
        if all(k in m for k in ["p_c", "p_hw", "f_c", "f_hw"]):
            p_c = float(m["p_c"])
            p_hw = float(m["p_hw"])
            f_c = float(m["f_c"])
            f_hw = float(m["f_hw"])
            p_lo = p_c - p_hw
            p_hi = p_c + p_hw
            f_lo = f_c - f_hw
            f_hi = f_c + f_hw
        elif all(k in m for k in ["p_lo", "p_hi", "f_lo", "f_hi"]):
            p_lo = float(m["p_lo"])
            p_hi = float(m["p_hi"])
            f_lo = float(m["f_lo"])
            f_hi = float(m["f_hi"])
        else:
            raise RuntimeError("Cannot reconstruct window parameters from metrics.json")

        window_m, p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff = select_samples_by_window(
            P_all=P_all,
            F_all=F_all,
            p_min=p_min,
            p_max=p_max,
            f_min=f_min,
            f_max=f_max,
            p_half_min=p_half_min,
            f_half_min=f_half_min,
            min_train=min_train,
            min_val=min_val,
            p_lo=p_lo,
            p_hi=p_hi,
            f_lo=f_lo,
            f_hi=f_hi,
        )
        _ = window_m

    in_window_mask = (
        (P_all >= p_lo_eff) & (P_all <= p_hi_eff)
        & (F_all >= f_lo_eff) & (F_all <= f_hi_eff)
    )
    gen_mask = ~in_window_mask
    gen_m = np.where(gen_mask)[0].astype(int)

    window_m = np.where(in_window_mask)[0].astype(int)
    seed, split_hash8 = _window_seed(p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(int(window_m.size))

    k = int(window_m.size)
    if k >= (min_train + min_val):
        n_val_target = int(np.round(0.1 * k))
        n_val = max(min_val, n_val_target)
        n_val = min(n_val, k - min_train)
    else:
        n_val = max(0, k - min_train)

    val_m = window_m[perm[:n_val]] if n_val > 0 else np.asarray([], dtype=int)
    train_m = window_m[perm[n_val:]] if n_val > 0 else window_m

    return train_m, val_m, gen_m, p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff, split_hash8


def run_one_trial(
    trial_name, exp_dir,
    Theta, symbols_list, flux,
    P_all, F_all,
    p_min, p_max, f_min, f_max,
    p_half_min, f_half_min,
    min_train, min_val,
    SST_FLOOR,
    initial_coef_full,
    scaling_consts,
    show_knn=True,
    show_parity=True,
    write_json=False,
):
    trial_dir = _find_trial_dir(exp_dir, trial_name)
    metrics_path = os.path.join(trial_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")

    with open(metrics_path, "r") as f:
        m = json.load(f)

    o = Consts.OtherConstants()
    tkwargs = {"dtype": o.dtype, "device": o.device}
    lib = Lib.Library(**tkwargs)

    lmb1 = float(m.get("lmb1", np.nan))
    lmb2 = float(m.get("lmb2", np.nan))

    if "train_idx" in m and "val_idx" in m and "gen_idx" in m:
        train_m = _to_numpy_int(m["train_idx"])
        val_m = _to_numpy_int(m["val_idx"])
        gen_m = _to_numpy_int(m["gen_idx"])

        if all(k in m for k in ["p_lo_eff", "p_hi_eff", "f_lo_eff", "f_hi_eff"]):
            p_lo_eff = float(m["p_lo_eff"])
            p_hi_eff = float(m["p_hi_eff"])
            f_lo_eff = float(m["f_lo_eff"])
            f_hi_eff = float(m["f_hi_eff"])
        else:
            in_window_mask = np.zeros(P_all.size, dtype=bool)
            in_window_mask[train_m] = True
            in_window_mask[val_m] = True
            p_lo_eff = float(P_all[in_window_mask].min())
            p_hi_eff = float(P_all[in_window_mask].max())
            f_lo_eff = float(F_all[in_window_mask].min())
            f_hi_eff = float(F_all[in_window_mask].max())

        split_hash8 = str(m.get("window_split_hash8", ""))
    else:
        train_m, val_m, gen_m, p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff, split_hash8 = _fallback_split_indices_from_metrics(
            m=m,
            P_all=P_all,
            F_all=F_all,
            p_min=p_min,
            p_max=p_max,
            f_min=f_min,
            f_max=f_max,
            p_half_min=p_half_min,
            f_half_min=f_half_min,
            min_train=min_train,
            min_val=min_val,
        )

    M_total = int(Theta.shape[0])
    train_mask = np.zeros(M_total, dtype=bool)
    val_mask = np.zeros(M_total, dtype=bool)
    gen_mask = np.zeros(M_total, dtype=bool)

    train_mask[train_m] = True
    if val_m.size > 0:
        val_mask[val_m] = True
    gen_mask[gen_m] = True

    if np.any(train_mask & val_mask) or np.any(train_mask & gen_mask) or np.any(val_mask & gen_mask):
        raise RuntimeError("train, val, gen masks overlap")

    pf_ranges = {"window": (p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff)}

    train_idx_t = torch.as_tensor(train_m, device=o.device, dtype=torch.long)
    Theta_tr = Theta.index_select(0, train_idx_t)
    flux_tr = flux.index_select(0, train_idx_t)
    is_bulk_tr = np.full(Theta_tr.shape[0:2], True, dtype=bool)

    X_tr, y_tr, rid_tr, sid_tr, jid_tr = lib.region_flatten(Theta_tr, is_bulk_tr, flux_tr)
    f_train_full = lib.train_(X_tr, y_tr, rid_tr, sid_tr, jid_tr, Theta_tr.shape[1])

    keep_cols = None
    if "keep_cols" in m:
        keep_cols = _to_numpy_int(m["keep_cols"])
    if keep_cols is None or keep_cols.size == 0:
        pruner = StatPrune(
            topk_remove=int(0.3 * len(symbols_list)),
            pruning_lmb=1e-6,
            verbose=False,
        )
        f_Xb_opt, _, keep_cols = pruner.run_prune(
            f_train_full=f_train_full,
            f_Xb=X_tr,
            f_symbols_list_full=symbols_list,
        )
        keep_cols = _to_numpy_int(keep_cols)
    else:
        f_Xb_opt = _select_columns(X_tr, keep_cols, device=o.device)

    f_train = lib.train_(f_Xb_opt, y_tr, rid_tr, sid_tr, jid_tr, Theta_tr.shape[1])
    f_symbols_list = [symbols_list[i] for i in keep_cols.tolist()]
    f_initial_coef = initial_coef_full[keep_cols] if keep_cols.size > 0 else np.zeros((0,), dtype=float)

    f_coef_star_std = None
    if "coefficients" in m and m["coefficients"] is not None:
        f_coef_star_std = np.asarray(m["coefficients"], dtype=float)
    if f_coef_star_std is None or f_coef_star_std.size != keep_cols.size:
        if keep_cols.size == 0:
            f_coef_star_std = np.zeros((0,), dtype=float)
        else:
            f_coef_star_std = np.asarray(run_opt(f_train, lmb1, lmb2, f_initial_coef), dtype=float)

    raw_nonzero = 0
    equation = ""
    coef_q_orig = np.zeros((keep_cols.size,), dtype=float)

    if keep_cols.size == 0:
        raw_nonzero = 0
        equation = ""
        coef_q_orig = np.zeros((0,), dtype=float)
        f_coef_star_std = np.zeros((0,), dtype=float)
    else:
        if not np.all(np.isfinite(f_coef_star_std)):
            raw_nonzero = 0
            equation = ""
            coef_q_orig = np.zeros((keep_cols.size,), dtype=float)
            f_coef_star_std = np.zeros((keep_cols.size,), dtype=float)
        else:
            flux_pred = post_processing(
                f_coef_star_std,
                f_symbols_list,
                f_train,
                thr_std=o.max_coef_thr,
                lib_shape=(Theta_tr.shape[0], Theta_tr.shape[1], int(len(keep_cols))),
            )
            raw_nonzero = int(flux_pred.get("nonzero_count", 0))
            equation = str(flux_pred.get("equation", ""))

            coef_q_orig_t = flux_pred.get("nondim_coef", None)
            if coef_q_orig_t is None:
                coef_q_orig = np.zeros((keep_cols.size,), dtype=float)
            else:
                if torch.is_tensor(coef_q_orig_t):
                    coef_q_orig = coef_q_orig_t.detach().cpu().numpy().astype(float).reshape(-1)
                else:
                    coef_q_orig = np.asarray(coef_q_orig_t, dtype=float).reshape(-1)

            if coef_q_orig.size != keep_cols.size:
                coef_q_orig = np.zeros((keep_cols.size,), dtype=float)

    KNN = KNN_Generalizability()

    keep_t = torch.as_tensor(
        np.asarray(keep_cols, dtype=np.int64),
        device=o.device,
        dtype=torch.long,
    )
    Theta_sel_all = Theta.index_select(dim=2, index=keep_t)

    coef_t = torch.as_tensor(np.asarray(coef_q_orig, dtype=float), **tkwargs).view(-1, 1)
    y_hat_all_t = (Theta_sel_all @ coef_t).squeeze(-1)

    _, _, _, sse_all, sst_all, var_ok = KNN._sample_metrics(y_hat_all_t, flux)

    sse_all = np.asarray(sse_all, float)
    sst_all = np.asarray(sst_all, float)
    var_ok = np.asarray(var_ok, bool)

    denom_all = np.maximum(sst_all, SST_FLOOR)
    err_ratio_all = np.full_like(sse_all, np.nan, dtype=float)

    good_err = var_ok & np.isfinite(sse_all) & np.isfinite(denom_all) & (denom_all > 0.0)
    err_ratio_all[good_err] = sse_all[good_err] / denom_all[good_err]

    perf_field = scale_err_array(err_ratio_all, invalid_value=1.0)

    validation_error, val_sem = masked_mean_and_sem(perf_field, val_mask, empty_value=1.0, sem_default=0.5)
    generalizability_error, gen_sem = masked_mean_and_sem(perf_field, gen_mask, empty_value=1.0, sem_default=0.5)

    validation_error = _clip01(validation_error, fallback=1.0)
    generalizability_error = _clip01(generalizability_error, fallback=1.0)
    val_sem = _safe_sem(val_sem, fallback=0.5)
    gen_sem = _safe_sem(gen_sem, fallback=0.5)

    complexity_norm = complexity_norm_from_nonzero(raw_nonzero, len(symbols_list))
    complexity_norm = _clip01(complexity_norm, fallback=1.0)

    Perf, _, (pp, ff) = KNN.performance_map_continuous(
        p_all=P_all,
        f_all=F_all,
        performance_all=perf_field,
        in_mask=train_mask,
    )

    degenerate = (raw_nonzero <= 0)
    if degenerate:
        generalizability_error = 1.0
        validation_error = 1.0
        gen_sem = 0.5
        val_sem = 0.5
        complexity_norm = 1.0

    rerun_metrics = {
        "trial_name": str(trial_name),
        "lmb1": float(lmb1),
        "lmb2": float(lmb2),
        "equation": str(equation),
        "nonzero_count": int(raw_nonzero),
        "complexity_norm": float(complexity_norm),
        "generalizability_error": float(generalizability_error),
        "validation_error": float(validation_error),
        "val_sem": float(val_sem),
        "gen_sem": float(gen_sem),
        "window_split_hash8": str(split_hash8),
        "p_lo_eff": float(p_lo_eff),
        "p_hi_eff": float(p_hi_eff),
        "f_lo_eff": float(f_lo_eff),
        "f_hi_eff": float(f_hi_eff),
        "train_idx": _to_numpy_int(train_m).tolist(),
        "val_idx": _to_numpy_int(val_m).tolist(),
        "gen_idx": _to_numpy_int(gen_m).tolist(),
        "keep_cols": _to_numpy_int(keep_cols).tolist(),
        "coefficients": np.asarray(f_coef_star_std, float).tolist(),
    }
    if write_json:
        _write_json(os.path.join(trial_dir, "rerun_metrics.json"), rerun_metrics)

    plotting = Pl.Plotting(trial_dir, png_figures=False, svg_figures=False)

    if show_knn:
        plotting.knn(
            pp=pp,
            ff=ff,
            Perf=Perf,
            pf_ranges=pf_ranges,
            p_all=P_all,
            f_all=F_all,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=gen_mask,
            vmin=0.0,
            vmax=1.0,
            show=True,
        )

    if show_parity:
        flux0 = scaling_consts["flux0"]
        plotting.parity_CI(
            flux.detach().cpu().numpy() * flux0,
            y_hat_all_t.detach().cpu().numpy() * flux0,
            save_name="parity_all",
            show=True,
        )

    return rerun_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--trial", default=None)
    args = ap.parse_args()
    lib = Lib.Library(**tkwargs)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    base_path = os.path.join(project_root, "data", "rf", "pic")
    sub_case_dir = os.path.join(base_path, args.case_dir)
    PATH_TO_DATA = os.path.join(base_path, "ml_data")
    exp_dir = os.path.join(sub_case_dir, args.exp)

    o = Consts.OtherConstants()
    tkwargs = {"dtype": o.dtype, "device": o.device}
    # Load data
    data = np.load(os.path.join(PATH_TO_DATA, "mean_bulk_ccp_dataset.npz"))["profiles"]

    # Create phyics-informed library Theta^(g)
    flux_info, inputs_by_name, flux, roles = lib.make_flux_model(data, verbose=True)
    adhoc_Theta, adhoc_symbols_list, adhoc_exps = lib.create_library(flux_info, inputs_by_name)
    pruner = StatPrune(topk_remove=0, pruning_lmb=1e-6, verbose=True)
    phys_Theta, phys_symbols_list_full, phys_exps = pruner.keep_whitelist(flux_info, adhoc_Theta, adhoc_symbols_list, adhoc_exps, roles)

    scaling_consts = np.load(os.path.join(base_path, "scaling_consts.npz"))

    Theta = phys_Theta
    symbols_list = phys_symbols_list_full
    flux = flux

    M_total = int(Theta.shape[0])

    P_all = inputs_by_name["P"].to(**tkwargs)[:, 0].detach().cpu().numpy()
    F_all = inputs_by_name["F"].to(**tkwargs)[:, 0].detach().cpu().numpy()

    p_min = float(P_all.min())
    p_max = float(P_all.max())
    f_min = float(F_all.min())
    f_max = float(F_all.max())

    P_unique = np.unique(P_all)
    F_unique = np.unique(F_all)
    p_half_min = float(np.min(np.diff(np.sort(P_unique))))
    f_half_min = float(np.min(np.diff(np.sort(F_unique))))

    min_train = 5
    min_val = 1

    SST_FLOOR = _sst_floor_from_flux(flux, np.ones(M_total, dtype=bool), q=0.05, eps=1e-12)

    rng0 = np.random.default_rng(1)
    initial_coef_full = rng0.uniform(low=-1.0, high=1.0, size=len(symbols_list))

    if args.trial:
        run_one_trial(
            trial_name=args.trial,
            exp_dir=exp_dir,
            Theta=Theta,
            symbols_list=symbols_list,
            flux=flux,
            P_all=P_all,
            F_all=F_all,
            p_min=p_min,
            p_max=p_max,
            f_min=f_min,
            f_max=f_max,
            p_half_min=p_half_min,
            f_half_min=f_half_min,
            min_train=min_train,
            min_val=min_val,
            SST_FLOOR=SST_FLOOR,
            initial_coef_full=initial_coef_full,
            scaling_consts=scaling_consts
        )
        return

    summary_csv = os.path.join(exp_dir, "summary_all.csv")
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Could not find summary_all.csv in {exp_dir}")

    trial_names = _read_summary_trials(summary_csv)
    if len(trial_names) == 0:
        raise ValueError(f"No trials found in {summary_csv}")

    for trial_name in trial_names:
        run_one_trial(
            trial_name=trial_name,
            exp_dir=exp_dir,
            Theta=Theta,
            symbols_list=symbols_list,
            flux=flux,
            P_all=P_all,
            F_all=F_all,
            p_min=p_min,
            p_max=p_max,
            f_min=f_min,
            f_max=f_max,
            p_half_min=p_half_min,
            f_half_min=f_half_min,
            min_train=min_train,
            min_val=min_val,
            SST_FLOOR=SST_FLOOR,
            initial_coef_full=initial_coef_full,
            scaling_consts=scaling_consts
        )


if __name__ == "__main__":
    main()
