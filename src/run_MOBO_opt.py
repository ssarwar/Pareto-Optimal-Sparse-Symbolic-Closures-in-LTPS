import os, sys, warnings, datetime, torch, json
import hashlib, json as _json
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from ax.generation_strategy.generation_strategy import GenerationStrategy, GenerationStep
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from ax.modelbridge.registry import Generators

import utils.constants as Consts
import utils.plotting as Pl
import utils.library as Lib
from utils.prune import StatPrune
from utils.generalization import KNN_Generalizability
from utils.post_processing import post_processing
from utils.run_opt import run_opt

warnings.filterwarnings(
    "ignore",
    message=".*empty or all-NA columns.*",
    category=FutureWarning,
)

p = Consts.PhysicalConstants()
o = Consts.OtherConstants()
tkwargs = {"dtype": o.dtype, "device": o.device}
lib = Lib.Library(**tkwargs)

if len(sys.argv) < 3:
    raise SystemExit("Usage: run_MOBO.py <case> [num_new_trials] [optional_snapshot_path]")
case = sys.argv[1]
num_new_trials = int(sys.argv[2])

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
base_path = os.path.join(project_root, "data", "rf", "pic")
PATH_TO_DATA = os.path.join(base_path, "ml_data")
sub_case_dir = os.path.join(base_path, case)


l1_min, l1_max = -8, -1
l2_min, l2_max = -8, -1

obj1gen_th = 1.0
obj2comp_th = 0.08
obj3val_th = 1.0
obj_cols = ["generalizability_error", "nonzero_count", "validation_error"]
obj_cols_labels = ["Generalizability", "Complexity", "Validation"]
ref = [obj1gen_th, obj2comp_th, obj3val_th]
thresholds = torch.tensor(ref, dtype=torch.double)

run_tag = datetime.datetime.now().strftime("%H%M%S_%d%m%Y")
mobo_dir = os.path.join(sub_case_dir, f"{case}_{run_tag}")
os.makedirs(mobo_dir, exist_ok=True)


# Load data
data = np.load(os.path.join(PATH_TO_DATA, "mean_bulk_ccp_dataset.npz"))["profiles"]

# Create phyics-informed library Theta^(g)
flux_info, inputs_by_name, flux, roles = lib.make_flux_model(data, verbose=False)
adhoc_Theta, adhoc_symbols_list, adhoc_exps = lib.create_library(flux_info, inputs_by_name)
pruner = StatPrune(topk_remove=0, pruning_lmb=1e-6, verbose=False)
phys_Theta, phys_symbols_list_full, phys_exps = pruner.keep_whitelist(flux_info, adhoc_Theta, adhoc_symbols_list, adhoc_exps, roles)

P_all = (inputs_by_name["P"].to(**tkwargs))[:, 0].detach().cpu().numpy()
F_all = (inputs_by_name["F"].to(**tkwargs))[:, 0].detach().cpu().numpy()

# Full PF grid used to build the library
p_min, p_max = float(P_all.min()), float(P_all.max())
f_min, f_max = float(F_all.min()), float(F_all.max())
P_unique, F_unique = np.unique(P_all), np.unique(F_all)
p_half_min = float(np.min(np.diff(np.sort(P_unique))))
f_half_min = float(np.min(np.diff(np.sort(F_unique))))

p_half_max = float(0.5 * (p_max - p_min) - p_half_min)
f_half_max = float(0.5 * (f_max - f_min) - f_half_min)

Theta = phys_Theta
symbols_list = phys_symbols_list_full
M_total, N, _ = Theta.shape

min_train = 5
min_val = 1

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
SST_FLOOR = _sst_floor_from_flux(flux, np.ones(M_total, dtype=bool), q=0.05, eps=1e-12)

EXPERIMENT_SEED = 1
rng0 = np.random.default_rng(EXPERIMENT_SEED)
initial_coef_full = rng0.uniform(low=-1.0, high=1.0, size=len(symbols_list))

eval_counter = 0

def complexity_norm_from_nonzero(raw_nonzero, d_full):
    if int(raw_nonzero) <= 0:
        return 1.0
    return float(raw_nonzero) / float(d_full)

def scale_err_array(x, invalid_value=1.0):
    x = np.asarray(x, float)
    out = np.full_like(x, invalid_value, dtype=float)
    mask = np.isfinite(x) & (x >= 0.0)
    out[mask] = x[mask] / (1.0 + x[mask])
    return out

def select_samples_by_window(p_lo, p_hi, f_lo, f_hi):
    min_window = max(min_train + min_val, 6)

    p_lo_eff = max(p_lo, p_min)
    p_hi_eff = min(p_hi, p_max)
    f_lo_eff = max(f_lo, f_min)
    f_hi_eff = min(f_hi, f_max)

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

        p_lo_eff = p_lo2
        p_hi_eff = p_hi2
        f_lo_eff = f_lo2
        f_hi_eff = f_hi2

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

def _to_numpy_int(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(int)
    return np.asarray(x, dtype=int)

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

def evaluate(params) -> dict:
    global eval_counter
    eval_counter += 1
    iteration = eval_counter

    lmb1 = float(params["lmb1"])
    lmb2 = float(params["lmb2"])

    p_c = float(params["p_c"])
    p_hw = float(params["p_hw"])
    f_c = float(params["f_c"])
    f_hw = float(params["f_hw"])

    p_lo = p_c - p_hw
    p_hi = p_c + p_hw
    f_lo = f_c - f_hw
    f_hi = f_c + f_hw

    window_m, p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff = select_samples_by_window(
        p_lo, p_hi, f_lo, f_hi
    )

    seed, split_hash8 = _window_seed(p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff)
    rng = np.random.default_rng(seed)

    tag_hash = hashlib.sha1(
        _json.dumps(
            {
                "l1": lmb1, "l2": lmb2,
                "p_c": p_c, "p_hw": p_hw,
                "f_c": f_c, "f_hw": f_hw,
                "p_lo_eff": float(p_lo_eff),
                "p_hi_eff": float(p_hi_eff),
                "f_lo_eff": float(f_lo_eff),
                "f_hi_eff": float(f_hi_eff),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:8]
    tag = f"{int(iteration)}_{tag_hash}"

    out_dir = os.path.join(mobo_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    plotting_trial = Pl.Plotting(out_dir, png_figures=False, svg_figures=False)

    # window membership on the full dataset, defines generalizability set as outside-window
    in_window_mask = (
        (P_all >= p_lo_eff) & (P_all <= p_hi_eff)
        & (F_all >= f_lo_eff) & (F_all <= f_hi_eff)
    )
    gen_mask = ~in_window_mask
    gen_m = np.where(gen_mask)[0].astype(int)

    window_m = np.asarray(window_m, dtype=int)
    k = int(window_m.size)
    perm = rng.permutation(k)

    # 10% holdout within the window
    if k >= (min_train + min_val):
        n_val_target = int(np.round(0.1 * k))
        n_val = max(min_val, n_val_target)
        n_val = min(n_val, k - min_train)
    else:
        n_val = max(0, k - min_train)

    val_m = window_m[perm[:n_val]] if n_val > 0 else np.asarray([], dtype=int)
    train_m = window_m[perm[n_val:]] if n_val > 0 else window_m

    train_mask = np.zeros(M_total, dtype=bool)
    train_mask[train_m] = True

    val_mask = np.zeros(M_total, dtype=bool)
    if val_m.size > 0:
        val_mask[val_m] = True

    pf_ranges = {"window": (p_lo_eff, p_hi_eff, f_lo_eff, f_hi_eff)}

    train_idx_t = torch.as_tensor(train_m, device=o.device, dtype=torch.long)

    ############ CONSTRUCT TRAIN LIBRARY ############
    Theta_tr = Theta.index_select(0, train_idx_t)
    flux_tr = flux.index_select(0, train_idx_t)
    is_bulk_tr = np.full(Theta_tr.shape[0:2], True, dtype=bool)
    X_tr, y_tr, rid_tr, sid_tr, jid_tr = lib.region_flatten(
        Theta_tr, is_bulk_tr, flux_tr
    )
    f_train_full = lib.train_(X_tr, y_tr, rid_tr, sid_tr, jid_tr, N)

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

    f_train = lib.train_(f_Xb_opt, y_tr, rid_tr, sid_tr, jid_tr, N)
    f_symbols_list = [symbols_list[i] for i in keep_cols.tolist()]
    f_initial_coef = initial_coef_full[keep_cols]

    ############ RUN REGRESSION ############
    raw_nonzero = 0
    equation = ""
    coef_q_orig = None

    if keep_cols.size == 0:
        raw_nonzero = 0
        equation = ""
        coef_q_orig = np.zeros((0,), dtype=float)
        f_coef_star_std = np.zeros((0,), dtype=float)
    else:
        f_coef_star_std = run_opt(f_train, lmb1, lmb2, f_initial_coef)
        f_coef_star_std = np.asarray(f_coef_star_std, dtype=float)

        if not np.all(np.isfinite(f_coef_star_std)):
            raw_nonzero = 0
            equation = ""
            coef_q_orig = np.zeros((keep_cols.size,), dtype=float)
            f_coef_star_std = np.zeros((keep_cols.size,), dtype=float)
        else:
    ############ POST-PROCESSING ############
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

    ############ EVAL ON FULL PF GRID ############
    KNN = KNN_Generalizability()

    keep_t = torch.as_tensor(
        np.asarray(keep_cols, dtype=np.int64),
        device=o.device,
        dtype=torch.long,
    )
    Theta_sel_all = Theta.index_select(dim=2, index=keep_t)

    if coef_q_orig is None:
        coef_q_orig = np.zeros((keep_cols.size,), dtype=float)

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

    complexity_norm = complexity_norm_from_nonzero(raw_nonzero, len(symbols_list))
    comp_sem = 1e-3

    nonzero_count = float(raw_nonzero)
    nz_sem = 1e-3


    ############ PERFORMANCE MAP ############
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
        comp_sem = 0.2

        nonzero_count = float(raw_nonzero)
        nz_sem = 0.2

    ############ RECORD ############
    metrics = {
        "tag": f"{tag}",
        "lmb1": float(lmb1),
        "lmb2": float(lmb2),

        "equation": equation,

        "nonzero_count": raw_nonzero,
        "complexity_norm": float(complexity_norm),

        "generalizability_error": float(generalizability_error),
        "validation_error": float(validation_error),
        "window_split_hash8": str(split_hash8),

        "train_idx": np.asarray(train_m, int).tolist(),
        "val_idx": np.asarray(val_m, int).tolist(),
        "gen_idx": np.asarray(gen_m, int).tolist(),

        "keep_cols": keep_cols.tolist(),
        "coefficients": np.asarray(f_coef_star_std, float).tolist(),

        "iteration": int(iteration),

        "p_c": float(p_c),
        "p_hw": float(p_hw),
        "f_c": float(f_c),
        "f_hw": float(f_hw),

        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "f_lo": float(f_lo),
        "f_hi": float(f_hi),
        "p_lo_eff": float(p_lo_eff),
        "p_hi_eff": float(p_hi_eff),
        "f_lo_eff": float(f_lo_eff),
        "f_hi_eff": float(f_hi_eff),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # plotting_trial.parity_CI( flux_pred["y_true_all"], flux_pred["y_pred_masked_all"] )

    plotting_trial.knn(
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
    )

    return {
        "generalizability_error": (generalizability_error, gen_sem),
        "complexity_norm": (complexity_norm, comp_sem),
        "validation_error": (validation_error, val_sem),
        "nonzero_count": (nonzero_count, nz_sem),
    }

gs = GenerationStrategy(steps=[
    GenerationStep(model=Generators.SOBOL, num_trials=40),
    GenerationStep(
        model=Generators.BOTORCH_MODULAR,
        num_trials=-1,
        model_kwargs={
            "botorch_acqf_class": qLogNoisyExpectedHypervolumeImprovement,
            "acquisition_options": {"objective_thresholds": thresholds},
        },
        max_parallelism=None,
    ),
])

parameters = [
    {
        "name": "lmb1",
        "type": "range",
        "bounds": [10**l1_min, 10**l1_max],
        "log_scale": True,
        "value_type": "float",
    },
    {
        "name": "lmb2",
        "type": "range",
        "bounds": [10**l2_min, 10**l2_max],
        "log_scale": True,
        "value_type": "float",
    },
    {
        "name": "p_c",
        "type": "range",
        "bounds": [p_min, p_max],
        "value_type": "float",
    },
    {
        "name": "p_hw",
        "type": "range",
        "bounds": [p_half_min, p_half_max],
        "value_type": "float",
    },
    {
        "name": "f_c",
        "type": "range",
        "bounds": [f_min, f_max],
        "value_type": "float",
    },
    {
        "name": "f_hw",
        "type": "range",
        "bounds": [f_half_min, f_half_max],
        "value_type": "float",
    },
]

objectives = {
    "generalizability_error": ObjectiveProperties(minimize=True, threshold=obj1gen_th),
    "complexity_norm": ObjectiveProperties(minimize=True, threshold=obj2comp_th),
    "validation_error": ObjectiveProperties(minimize=True, threshold=obj3val_th),
}

print("[Fresh] Creating new Ax experiment")
ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name=f"{case}_moo",
    parameters=parameters,
    objectives=objectives,
    tracking_metric_names=["nonzero_count"],
    parameter_constraints=[
        f"p_c - p_hw >= {p_min}",
        f"p_c + p_hw <= {p_max}",
        f"f_c - f_hw >= {f_min}",
        f"f_c + f_hw <= {f_max}",
    ],
    overwrite_existing_experiment=True,
    is_test=False,
)

for i in range(num_new_trials):
    params, trial_index = ax_client.get_next_trial()
    try:
        raw_data = evaluate(params)
    except Exception as e:
        print(f"[evaluate error] trial {trial_index} failed with {repr(e)}")
        ax_client.abandon_trial(trial_index=trial_index, reason=f"error: {e}")
        continue
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

ax_client.save_to_json_file(os.path.join(mobo_dir, "ax_client_snapshot_after.json"))

# collate results
records = []
for tag in sorted(os.listdir(mobo_dir)):
    path = os.path.join(mobo_dir, tag, "metrics.json")
    if os.path.isfile(path):
        with open(path, "r") as f:
            records.append(json.load(f))

if len(records) > 0:
    df = pd.DataFrame(records)
    if "iteration" in df.columns:
        df = df.sort_values("iteration").reset_index(drop=True)
    df.to_csv(os.path.join(mobo_dir, "summary_all.csv"), index=False)

print(f"Done. Outputs in: {mobo_dir}")
