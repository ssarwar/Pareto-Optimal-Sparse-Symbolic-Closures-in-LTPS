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

if len(sys.argv) < 2:
    raise SystemExit("Usage: run_MOBO_static.py <case> [num_new_trials] [optional_snapshot_path]")
case = sys.argv[1]
num_new_trials = int(sys.argv[2])

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
base_path = os.path.join(project_root, "data", "rf", "pic")
PATH_TO_DATA = os.path.join(base_path, "ml_data")
sub_case_dir = os.path.join(base_path, case)


l1_min, l1_max = -8, -1
l2_min, l2_max = -8, -1

cv_th = 1.0
comp_th = 0.2
obj1_col = "cv5_error"
obj1_col_label = "Cross-Validation Error"
ref = [cv_th, comp_th]
thresholds = torch.tensor(ref, dtype=torch.double)

run_tag = datetime.datetime.now().strftime("%H%M%S_%d%m%Y")
mobo_dir = os.path.join(sub_case_dir, f"{case}_static_{run_tag}")
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

p_min, p_max = float(P_all.min()), float(P_all.max())
f_min, f_max = float(F_all.min()), float(F_all.max())

Theta = phys_Theta
symbols_list = phys_symbols_list_full
M_total, N, _ = Theta.shape

N_FOLDS = 5
all_idx = np.arange(M_total, dtype=int)

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

perm_all = rng0.permutation(M_total)
folds = np.array_split(perm_all, N_FOLDS)

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

def _to_numpy_int(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(int)
    return np.asarray(x, dtype=int)

def fit_model_on_profiles(train_m, lmb1, lmb2):
    train_m = np.asarray(train_m, dtype=int)
    train_idx_t = torch.as_tensor(train_m, device=o.device, dtype=torch.long)

    Theta_tr = Theta.index_select(0, train_idx_t)
    flux_tr = flux.index_select(0, train_idx_t)

    is_bulk_tr = np.full(Theta_tr.shape[0:2], True, dtype=bool)
    X_tr, y_tr, rid_tr, sid_tr, jid_tr = lib.region_flatten(Theta_tr, is_bulk_tr, flux_tr)
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

    if keep_cols.size == 0:
        return {
            "keep_cols": keep_cols,
            "coef_q_orig": np.zeros((0,), dtype=float),
            "raw_nonzero": 0,
            "equation": "",
            "coef_star": np.zeros((0,), dtype=float),
        }

    f_train = lib.train_(f_Xb_opt, y_tr, rid_tr, sid_tr, jid_tr, N)
    f_symbols_list = [symbols_list[i] for i in keep_cols.tolist()]
    f_initial_coef = initial_coef_full[keep_cols]

    coef_star = run_opt(f_train, float(lmb1), float(lmb2), f_initial_coef)
    coef_star = np.asarray(coef_star, dtype=float)

    if not np.all(np.isfinite(coef_star)):
        return {
            "keep_cols": keep_cols,
            "coef_q_orig": np.zeros((keep_cols.size,), dtype=float),
            "raw_nonzero": 0,
            "equation": "",
            "coef_star": np.zeros((keep_cols.size,), dtype=float),
        }

    flux_pred = post_processing(
        coef_star,
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

    return {
        "keep_cols": keep_cols,
        "coef_q_orig": coef_q_orig,
        "raw_nonzero": raw_nonzero,
        "equation": equation,
        "coef_star": coef_star,
    }

def masked_mean_and_sem(v, empty_value=1.0, sem_default=0.5, sem_floor=1e-3):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    k = v.size
    if k == 0:
        return float(empty_value), float(sem_default)
    mu = float(np.mean(v))
    if k == 1:
        return mu, float(sem_default)
    sem = float(np.std(v, ddof=1) / np.sqrt(k))
    return mu, float(max(sem, sem_floor))

def eval_scaled_error_on_profiles(val_m, keep_cols, coef_q_orig):
    val_m = np.asarray(val_m, dtype=int)
    if val_m.size == 0:
        return np.asarray([], dtype=float)

    val_idx_t = torch.as_tensor(val_m, device=o.device, dtype=torch.long)
    y_true = flux.index_select(0, val_idx_t).detach().cpu().numpy().astype(np.float64)

    if keep_cols.size == 0 or coef_q_orig is None or np.asarray(coef_q_orig).size == 0:
        y_pred = np.zeros_like(y_true)
    else:
        keep_t = torch.as_tensor(np.asarray(keep_cols, dtype=np.int64), device=o.device, dtype=torch.long)
        Theta_val = Theta.index_select(0, val_idx_t).index_select(dim=2, index=keep_t)
        coef_t = torch.as_tensor(np.asarray(coef_q_orig, dtype=float), **tkwargs).view(-1, 1)
        y_pred_t = (Theta_val @ coef_t).squeeze(-1)
        y_pred = y_pred_t.detach().cpu().numpy().astype(np.float64)

    mu = np.mean(y_true, axis=1, keepdims=True)
    sst = np.sum((y_true - mu) ** 2, axis=1)
    sse = np.sum((y_pred - y_true) ** 2, axis=1)

    denom = np.maximum(sst, SST_FLOOR)
    err_ratio = np.full_like(sse, np.nan, dtype=float)
    good = np.isfinite(sse) & np.isfinite(denom) & (denom > 0.0)
    err_ratio[good] = sse[good] / denom[good]

    scaled = scale_err_array(err_ratio, invalid_value=1.0)
    return scaled

def evaluate(params) -> dict:
    global eval_counter
    eval_counter += 1
    iteration = eval_counter

    lmb1 = float(params["lmb1"])
    lmb2 = float(params["lmb2"])

    tag_hash = hashlib.sha1(
        _json.dumps({"l1": lmb1, "l2": lmb2}, sort_keys=True).encode()
    ).hexdigest()[:8]
    tag = f"{int(iteration)}_{tag_hash}"

    out_dir = os.path.join(mobo_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    plotting_trial = Pl.Plotting(out_dir, png_figures=False, svg_figures=False)

    fold_means = []
    fold_raw_nonzero = []

    perf_field_cv = np.full(M_total, np.nan, dtype=float)

    for k in range(N_FOLDS):
        val_m = np.asarray(folds[k], dtype=int)
        train_m = np.setdiff1d(all_idx, val_m, assume_unique=False)

        fit = fit_model_on_profiles(train_m, lmb1, lmb2)
        keep_cols = fit["keep_cols"]
        coef_q_orig = fit["coef_q_orig"]

        scaled_err = eval_scaled_error_on_profiles(val_m, keep_cols, coef_q_orig)
        perf_field_cv[val_m] = scaled_err

        mu_k, _ = masked_mean_and_sem(scaled_err, empty_value=1.0, sem_default=0.5)
        fold_means.append(mu_k)
        fold_raw_nonzero.append(int(fit["raw_nonzero"]))

    perf_field_cv = np.where(np.isfinite(perf_field_cv), perf_field_cv, 1.0).astype(float)

    cv5_error = float(np.mean(fold_means))
    cv5_sem = float(np.std(fold_means, ddof=1) / np.sqrt(N_FOLDS)) if N_FOLDS > 1 else 0.5

    fit_full = fit_model_on_profiles(all_idx, lmb1, lmb2)
    raw_nonzero_full = int(fit_full["raw_nonzero"])
    complexity_norm = float(raw_nonzero_full) / float(len(symbols_list))
    complexity_norm = float(np.clip(complexity_norm, 0.0, 1.0))
    comp_sem = 1e-3

    degenerate = (raw_nonzero_full <= 0)
    if degenerate:
        cv5_sem = 0.5
        comp_sem = 0.2

    KNN = KNN_Generalizability()
    Perf, _, (pp, ff) = KNN.performance_map_continuous(
        p_all=P_all,
        f_all=F_all,
        performance_all=perf_field_cv,
        in_mask=np.ones(M_total, dtype=bool),
    )

    mask0 = np.zeros(M_total, dtype=bool)
    plotting_trial.knn(
        pp=pp,
        ff=ff,
        Perf=Perf,
        pf_ranges={"window": (p_min, p_max, f_min, f_max)},
        p_all=P_all,
        f_all=F_all,
        train_mask=mask0,
        val_mask=mask0,
        test_mask=mask0,
        vmin=0.0,
        vmax=1.0,
    )

    metrics = {
        "tag": str(tag),
        "iteration": int(iteration),
        "lmb1": float(lmb1),
        "lmb2": float(lmb2),
        "cv5_error": float(cv5_error),
        "cv5_sem": float(cv5_sem),
        "complexity_norm": float(complexity_norm),
        "nonzero_count": int(raw_nonzero_full),
        "equation": str(fit_full.get("equation", "")),
        "fold_means": [float(x) for x in fold_means],
        "fold_raw_nonzero": [int(x) for x in fold_raw_nonzero],
        "degenerate_full_fit": bool(degenerate),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "cv5_error": (cv5_error, cv5_sem),
        "complexity_norm": (complexity_norm, comp_sem),
        "nonzero_count": (float(raw_nonzero_full), 0.2 if degenerate else 1e-3),
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
        "bounds": [10 ** l1_min, 10 ** l1_max],
        "log_scale": True,
        "value_type": "float",
    },
    {
        "name": "lmb2",
        "type": "range",
        "bounds": [10 ** l2_min, 10 ** l2_max],
        "log_scale": True,
        "value_type": "float",
    },
]

objectives = {
    "cv5_error": ObjectiveProperties(minimize=True, threshold=cv_th),
    "complexity_norm": ObjectiveProperties(minimize=True, threshold=comp_th),
}

print("[Static] Creating new Ax experiment")
ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name=f"{case}_moo_static",
    parameters=parameters,
    objectives=objectives,
    tracking_metric_names=["nonzero_count"],
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
