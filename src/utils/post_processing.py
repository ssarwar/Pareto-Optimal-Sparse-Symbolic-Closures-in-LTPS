import numpy as np
import torch
import sympy as syp 
import utils.constants as Consts
# Constants + config
p = Consts.PhysicalConstants() # physical constants
o = Consts.OtherConstants() # other constants

def post_processing(coef_star_std, symbols_list, train_info, thr_std=1e-2, lib_shape=None):
    M, N, _ = lib_shape
    
    y_tr = train_info["y"]
    if torch.is_tensor(y_tr) and y_tr.ndim == 2 and y_tr.shape[1] == 1:
        y_tr = y_tr.squeeze(-1)
    w = train_info.get("w", None)
    if w is None:
        w = torch.ones_like(y_tr, dtype=o.dtype, device=o.device)
    if torch.is_tensor(w) and w.ndim == 2 and w.shape[1] == 1:
        w = w.squeeze(-1)

    # threshold in standardized space
    mask_std = (np.abs(coef_star_std) > thr_std)
    nonzero_count = int(mask_std[1:].sum()) 
    
    # backtransform to theta space
    coef_std_t = torch.tensor(coef_star_std, device=o.device, dtype=o.dtype)
    coef_orig_full_t = train_info["lib_scaler"].backtransform_coefs(
        coef_std_t, train_info["mu"], train_info["sigma"])

    # coef_orig_full_t[0] = 0

    mask_apply = torch.as_tensor(mask_std, dtype=torch.bool, device=o.device)
    mask_apply[0] = True
    coef_std_masked_t = coef_std_t.clone()
    coef_std_masked_t[~mask_apply] = 0.0
    coef_orig_masked_t = train_info["lib_scaler"].backtransform_coefs(
        coef_std_masked_t, train_info["mu"], train_info["sigma"]
    )
    coef_star_std_masked = coef_std_t.clone()
    coef_star_std_masked[~mask_apply] = 0.0
    # coef_orig_masked_t[0] = 0
    
    coef_print = coef_orig_masked_t.cpu().numpy()
    nonzero_idxs = np.where(coef_print)[0]
    intercept_term = syp.N(coef_print[0], o.sig_figs) if 0 in nonzero_idxs else 0
    if 0 in nonzero_idxs:
        nonzero_idxs = nonzero_idxs[nonzero_idxs != 0]
    expr_phys = sum(syp.N(coef_print[j], o.sig_figs) * symbols_list[j] for j in nonzero_idxs)# + intercept_term
    latex_final = syp.latex(expr_phys)

    appeared = torch.zeros(M, dtype=torch.bool, device=o.device)
    sid_t = torch.as_tensor(train_info["sid"], device=o.device, dtype=torch.long)
    appeared[sid_t] = True
    samples = appeared.nonzero(as_tuple=False).squeeze(-1)

    sampled_idx = int(samples[0])
    rows_s = np.flatnonzero(train_info["sid"] == sampled_idx) # indices into X / y
    j_idx = train_info["jid"][rows_s]
    order = np.argsort(j_idx) # sort spatially
    j_idx = j_idx[order]
    rows_s  = rows_s[order]

    unique_te, counts_te = np.unique(train_info["sid"], return_counts=True)
    inv_counts_te = {s: 1.0 / c for s, c in zip(unique_te, counts_te)}
    w = torch.as_tensor(
        np.array([inv_counts_te[s] for s in train_info["sid"]]),
        device=o.device,
        dtype=o.dtype,
    )
    if w.ndim == 2 and w.shape[1] == 1:
        w = w.squeeze(-1)

    # flat train set
    y_pred_full_flat = train_info["X"] @ coef_std_t
    y_pred_masked_flat = train_info["X"] @ coef_std_masked_t
    if y_pred_full_flat.ndim == 2 and y_pred_full_flat.shape[1] == 1:
        y_pred_full_flat = y_pred_full_flat.squeeze(-1)
    if y_pred_masked_flat.ndim == 2 and y_pred_masked_flat.shape[1] == 1:
        y_pred_masked_flat = y_pred_masked_flat.squeeze(-1)

    y_true_all_np = y_tr.detach().cpu().numpy().ravel()
    y_pred_full_all_np = y_pred_full_flat.detach().cpu().numpy().ravel()
    y_pred_masked_all_np = y_pred_masked_flat.detach().cpu().numpy().ravel()
    sid_all_np = train_info["sid"]

    # scatter back
    true_full = torch.full((M, N), float('nan'), device=o.device, dtype=o.dtype)
    pred_full_canvas = torch.full_like(true_full, float('nan'))
    pred_masked_canvas = torch.full_like(true_full, float('nan'))

    rid = train_info["rid"]
    true_full.view(-1)[rid] = y_tr
    pred_full_canvas.view(-1)[rid] = y_pred_full_flat
    pred_masked_canvas.view(-1)[rid] = y_pred_masked_flat

    # metrics
    y_bar = torch.sum(w * y_tr) / w.sum()

    resid_masked = y_pred_masked_flat - y_tr
    sse_masked = torch.sum(w * resid_masked ** 2)
    mse_masked = sse_masked / w.sum()
    sst_masked = torch.sum(w * (y_tr - y_bar) ** 2)
    r2_masked = 1.0 - sse_masked / sst_masked.clamp_min(o.delta)

    rmse_masked = torch.sqrt(mse_masked)
    y_std_masked = torch.std(y_tr)
    nrmse_masked = rmse_masked / y_std_masked.clamp_min(o.delta)

    resid_full = y_pred_full_flat - y_tr
    sse_full = torch.sum(w * resid_full ** 2)
    mse_full = sse_full / w.sum()

    rmse_full = torch.sqrt(mse_full)
    y_std_full = torch.std(y_tr)
    nrmse_full = rmse_full / y_std_full.clamp_min(o.delta)

    sst_full = torch.sum(w * (y_tr - y_bar) ** 2)
    r2_full = 1.0 - sse_full / sst_full.clamp_min(o.delta)
    
    gt = true_full.detach().cpu().numpy()
    pall = pred_full_canvas.detach().cpu().numpy()
    pm = pred_masked_canvas.detach().cpu().numpy()

    model_info = {
        "equation" : latex_final,
        "nonzero_count" : nonzero_count,
        "theta_coef_full" : coef_std_t.detach().cpu().numpy(),
        "theta_coef_masked" : coef_star_std_masked.detach().cpu().numpy(),
        "nondim_coef" : coef_orig_masked_t,

        "sse_masked" : sse_masked.item(),
        "mse_masked" : mse_masked.item(),
        "r2_masked" : r2_masked.item(),
        "nrmse_masked" : nrmse_masked.item(),

        "sse_full" : sse_full.item(),
        "mse_full" : mse_full.item(),
        "r2_full" : r2_full.item(),
        "nrmse_full" : nrmse_full.item(),
        "sample_idx" : samples,
        "j_idx" : j_idx,
        "ground_truth" : gt,
        "predicted" : pall,
        "predicted_masked" : pm,
        "y_true_all": y_true_all_np,
        "y_pred_full_all": y_pred_full_all_np,
        "y_pred_masked_all": y_pred_masked_all_np,
        "sid_all": sid_all_np
    }
    return model_info