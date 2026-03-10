import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

class StatPrune:
    def __init__(self, topk_remove, pruning_lmb, verbose=False):
        self.topk_remove = topk_remove
        self.pruning_lmb = pruning_lmb
        self.verbose = verbose

    def _whitelist(self, term_exps, names, roles):
        """
        Enforce structural assumptions of the closure: 
        - flux is a linear combination of first derivatives
        - coefficients depend only on local state variables and parameters
        - there is one driver of gradient per term
        """
        eps = 1e-12
        d = dict(zip(names, term_exps))

        if all(abs(e) < eps for e in term_exps):
            return True
        
        for k in names:
            if k not in roles:
                roles[k] = 'param'

        def is_zero(x):
            return abs(x) < eps

        grad_vars = [k for k, r in roles.items() if r == 'grad']
        grad_exps = {g: d.get(g, 0.0) for g in grad_vars}
        nonzero_grads = [g for g, e in grad_exps.items() if not is_zero(e)]
        if len(nonzero_grads) > 1:
            return False
        if len(nonzero_grads) == 1:
            g = nonzero_grads[0]
            if not is_zero(grad_exps[g] - 1.0):
                return False
        has_grad = len(nonzero_grads) == 1

        for k in ('T_e', 'T_i'):
            e = d.get(k, 0.0)
            if is_zero(e):
                continue
            if not (is_zero(e - 0.5) or is_zero(e - 1.0)):
                return False

        if has_grad and not is_zero(d.get('E', 0.0)):
            return False

        n_exp = d.get('n', 0.0)
        if has_grad:
            if n_exp > eps:
                return False
            g = nonzero_grads[0]
            if g == 'dn':
                if not (is_zero(n_exp) or is_zero(n_exp + 1.0)):
                    return False
            else:
                if not is_zero(n_exp):
                    return False

        if not has_grad:
            if not is_zero(n_exp - 1.0):
                return False
            if is_zero(d.get('E', 0.0)):
                return False

        return True
    
    def keep_whitelist(self, model_info, Theta, syms, exps, roles):
        mask = [self._whitelist(e, model_info['var_names'], roles) for e in exps]
        Theta = Theta[:, :, mask]
        syms  = [s for s,m in zip(syms, mask) if m]
        exps  = [e for e,m in zip(exps, mask) if m]

        if model_info['verbose']:
            print(f"\nShape (masked): {Theta.shape} after whitelisting")
            for i, s in enumerate(syms):
                mn = Theta[:, :, i].min().item()
                mx = Theta[:, :, i].max().item()
                print(f"{i} | {s}: min={mn}, max={mx}")
        return Theta, syms, exps

    def weighted_ridge(self, X, y, w=None, lam=1e-6, no_penalize_const=True, const_tol=1e-12):
        d = X.shape[1]
        if no_penalize_const:
            col_std = X.std(dim=0, unbiased=False)
            const_mask = col_std <= const_tol
            penalty = torch.ones(d, device=X.device, dtype=X.dtype)
            penalty[const_mask] = 0.0
            P = torch.diag(penalty)
        else:
            P = torch.eye(d, device=X.device, dtype=X.dtype)

        if w is None:
            H = X.T @ X + lam * P
            b = X.T @ y
        else:
            w = w if w.ndim > 1 else w.unsqueeze(1)
            sqrtw = torch.sqrt(w + 1e-15)
            Xw, yw = X * sqrtw, y * sqrtw
            H = Xw.T @ Xw + lam * P
            b = Xw.T @ yw

        beta = torch.linalg.solve(H, b)
        return beta

    def safe_weighted_r2(self, y, yhat, w=None, var_floor=1e-10):
        y, yhat = y.squeeze(-1), yhat.squeeze(-1)
        if w is None:
            var = torch.var(y, unbiased=False)
            if float(var) < var_floor:
                return float("nan")
            ss_res = torch.sum((y - yhat)**2)
            ss_tot = torch.sum((y - torch.mean(y))**2)
        else:
            w = w[:, None]
            wsum = torch.sum(w)
            mu = torch.sum(w * y) / wsum
            var = torch.sum(w * (y - mu)**2) / wsum
            if float(var) < var_floor:
                return float("nan")
            ss_res = torch.sum(w * (y - yhat)**2)
            ss_tot = torch.sum(w * (y - mu)**2)
        return float(1.0 - ss_res / ss_tot)

    def model_r2(self, X, y, w=None, lam=1e-6, **ridge_kwargs):
        beta = self.weighted_ridge(X, y, w=w, lam=lam, **ridge_kwargs)
        yhat = X @ beta
        return self.safe_weighted_r2(y, yhat, w), yhat, beta

    def cx_alignment_report(self, Xstd, w, cx_idx, base_mask, names, lam=1e-6, top_k=20, label=""):
        Xb = Xstd[:, base_mask]
        cx_col = Xstd[:, cx_idx:cx_idx + 1]
        base_cols = np.where(base_mask)[0]
        base_names = [names[j] for j in base_cols]

        R2_all, _, beta = self.model_r2(Xb, cx_col, w=w, lam=lam)
        beta = beta.squeeze(-1)

        dR2_loo = []
        for jloc in range(Xb.shape[1]):
            keep = np.ones(Xb.shape[1], dtype=bool)
            keep[jloc] = False
            R2_drop, _, _ = self.model_r2(Xb[:, keep], cx_col, w=w, lam=lam)
            dR2_loo.append(R2_all - R2_drop)

        dR2_loo = np.array(dR2_loo)
        order_beta = torch.argsort(torch.abs(beta), descending=True).cpu().numpy()
        order_dR2 = np.argsort(-dR2_loo)

        if self.verbose:
            print(f"\n[CX ~ baseline alignment — {label}]  (R²_all={R2_all:.4f})")
            print("Top by |β̂| (standardized):")
            for r in range(min(top_k, len(order_beta))):
                jloc = order_beta[r]
                jglob = base_cols[jloc]
                print(f"  {r+1:3d}  {base_names[jloc]:25s} |β̂|={float(torch.abs(beta[jloc])): .3e} ΔR²={dR2_loo[jloc]: .3e}")

            print("\nTop by ΔR² leave-one-out (bigger ⇒ stronger CX overlap):")
            for r in range(min(top_k, len(order_dR2))):
                jloc = order_dR2[r]
                jglob = base_cols[jloc]
                print(f"  {r+1:3d}  {base_names[jloc]:25s} ΔR²={dR2_loo[jloc]: .3e} |β̂|={float(torch.abs(beta[jloc])): .3e}")

        return dict(
            R2_all=R2_all,
            beta=beta.detach().cpu().numpy(),
            dR2_loo=dR2_loo,
            order_by_beta=order_beta,
            order_by_dR2=order_dR2,
            base_cols=base_cols,
            base_names=base_names,
        )

    def is_cx_symbol(self, s: str) -> bool:
        return s.replace(" ", "") == "E**0.5*n*P**(-0.5)"

    def run_prune(self, f_train_full, f_Xb, f_symbols_list_full):
        sym_strs = [str(s) for s in f_symbols_list_full]
        bare_cx_name = "E**0.5*n*P**(-0.5)"
        idx_cx_bare = next((i for i, s in enumerate(sym_strs) if s == bare_cx_name), None)  

        cx_group_idx = [i for i, s in enumerate(sym_strs) if self.is_cx_symbol(s)]
    
        non_drift_mask = np.ones(len(sym_strs), dtype=bool)
        non_drift_mask[cx_group_idx] = False
        cx_mask = np.zeros(len(sym_strs), dtype=bool)
        cx_mask[cx_group_idx] = True
        base_mask = non_drift_mask.copy()

        train_align = self.cx_alignment_report(
            Xstd=f_train_full["X"], w=f_train_full["w"], cx_idx=idx_cx_bare,
            base_mask=base_mask, names=sym_strs, lam=self.pruning_lmb,
            top_k=len(sym_strs), label="TRAIN"
        )

        base_cols = train_align["base_cols"]
        base_names = train_align["base_names"]
        order_dR2 = train_align["order_by_dR2"]
        dR2_loo = train_align["dR2_loo"]

        jloc_top = order_dR2[:self.topk_remove]
        jglob_top = base_cols[jloc_top]

        to_keep, to_drop = [], []
        for jloc, jglob in zip(jloc_top, jglob_top):
            name = base_names[jloc]
            if name.strip() == "1":
                to_keep.append((jglob, name, float(dR2_loo[jloc])))
            else:
                to_drop.append((jglob, name, float(dR2_loo[jloc])))

        base_mask_pruned = base_mask.copy()
        for jglob, _, _ in to_drop:
            base_mask_pruned[jglob] = False
        full_mask_pruned = (base_mask_pruned | cx_mask)

        if self.verbose:
            print("\n[Pruning summary — TRAIN ΔR2_loo top overlap] Dropped:")
            for (jglob, name, score) in to_drop:
                print(f"  idx={jglob:3d}  {name:26s}  ΔR²_loo={score: .3e}")

        keep_cols = np.where(full_mask_pruned)[0].astype(np.int64)
        assert keep_cols.size > 0, "Pruning produced an empty column set."
        if idx_cx_bare is not None:
            assert idx_cx_bare in keep_cols, "CX term was accidentally pruned."

        keep_t = torch.as_tensor(keep_cols, device=f_Xb.device, dtype=torch.long)
        f_Xb_opt = f_Xb.index_select(1, keep_t)
        f_symbols_list = [f_symbols_list_full[i] for i in keep_cols.tolist()]

        if self.verbose:
            print(f"\nKept {len(f_symbols_list)} terms (from {len(sym_strs)} original).")

        return f_Xb_opt, f_symbols_list, keep_cols
    