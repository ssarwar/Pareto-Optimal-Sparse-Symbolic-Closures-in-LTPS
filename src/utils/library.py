import torch
import sympy as sp
import numpy as np
import utils.constants as Consts
from itertools import product

# Constants + config
p = Consts.PhysicalConstants() # physical constants
o = Consts.OtherConstants() # other constants

class Library:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def make_flux_model(self, loaded_data, verbose=False):
        # target
        fluxi = torch.tensor(loaded_data[..., 0], device=self.device, dtype=self.dtype)
        
        # spatial inputs
        ni = torch.tensor(loaded_data[..., 1], device=self.device, dtype=self.dtype)
        dni = torch.tensor(loaded_data[..., 2], device=self.device, dtype=self.dtype)
        E = torch.tensor(loaded_data[..., 3], device=self.device, dtype=self.dtype)
        Te = torch.tensor(loaded_data[..., 4], device=self.device, dtype=self.dtype)
        dTe = torch.tensor(loaded_data[..., 5], device=self.device, dtype=self.dtype)
        Ti = torch.tensor(loaded_data[..., 6], device=self.device, dtype=self.dtype)
        dTi = torch.tensor(loaded_data[..., 7], device=self.device, dtype=self.dtype)
        P = torch.tensor(loaded_data[..., 9], device=self.device, dtype=self.dtype)
        F = torch.tensor(loaded_data[..., 10], device=self.device, dtype=self.dtype)

        X = torch.stack([ni, dni, E, dTe, dTi, Te, Ti, P, F], dim=2)
        
        model_info = {
            "X": X,
            "var_names": ['n','dn','E','dT_e','dT_i','T_e','T_i','P','F'],
            "spatial":   [ True, True, True, True, True, True, True, False, False],
            "var_modes": ['rectified','signed','rectified','rectified','rectified','rectified','rectified','rectified','rectified'],
            "max_poly": 1,
            "max_inverse_power": 1,
            "max_total_order": 3,
            "exponent_increment": 0.5,
            "frac_allowed": [False, False, True, False, False, True, True, False, False],
            "exp_bounds": [(-1,1), (1,1), (0.5,1), (1,1), (1,1), (0.5,1), (0.5,1), (-1,1), (1,1)],
            "even_only_int": [False]*9,
            "verbose": verbose,
            "allow_const_fractional": True,
        }
        inputs_by_name = {"n": ni, "dn": dni, "E": E, "T_e": Te, "dT_e": dTe, "T_i": Ti, "dT_i": dTi, "P": P, "F": F}
        roles = {'n': 'state', 'dn': 'grad',
            'E': 'field',
            'dT_e': 'grad', 'dT_i': 'grad',
            'T_e': 'state','T_i': 'state',
            'P': 'param', 'F': 'param',
        }
        return model_info, inputs_by_name, fluxi, roles

    def create_library(self, model_info, tensors_by_name):
        feature_lib = FeatureLibrary(model_info['X'], 
                                    model_info['var_names'], 
                                    model_info['var_modes'], 
                                    model_info['spatial'],
                                    frac_allowed=model_info['frac_allowed'],
                                    even_only_int=model_info['even_only_int'],
                                    exp_bounds=model_info['exp_bounds'],
                                    device=self.device, dtype=self.dtype)

        Theta, syms, exps = feature_lib.full_monomial_basis(
            max_poly=model_info['max_poly'],
            max_inverse_power=model_info['max_inverse_power'],
            max_total_order=model_info['max_total_order'],
            exponent_increment=model_info['exponent_increment'],
            verbose=model_info['verbose'],
            allow_const_fractional=model_info['allow_const_fractional']
        )

        ni = tensors_by_name["n"]
        E = tensors_by_name["E"]
        P = tensors_by_name["P"]
        cx_tensor = ni * (E.abs().pow(0.5)) * P.clamp_min(1e-32).pow(-0.5)

        E_sym = sp.Symbol('E', commutative=False)
        n_sym = sp.Symbol('n', commutative=False)
        P_sym = sp.Symbol('P', commutative=False)
        cx_expr = E_sym**0.5 * n_sym * (1 / (P_sym**0.5))

        Theta = torch.cat([Theta, cx_tensor.unsqueeze(-1)], dim=2)
        syms.append(cx_expr)
        exps.append([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0])
        return Theta, syms, exps

    def region_flatten(self, Theta, mask, fluxi_p):
        M_local, N_local, d_local = Theta.shape

        if isinstance(mask, np.ndarray):
            assert mask.shape == (M_local, N_local), "mask shape must match Theta spatial shape"
            mask_t = torch.from_numpy(mask.astype(np.bool_)).to(self.device)
        else:
            mask_t = mask.to(self.device)
            assert mask_t.shape == (M_local, N_local), "mask tensor shape must match Theta spatial shape"

        rid_mat = torch.arange(M_local * N_local, device=self.device).reshape(M_local, N_local)
        rid_sel = rid_mat[mask_t]

        X_all = Theta.reshape(M_local * N_local, d_local)
        X_flat = X_all[rid_sel]

        y_all = fluxi_p.reshape(M_local * N_local)
        y_flat = y_all[rid_sel]

        sid_flat = (rid_sel // N_local).cpu().numpy().astype(np.int32)
        jid_flat = (rid_sel % N_local).cpu().numpy().astype(np.int32)

        return X_flat, y_flat, rid_sel, sid_flat, jid_flat
    
    def train_(self, Xb, yb, rid_tr, sid_tr, jid_tr, N):
        X_tr = Xb
        y_tr = yb.unsqueeze(-1) if yb.ndim == 1 else yb

        key = sid_tr.astype(np.int64) * N + jid_tr.astype(np.int64)
        order = np.argsort(key)
        X_tr, y_tr = X_tr[order], y_tr[order]
        sid_tr, jid_tr, rid_tr = sid_tr[order], jid_tr[order], rid_tr[order]

        unique, counts = np.unique(sid_tr, return_counts=True)
        inv_counts = {s: 1.0 / c for s, c in zip(unique, counts)}
        w_np = np.array([inv_counts[s] for s in sid_tr])
        w_tr = torch.tensor(w_np, device=o.device, dtype=o.dtype)

        lib_scaler = StandardizeLibrary()
        mu, sigma = lib_scaler.fit_weighted_lib_scaler(X_tr, w_tr)
        X_tr_std = lib_scaler.standardize_with(mu, sigma, X_tr)
        
        train_info = {
            "X" : X_tr_std,
            "y" : y_tr,
            "M" : len(np.unique(sid_tr)),
            "sid" : sid_tr,
            "jid" : jid_tr, 
            "rid" : rid_tr,
            "w_np" : w_np, 
            "w" : w_tr,
            "mu" : mu,
            "sigma" : sigma,
            "lib_scaler" : lib_scaler,
            "order" : order
        }
        return train_info

class StandardizeLibrary:
    def __init__(self):
        pass
    def fit_weighted_lib_scaler(self, X_train, w):
        w = w / w.sum()
        X1 = X_train[:, 1:] 
        mu = (w[:, None] * X1).sum(dim=0)
        var = (w[:, None] * (X1 - mu)**2).sum(dim=0)
        sigma = torch.sqrt(var).clamp_min(1e-32)
        return mu, sigma

    def standardize_with(self, mu, sigma, X):
        Z = X.clone()
        Z[:, 1:] = (X[:, 1:] - mu) / sigma
        return Z

    def backtransform_coefs(self, coef_std, mu, sigma):
        coef_orig = coef_std.clone()
        coef_orig[1:] = coef_std[1:] / sigma
        coef_orig[0]  = coef_std[0] - (mu * coef_orig[1:]).sum()
        return coef_orig

class FeatureLibrary:
    def __init__(self, U_stack, var_names, var_modes, spatial, 
                 frac_allowed=None, even_only_int=None, exp_bounds=None,
                 device=torch.device('cpu'), dtype=torch.float32):
        self.dtype = dtype
        self.device = device
        self.U_stack = U_stack
        self.var_names = var_names
        self.var_modes = var_modes
        self.spatial = spatial
        self.eps_rel = 1e-15
        self.eps_abs = 1e-32

        self.M, self.N, self.d = self.U_stack.shape
        
        self.frac_allowed = [False] * self.d if frac_allowed is None else list(frac_allowed)
        self.even_only_int = [False] * self.d if even_only_int is None else list(even_only_int)
        self.exp_bounds = exp_bounds

    def _is_int(self, e, tol=1e-12):
        e = float(e)
        return abs(e - round(e)) < tol

    def _is_derivative_name(self, name: str) -> bool:
        s = name.lower()
        if s.startswith('d'):
            return True
        tokens = ['grad', '_dx', '_dy', '_dz', 'd/dx', 'd/dy', 'd/dz']
        return any(tok in s for tok in tokens)

    def power_op(self, base, e, mode):
        with torch.no_grad():
            med = torch.median(base.abs().flatten())
        eps = self.eps_abs + self.eps_rel * (med + 0.0)

        e = float(e)
        # integer exponent
        if self._is_int(e):
            k = int(round(e))
            if k == 0:
                return torch.ones_like(base)
            # zero-guard but preserve sign for negative k
            base_safe = torch.where(base.abs() < eps, torch.sign(base) * eps, base)
            return base_safe ** k

        # fractional exponent
        if mode == 'signed':
            return torch.sign(base) * (base.abs().clamp_min(eps) ** e)
        elif mode == 'rectified':
            return (base.clamp_min(0.0) + eps) ** e
        elif mode == 'abs':
            return (base.abs().clamp_min(eps)) ** e
        else:
            raise ValueError(f"Unknown mode {mode}")

    def _per_var_exponent_grid(self, max_poly, max_inverse_power, exponent_increment):
        grids = []

        for i in range(self.d):
            if self.exp_bounds is not None and self.exp_bounds[i] is not None:
                emin_i, emax_i = self.exp_bounds[i]
            else:
                emin_i, emax_i = -max_inverse_power, max_poly
            
            exps = {0.0}

            int_start = int(np.ceil(emin_i))
            int_stop = int(np.floor(emax_i))
            int_vals = list(range(int_start, int_stop+1))

            if self.even_only_int[i]:
                int_vals = [k for k in int_vals if (k % 2 == 0)]
            for k in int_vals:
                exps.add(float(k))

            if self.frac_allowed[i]:
                raw = np.arange(emin_i, emax_i + 1e-12, exponent_increment, dtype=float)
                for e in raw:
                    if abs(e - round(e)) < 1e-12:
                        continue
                    exps.add(float(np.round(e, 6)))

            grid_i = sorted(exps)
            grids.append(grid_i)
        return grids
    
    def full_monomial_basis(
        self, max_poly=3, max_inverse_power=2, max_total_order=None, 
        exponent_increment=1.0, verbose=False, allow_const_fractional=True):        
        if self.var_modes is None:
            self.var_modes = ['rectified'] * self.d

        if max_total_order is None:
            max_total_order = max_poly + max_inverse_power

        rtol, atol = 1e-3, 1e-6
        is_const = []
        for i in range(self.d):
            feat = self.U_stack[:, :, i]
            base = feat.flatten()[0]
            is_flat = torch.allclose(feat, base, rtol=rtol, atol=atol)
            is_const.append(bool(is_flat))

        per_var_grids = self._per_var_exponent_grid(max_poly, max_inverse_power, exponent_increment)

        candidate_terms = []
        candidate_symbols = []
        candidate_exponents = []
        skipped_terms = []
        flagged = False

        for exponents in product(*per_var_grids):#, repeat=self.d):
            # skip all-zero exponents
            if all(abs(e) < self.eps_rel for e in exponents):
                continue
            # restrict by total absolute order
            if sum(abs(e) for e in exponents) > max_total_order:
                continue
            # restrict constants to 0 or 1 exponents            
            if not allow_const_fractional:
                bad = any(is_const[i] and (abs(e) > self.eps_rel and abs(e-1.0) > self.eps_rel)
                        for i, e in enumerate(exponents))
                if bad:
                    continue

            uses_nonspatial = any((abs(e) > self.eps_rel) and (not self.spatial[i]) for i, e in enumerate(exponents))
            uses_spatial = any((abs(e) > self.eps_rel) and self.spatial[i] for i, e in enumerate(exponents))
            if uses_nonspatial and (not uses_spatial):
                continue

            # build numerical term
            term = torch.ones((self.M, self.N, 1), dtype=self.U_stack.dtype, device=self.device)

            # collect symbol factors
            sym_factors = []

            for i, e in enumerate(exponents):
                if abs(e) < self.eps_rel:
                    continue
                base = self.U_stack[:, :, i:i+1]
                factor = self.power_op(base, e, self.var_modes[i])
                term = term * factor

                name_i = self.var_names[i] if self.var_names and i < len(self.var_names) else f"U_{i}"
                var_sym = sp.Symbol(name_i, commutative=False)

                e_float = float(e)
                if abs(e_float) < self.eps_rel:
                    continue

                if e_float > 0.0:
                    # positive exponent
                    if self._is_int(e_float):
                        k = int(round(e_float))
                        if k == 0:
                            continue
                        elif k == 1:
                            sym_factor = var_sym
                        else:
                            sym_factor = var_sym ** k
                    else:
                        sym_factor = var_sym ** e_float
                else:
                    # negative exponent -> reciprocal form
                    e_pos = -e_float
                    if self._is_int(e_pos):
                        k = int(round(e_pos))
                        if k == 0:
                            continue
                        elif k == 1:
                            sym_factor = 1 / var_sym
                        else:
                            sym_factor = 1 / (var_sym ** k)
                    else:
                        sym_factor = 1 / (var_sym ** e_pos)

                is_deriv = self._is_derivative_name(name_i)
                sym_factors.append((is_deriv, i, sym_factor))

            if not torch.isfinite(term).all():
                flagged = True
                sym_term_fallback = sp.Integer(1)
                for _, _, f in sym_factors:
                    sym_term_fallback *= f
                skipped_terms.append(sym_term_fallback)
                continue

            sym_factors.sort(key=lambda t: (t[0], t[1]))
            sym_term = sp.Integer(1)
            for _, _, f in sym_factors:
                sym_term *= f

            candidate_terms.append(term)
            candidate_symbols.append(sym_term)
            candidate_exponents.append(exponents)
            
        # constant term
        constant_term = torch.ones((self.M, self.N, 1), dtype=self.U_stack.dtype, device=self.device)
        candidate_terms.insert(0, constant_term)
        candidate_symbols.insert(0, sp.sympify(1))
        candidate_exponents.insert(0, tuple([0] * self.d))

        Theta_U = torch.cat(candidate_terms, dim=2)

        if verbose:
            if flagged: print(f"\nInf or NaN detected in {skipped_terms}. \nConsider nondimensionalizing the data before the expansion step.\n")
            print(f"Shape of library: {Theta_U.shape} [M, N, d]")
            for idx, sym in enumerate(candidate_symbols):
                mn = Theta_U[:, :, idx].min().item()
                mx = Theta_U[:, :, idx].max().item()
                print(f"{idx} | {sym}: min={mn}, max={mx}")

        return Theta_U, candidate_symbols, candidate_exponents