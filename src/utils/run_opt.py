import casadi as cas 
import numpy as np

def run_opt(f_train, lmb1, lmb2, initial_coef):
    opti = cas.Opti()
    # flux variables
    f_Theta_DM = cas.DM(f_train["X"].tolist())
    f_y_DM = cas.DM(f_train["y"].tolist())
    f_w_DM = cas.DM(f_train["w_np"][:, None])
    f_coef_plus  = opti.variable(f_train["X"].shape[-1]) 
    f_coef_minus = opti.variable(f_train["X"].shape[-1])
    opti.subject_to(f_coef_plus >= 0) 
    opti.subject_to(f_coef_minus >= 0)
    opti.set_initial(f_coef_plus, np.maximum(initial_coef, 0))
    opti.set_initial(f_coef_minus, -np.minimum(initial_coef, 0))
    opti.subject_to(f_coef_plus <= cas.DM(1))
    opti.subject_to(f_coef_minus <= cas.DM(1))
    f_coef = f_coef_plus - f_coef_minus
    f_l1 = cas.sum1(f_coef_plus[1:]) + cas.sum1(f_coef_minus[1:])
    f_l2 = cas.sumsqr(f_coef[1:])
    f_pred = cas.reshape(f_Theta_DM @ f_coef, (-1, 1))
    f_mse = cas.sumsqr(cas.sqrt(f_w_DM) * (f_y_DM - f_pred)) / float(f_train["w_np"].sum())

    l1 = lmb1 * f_l1
    l2 = lmb2 * f_l2

    objective = f_mse + l1 + l2
    opti.minimize(objective)

    p_opts = {'verbose': False, 'expand': False, 'print_time': 0,}
    s_opts = {}
    opti.solver('ipopt', p_opts, s_opts)

    try: 
        sol = opti.solve()
    except RuntimeError: 
        sol = opti.debug

    f_cp = sol.value(f_coef_plus)
    f_cm = sol.value(f_coef_minus)
    f_coef_star_std = f_cp - f_cm
    return f_coef_star_std