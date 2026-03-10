from dataclasses import dataclass
import torch
import numpy as np

@dataclass(frozen=True)
class PhysicalConstants:
    # physical constants
    q = 1.602176634e-19 # fundamental charge [C]
    eps0 = 8.8541878128e-12 # vacuum constant [F/m]
    m_i = 6.634e-26 # ar mass [kg]
    m_e = 9.1093837e-31 # electron mass [kg]
    mTorr_Pa = 133.322 * 1e-3 # 1 Mtorr -> 133.322e-3 Pa
    MHz_Hz = 1e6 # 1MHz -> Hz
    kb = 1.380649e-23 # boltzmann constant [JK^-1]
    K_eV = 11600
    mean_pressure = 15.422872340425531 # [mPa]
    mean_frequency = 32.81808510638297 # [MHz]

@dataclass(frozen=True)
class OtherConstants:
    # other constants
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32   
    delta = 1e-32 # non-zero division 
    upperbound = 1
    max_coef_thr = 1e-4
    sig_figs = 5
    random_state = 42
    rng = np.random.default_rng(seed=random_state)


