import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer

from .exch_disp2 import get_u_rsab_amplitudes


def calc_exch_disp2_sinf_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
) -> np.ndarray:

    xt_A_ar = xt_A_ra.T
    xt_B_bs = xt_B_sb.T

    u_rsab = get_u_rsab_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    x2_exch_disp_sinf = np.array(
        [
            # < R(X) | V P R(V) > TODO
            0.0
            # < V P R([V, R(X)]) > + < V P R([X, R(V)]) >
            + 0.0
            # < V P R(X) R(V) >
            + 0.0
        ]
    )

    return x2_exch_disp_sinf
