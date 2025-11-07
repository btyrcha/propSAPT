import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer

from ..exch_disp2 import get_u_rsab_amplitudes


def get_exch_disp2_sinf_property_term2(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
):

    u_rsab = get_u_rsab_amplitudes(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )

    # TODO < V P R([V, R(X)]) > + < V P R([X, R(V)]) >
    x2_exch_disp_sinf_term2 = 0.0

    return x2_exch_disp_sinf_term2
