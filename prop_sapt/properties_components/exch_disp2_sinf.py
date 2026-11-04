import numpy as np

from ..molecule import Dimer

from .exch_disp2_sinf_terms import (
    get_exch_disp2_sinf_property_term1,
    get_exch_disp2_sinf_property_term2,
    get_exch_disp2_sinf_property_term3,
)


def calc_exch_disp2_sinf_property(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
    prop_A_aa: np.ndarray,
    prop_A_rr: np.ndarray,
    prop_B_bb: np.ndarray,
    prop_B_ss: np.ndarray,
) -> float:

    # < V R(X) | P R(V) >
    x2_exch_disp_sinf = get_exch_disp2_sinf_property_term1(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
    )
    # < V P R([V, R(X)]) > + < V P R([X, R(V)]) >
    x2_exch_disp_sinf += get_exch_disp2_sinf_property_term2(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
        prop_A_aa=prop_A_aa,
        prop_A_rr=prop_A_rr,
        prop_B_bb=prop_B_bb,
        prop_B_ss=prop_B_ss,
    )
    # < V P R(X) R(V) >
    x2_exch_disp_sinf += get_exch_disp2_sinf_property_term3(
        mol=mol,
        xt_A_ra=xt_A_ra,
        xt_B_sb=xt_B_sb,
    )

    return float(x2_exch_disp_sinf)
