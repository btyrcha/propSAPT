import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_exch_disp2_sinf_property_term3(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
):

    # TODO < V P R(X) R(V) >
    x2_exch_disp_sinf_term3 = 0.0

    return x2_exch_disp_sinf_term3
