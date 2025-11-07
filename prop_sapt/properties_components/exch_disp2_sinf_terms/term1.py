import numpy as np
import opt_einsum as oe

from prop_sapt.molecule import Dimer


def get_exch_disp2_sinf_property_term1(
    mol: Dimer,
    xt_A_ra: np.ndarray,
    xt_B_sb: np.ndarray,
):

    xt_A_ar = xt_A_ra.T
    xt_B_bs = xt_B_sb.T

    vrx_ar = (
        -oe.contract("Ar,aA->ar", xt_A_ar, mol.omegaB_aa)
        + oe.contract("aR,Rr->ar", xt_A_ar, mol.omegaB_rr)
        + 2 * oe.contract("bs,Qar,Qbs->ar", xt_B_bs, mol.Qar, mol.Qbs)
    )

    vrx_bs = (
        -oe.contract("Bs,bB->bs", xt_B_bs, mol.omegaA_bb)
        + oe.contract("bS,Ss->bs", xt_B_bs, mol.omegaA_ss)
        + 2 * oe.contract("ar,Qar,Qbs->bs", xt_A_ar, mol.Qar, mol.Qbs)
    )

    vrx_abrs = (
        oe.contract("aR,QRr,Qbs->abrs", xt_A_ar, mol.Qrr, mol.Qbs)
        - oe.contract("Ar,QaA,Qbs->abrs", xt_A_ar, mol.Qaa, mol.Qbs)
        + oe.contract("bS,Qar,QSs->abrs", xt_B_bs, mol.Qar, mol.Qss)
        - oe.contract("Bs,Qar,QbB->abrs", xt_B_bs, mol.Qar, mol.Qbb)
    )

    # TODO < V R(X) | P R(V) >
    x2_exch_disp_sinf_term1 = 0.0

    return x2_exch_disp_sinf_term1
