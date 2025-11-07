import numpy as np
import opt_einsum as oe

from prop_sapt import Dimer


def get_exch_disp_density_ra(mol: Dimer, theta_sinf_t_abrs: np.ndarray) -> np.ndarray:

    raise NotImplementedError("exch_disp density not yet implemented")

    rho_MO_exch_disp_ra = 0.5 * (
        # < R(X) | V P2 R(V) >
        # TODO vrx_ar terms
        # TODO vrx_bs terms
        # TODO vrx_abrs terms
        # TODO product-like terms
        # TODO < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
        +oe.contract("QRr,Qbs,abRs->ra", mol.Qrr, mol.Qbs, theta_sinf_t_abrs)
        - oe.contract("QaA,Qbs,Abrs->ra", mol.Qaa, mol.Qbs, theta_sinf_t_abrs)
        # TODO < V P2 R(X) R(V) >
    )

    rho_MO_exch_disp_ra = mol.cpscf("A", perturbation=rho_MO_exch_disp_ra.T)

    return rho_MO_exch_disp_ra
