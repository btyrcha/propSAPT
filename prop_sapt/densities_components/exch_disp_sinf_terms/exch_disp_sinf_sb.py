import numpy as np

from prop_sapt import Dimer


def get_exch_disp_density_sb(mol: Dimer, theta_sinf_t_abrs: np.ndarray) -> np.ndarray:

    raise NotImplementedError("exch_disp density not yet implemented")

    rho_MO_exch_disp_sb = 0.5 * (
        # < R(X) | V P2 R(V) >
        # TODO vrx_ar terms
        # TODO vrx_bs terms
        # TODO vrx_abrs terms
        # TODO product-like terms
        # TODO < V P2 R([V, R(X)]) > + < V P2 R([X, R(V)]) >
        # TODO < V P2 R(X) R(V) >
    )

    rho_MO_exch_disp_sb = mol.cpscf("B", perturbation=rho_MO_exch_disp_sb.T)

    return rho_MO_exch_disp_sb
