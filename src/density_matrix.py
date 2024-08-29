import psi4
import numpy as np
from helper_SAPT_DF import helper_SAPT
from sinfinity import sinfinity


def density_mo_to_ao(
    sapt: helper_SAPT, density_matrix: np.ndarray, monomer: str
) -> np.ndarray:
    """
    Transform density matrix from MO to AO.
    """

    if monomer == "A":
        return sapt.C_A.dot(density_matrix).dot(sapt.C_A.T)
    if monomer == "B":
        return sapt.C_B.dot(density_matrix).dot(sapt.C_B.T)


def get_density_matirx(
    sapt: helper_SAPT, sinf: sinfinity, monomer: str, orbital_basis="MO"
) -> np.ndarray:
    """
    Calculate first-order induced chagne in the density matrix of the 'monomer'.
    """

    if monomer not in ["A", "B"]:
        psi4.core.clean()
        raise ValueError(f"'{monomer}' is not a valid monomer for density matrix.")

    if orbital_basis not in ["MO", "AO"]:
        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )

    rho_MO_pol = np.zeros((sapt.nmo, sapt.nmo))
    rho_MO_exch = np.zeros((sapt.nmo, sapt.nmo))

    if monomer == "A":
        rho_pol_ra = sapt.chf("B")
        rho_exch_ra = 0.5 * sapt.chf(
            "B", perturbation=sinf.omega_exchB_ar + sinf.omega_exchB_ra.T
        )

        rho_MO_pol[sapt.ndocc_A :, : sapt.ndocc_A] = rho_pol_ra
        rho_MO_pol[: sapt.ndocc_A, sapt.ndocc_A :] = rho_pol_ra.T

        rho_MO_exch[sapt.ndocc_A :, : sapt.ndocc_A] = rho_exch_ra
        rho_MO_exch[: sapt.ndocc_A, sapt.ndocc_A :] = rho_exch_ra.T

    if monomer == "B":
        rho_pol_sb = sapt.chf("A")
        rho_exch_sb = 0.5 * sapt.chf(
            "A", perturbation=sinf.omega_exchA_bs + sinf.omega_exchA_sb.T
        )

        rho_MO_pol[sapt.ndocc_B :, : sapt.ndocc_B] = rho_pol_sb
        rho_MO_pol[: sapt.ndocc_B, sapt.ndocc_B :] = rho_pol_sb.T

        rho_MO_exch[sapt.ndocc_B :, : sapt.ndocc_B] = rho_exch_sb
        rho_MO_exch[: sapt.ndocc_B, sapt.ndocc_B :] = rho_exch_sb.T

    if orbital_basis == "MO":
        return {
            "pol": rho_MO_pol,
            "exch": rho_MO_exch,
        }
    elif orbital_basis == "AO":
        return {
            "pol": density_mo_to_ao(sapt, rho_MO_pol, monomer),
            "exch": density_mo_to_ao(sapt, rho_MO_exch, monomer),
        }
