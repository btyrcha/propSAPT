import psi4
import numpy as np
import opt_einsum as oe
from .molecule import Dimer
from .utils import trace_memory_peak


def density_mo_to_ao(
    mol: Dimer, monomer: str, density_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform density matrix from MO to AO.

    Args:
        mol (Dimer): A dimer system.
        monomer (str): Select a monomer, either "A" or "B".
        density_matrix (np.ndarray): Density matrix to transform.

    Returns:
        np.ndarray: Transformed density matrix.
    """

    if monomer == "A":
        return mol.C_A.dot(density_matrix).dot(mol.C_A.T)
    elif monomer == "B":
        return mol.C_B.dot(density_matrix).dot(mol.C_B.T)
    else:
        raise ValueError(f"Invalid monomer: {monomer}")


@trace_memory_peak
def calc_density_matrix(
    mol: Dimer, monomer: str, orbital_basis="AO"
) -> dict[str, np.ndarray]:
    """
    Calculate first-order induced chagne in the density matrix of the 'monomer'.

    Args:
        mol (Dimer): A dimer system.
        monomer (str): Select a monomer, either "A" or "B".
        orbital_basis (str, optional): Select orbital baisi of the returned density matrix, either
            "AO" or "MO". Defaults to "AO".

    Returns:
        dict[str, np.ndarray]: Dictionary with three matrices, with keys "pol", "exch" and "total".
            "pol": First-order polarisation correction to the change in the monomer density matrix.
            "exch": First-order exchange correction to the change in the monomer density matrix.
            "total": Sum of polarisation and exchange corrections.
    """

    if monomer not in ["A", "B"]:
        psi4.core.clean()
        raise ValueError(f"'{monomer}' is not a valid monomer for density matrix.")

    orbital_basis = orbital_basis.upper()
    if orbital_basis not in ["MO", "AO"]:
        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )

    rho_MO_pol = np.zeros((mol.nmo, mol.nmo))
    rho_MO_exch = np.zeros((mol.nmo, mol.nmo))
    rho_MO_ind = np.zeros((mol.nmo, mol.nmo))
    rho_MO_disp = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":
        rho_pol_ra = mol.get_cpscf_ra()
        rho_exch_ra = 0.5 * mol.cpscf(
            "A", perturbation=mol.omega_exchB_ar + mol.omega_exchB_ra.T
        )

        # first-order polarisation
        rho_MO_pol[mol.ndocc_A :, : mol.ndocc_A] = rho_pol_ra
        rho_MO_pol[: mol.ndocc_A, mol.ndocc_A :] = rho_pol_ra.T

        # first-order exchange
        rho_MO_exch[mol.ndocc_A :, : mol.ndocc_A] = rho_exch_ra
        rho_MO_exch[: mol.ndocc_A, mol.ndocc_A :] = rho_exch_ra.T

        # second-order induction
        rho_MO_ind_ra = (
            +2 * oe.contract("Qar,Qbs,sb->ra", mol.Qar, mol.Qbs, mol.get_cpscf_sb())
            + 2 * oe.contract("sb,Qar,Qbs->ra", mol.get_cpscf_sb(), mol.Qar, mol.Qbs)
            - oe.contract("ra,aA->rA", mol.get_cpscf_ra(), mol.omegaB_aa)
            + oe.contract("ra,Rr->Ra", mol.get_cpscf_ra(), mol.omegaB_rr)
        )
        rho_MO_ind_ra = mol.cpscf("A", perturbation=rho_MO_ind_ra.T)

        rho_MO_ind[mol.ndocc_A :, : mol.ndocc_A] = rho_MO_ind_ra
        rho_MO_ind[: mol.ndocc_A, mol.ndocc_A :] = rho_MO_ind_ra.T

        rho_MO_ind[: mol.ndocc_A, : mol.ndocc_A] = -oe.contract(
            "rA,ra->aA", mol.get_cpscf_ra(), mol.get_cpscf_ra()
        )
        rho_MO_ind[mol.ndocc_A :, mol.ndocc_A :] = +oe.contract(
            "Ra,ra->Rr", mol.get_cpscf_ra(), mol.get_cpscf_ra()
        )

        # second-order dispersion
        rho_MO_disp_ra = 2 * oe.contract(
            "rsab,QRr,Qbs->Ra", mol.t_rsab, mol.Qrr, mol.Qbs
        ) - 2 * oe.contract("rsab,QaA,Qbs->rA", mol.t_rsab, mol.Qaa, mol.Qbs)
        rho_MO_disp_ra = mol.cpscf("A", perturbation=rho_MO_disp_ra.T)

        rho_MO_disp[mol.ndocc_A :, : mol.ndocc_A] = rho_MO_disp_ra
        rho_MO_disp[: mol.ndocc_A, mol.ndocc_A :] = rho_MO_disp_ra.T

        rho_MO_disp[: mol.ndocc_A, : mol.ndocc_A] = -2 * oe.contract(
            "rsAb,rsab->aA", mol.t_rsab, mol.t_rsab
        )
        rho_MO_disp[mol.ndocc_A :, mol.ndocc_A :] = +2 * oe.contract(
            "Rsab,rsab->Rr", mol.t_rsab, mol.t_rsab
        )

    if monomer == "B":
        rho_pol_sb = mol.get_cpscf_sb()
        rho_exch_sb = 0.5 * mol.cpscf(
            "B", perturbation=mol.omega_exchA_bs + mol.omega_exchA_sb.T
        )

        # first-order polarisation
        rho_MO_pol[mol.ndocc_B :, : mol.ndocc_B] = rho_pol_sb
        rho_MO_pol[: mol.ndocc_B, mol.ndocc_B :] = rho_pol_sb.T

        # first-order exchange
        rho_MO_exch[mol.ndocc_B :, : mol.ndocc_B] = rho_exch_sb
        rho_MO_exch[: mol.ndocc_B, mol.ndocc_B :] = rho_exch_sb.T

        # second-order induction
        rho_MO_ind_sb = (
            +2 * oe.contract("Qar,Qbs,ra->sb", mol.Qar, mol.Qbs, mol.get_cpscf_ra())
            + 2 * oe.contract("ra,Qar,Qbs->sb", mol.get_cpscf_ra(), mol.Qar, mol.Qbs)
            - oe.contract("sb,bB->sB", mol.get_cpscf_sb(), mol.omegaA_bb)
            + oe.contract("sb,Ss->Sb", mol.get_cpscf_sb(), mol.omegaA_ss)
        )
        rho_MO_ind_sb = mol.cpscf("B", perturbation=rho_MO_ind_sb.T)

        rho_MO_ind[mol.ndocc_B :, : mol.ndocc_B] = rho_MO_ind_sb
        rho_MO_ind[: mol.ndocc_B, mol.ndocc_B :] = rho_MO_ind_sb.T

        rho_MO_ind[: mol.ndocc_B, : mol.ndocc_B] = -oe.contract(
            "sB,sb->bB", mol.get_cpscf_sb(), mol.get_cpscf_sb()
        )
        rho_MO_ind[mol.ndocc_B :, mol.ndocc_B :] = +oe.contract(
            "Sb,sb->Ss", mol.get_cpscf_sb(), mol.get_cpscf_sb()
        )

        # second-order dispersion
        rho_MO_disp_sb = +2 * oe.contract(
            "rsab,Qar,QSs->Sb", mol.t_rsab, mol.Qar, mol.Qss
        ) - 2 * oe.contract("rsab,Qar,QbB->sB", mol.t_rsab, mol.Qar, mol.Qbb)
        rho_MO_disp_sb = mol.cpscf("B", perturbation=rho_MO_disp_sb.T)

        rho_MO_disp[mol.ndocc_B :, : mol.ndocc_B] = rho_MO_disp_sb
        rho_MO_disp[: mol.ndocc_B, mol.ndocc_B :] = rho_MO_disp_sb.T

        rho_MO_disp[: mol.ndocc_B, : mol.ndocc_B] = -2 * oe.contract(
            "rsaB,rsab->bB", mol.t_rsab, mol.t_rsab
        )
        rho_MO_disp[mol.ndocc_B :, mol.ndocc_B :] = +2 * oe.contract(
            "rSab,rsab->Ss", mol.t_rsab, mol.t_rsab
        )

    # sum up all contributions
    rho_MO_total = (
        rho_MO_pol
        + rho_MO_exch
        # + rho_MO_ind
        # + rho_MO_disp
    )

    if orbital_basis == "AO":
        return {
            "pol": density_mo_to_ao(mol, monomer, rho_MO_pol),
            "exch": density_mo_to_ao(mol, monomer, rho_MO_exch),
            "ind": density_mo_to_ao(mol, monomer, rho_MO_ind),
            "disp": density_mo_to_ao(mol, monomer, rho_MO_disp),
            "total": density_mo_to_ao(mol, monomer, rho_MO_total),
        }
    elif orbital_basis == "MO":
        return {
            "pol": rho_MO_pol,
            "exch": rho_MO_exch,
            "ind": rho_MO_ind,
            "disp": rho_MO_disp,
            "total": rho_MO_total,
        }
    else:
        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )
