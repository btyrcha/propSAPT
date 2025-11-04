import psi4
import numpy as np
import opt_einsum as oe
from .molecule import Dimer
from .utils import trace_memory_peak, CalcTimer
from .densities_components import (
    get_exch_disp_s2_density,
    get_exch_ind_s2_density,
)


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

        with CalcTimer("First-order Polarisation density calculation"):
            rho_pol_ra = mol.get_cpscf_ra()

            # first-order polarisation
            rho_MO_pol[mol.slices["r"], mol.slices["a"]] = rho_pol_ra
            rho_MO_pol[mol.slices["a"], mol.slices["r"]] = rho_pol_ra.T

        with CalcTimer("First-order Exchange density calculation"):
            rho_exch_ra = 0.5 * mol.cpscf(
                "A", perturbation=mol.omega_exchB_ar + mol.omega_exchB_ra.T
            )

            # first-order exchange
            rho_MO_exch[mol.slices["r"], mol.slices["a"]] = rho_exch_ra
            rho_MO_exch[mol.slices["a"], mol.slices["r"]] = rho_exch_ra.T

        with CalcTimer("Second-order Induction density calculation"):
            # second-order induction
            rho_MO_ind_ra = (
                +2 * oe.contract("Qar,Qbs,sb->ra", mol.Qar, mol.Qbs, mol.get_cpscf_sb())
                + 2
                * oe.contract("sb,Qar,Qbs->ra", mol.get_cpscf_sb(), mol.Qar, mol.Qbs)
                - oe.contract("ra,aA->rA", mol.get_cpscf_ra(), mol.omegaB_aa)
                + oe.contract("ra,Rr->Ra", mol.get_cpscf_ra(), mol.omegaB_rr)
            )
            rho_MO_ind_ra = mol.cpscf("A", perturbation=rho_MO_ind_ra.T)

            rho_MO_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_ind_ra
            rho_MO_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_ind_ra.T

            rho_MO_ind[mol.slices["a"], mol.slices["a"]] = -oe.contract(
                "rA,ra->aA", mol.get_cpscf_ra(), mol.get_cpscf_ra()
            )
            rho_MO_ind[mol.slices["r"], mol.slices["r"]] = +oe.contract(
                "Ra,ra->Rr", mol.get_cpscf_ra(), mol.get_cpscf_ra()
            )

        with CalcTimer("Second-order Dispersion density calculation"):
            # second-order dispersion
            rho_MO_disp_ra = 2 * oe.contract(
                "rsab,QRr,Qbs->Ra", mol.t_rsab, mol.Qrr, mol.Qbs
            ) - 2 * oe.contract("rsab,QaA,Qbs->rA", mol.t_rsab, mol.Qaa, mol.Qbs)
            rho_MO_disp_ra = mol.cpscf("A", perturbation=rho_MO_disp_ra.T)

            rho_MO_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_disp_ra
            rho_MO_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_disp_ra.T

            rho_MO_disp[mol.slices["a"], mol.slices["a"]] = -2 * oe.contract(
                "rsAb,rsab->aA", mol.t_rsab, mol.t_rsab
            )
            rho_MO_disp[mol.slices["r"], mol.slices["r"]] = +2 * oe.contract(
                "Rsab,rsab->Rr", mol.t_rsab, mol.t_rsab
            )

    if monomer == "B":

        with CalcTimer("First-order Polarisation density calculation"):
            rho_pol_sb = mol.get_cpscf_sb()

            # first-order polarisation
            rho_MO_pol[mol.slices["s"], mol.slices["b"]] = rho_pol_sb
            rho_MO_pol[mol.slices["b"], mol.slices["s"]] = rho_pol_sb.T

        with CalcTimer("First-order Exchange density calculation"):
            rho_exch_sb = 0.5 * mol.cpscf(
                "B", perturbation=mol.omega_exchA_bs + mol.omega_exchA_sb.T
            )

            # first-order exchange
            rho_MO_exch[mol.slices["s"], mol.slices["b"]] = rho_exch_sb
            rho_MO_exch[mol.slices["b"], mol.slices["s"]] = rho_exch_sb.T

        with CalcTimer("Second-order Induction density calculation"):
            # second-order induction
            rho_MO_ind_sb = (
                +2 * oe.contract("Qar,Qbs,ra->sb", mol.Qar, mol.Qbs, mol.get_cpscf_ra())
                + 2
                * oe.contract("ra,Qar,Qbs->sb", mol.get_cpscf_ra(), mol.Qar, mol.Qbs)
                - oe.contract("sb,bB->sB", mol.get_cpscf_sb(), mol.omegaA_bb)
                + oe.contract("sb,Ss->Sb", mol.get_cpscf_sb(), mol.omegaA_ss)
            )
            rho_MO_ind_sb = mol.cpscf("B", perturbation=rho_MO_ind_sb.T)

            rho_MO_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_ind_sb
            rho_MO_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_ind_sb.T

            rho_MO_ind[mol.slices["b"], mol.slices["b"]] = -oe.contract(
                "sB,sb->bB", mol.get_cpscf_sb(), mol.get_cpscf_sb()
            )
            rho_MO_ind[mol.slices["s"], mol.slices["s"]] = +oe.contract(
                "Sb,sb->Ss", mol.get_cpscf_sb(), mol.get_cpscf_sb()
            )

        with CalcTimer("Second-order Dispersion density calculation"):
            # second-order dispersion
            rho_MO_disp_sb = +2 * oe.contract(
                "rsab,Qar,QSs->Sb", mol.t_rsab, mol.Qar, mol.Qss
            ) - 2 * oe.contract("rsab,Qar,QbB->sB", mol.t_rsab, mol.Qar, mol.Qbb)
            rho_MO_disp_sb = mol.cpscf("B", perturbation=rho_MO_disp_sb.T)

            rho_MO_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_disp_sb
            rho_MO_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_disp_sb.T

            rho_MO_disp[mol.slices["b"], mol.slices["b"]] = -2 * oe.contract(
                "rsaB,rsab->bB", mol.t_rsab, mol.t_rsab
            )
            rho_MO_disp[mol.slices["s"], mol.slices["s"]] = +2 * oe.contract(
                "rSab,rsab->Ss", mol.t_rsab, mol.t_rsab
            )

    # second-order exchange corrections in S^2 approximation
    with CalcTimer("Second-order Exchange-Induction (S^2) density calculation"):
        rho_MO_exch_ind_s2 = get_exch_ind_s2_density(mol, monomer)

    with CalcTimer("Second-order Exchange-Dispersion (S^2) density calculation"):
        rho_MO_exch_disp_s2 = get_exch_disp_s2_density(mol, monomer)

    # sum up all contributions
    rho_MO_total = (
        rho_MO_pol
        + rho_MO_exch
        + rho_MO_ind
        + rho_MO_exch_ind_s2  # NOTE: change into Sinf version when ready
        + rho_MO_disp
        + rho_MO_exch_disp_s2  # NOTE: change into Sinf version when ready
    )

    if orbital_basis == "AO":
        return {
            "pol": density_mo_to_ao(mol, monomer, rho_MO_pol),
            "exch": density_mo_to_ao(mol, monomer, rho_MO_exch),
            "ind": density_mo_to_ao(mol, monomer, rho_MO_ind),
            "exch-ind_S2": density_mo_to_ao(mol, monomer, rho_MO_exch_ind_s2),
            "disp": density_mo_to_ao(mol, monomer, rho_MO_disp),
            "exch-disp_S2": density_mo_to_ao(mol, monomer, rho_MO_exch_disp_s2),
            "total": density_mo_to_ao(mol, monomer, rho_MO_total),
        }
    elif orbital_basis == "MO":
        return {
            "pol": rho_MO_pol,
            "exch": rho_MO_exch,
            "ind": rho_MO_ind,
            "exch-ind_S2": rho_MO_exch_ind_s2,
            "disp": rho_MO_disp,
            "exch-disp_S2": rho_MO_exch_disp_s2,
            "total": rho_MO_total,
        }
    else:
        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )
