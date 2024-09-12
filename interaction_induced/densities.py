import psi4
import numpy as np
from .molecule import Molecule
from .utils import trace_memory_peak


def density_mo_to_ao(
    mol: Molecule, monomer: str, density_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform density matrix from MO to AO.
    """

    if monomer == "A":
        return mol.C_A.dot(density_matrix).dot(mol.C_A.T)
    if monomer == "B":
        return mol.C_B.dot(density_matrix).dot(mol.C_B.T)


@trace_memory_peak
def calc_density_matirx(
    mol: Molecule, monomer: str, orbital_basis="AO"
) -> dict[str, np.ndarray]:
    """
    Calculate first-order induced chagne in the density matrix of the 'monomer'.
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

    if monomer == "A":
        rho_pol_ra = mol.cpscf("A")
        rho_exch_ra = 0.5 * mol.cpscf(
            "A", perturbation=mol.omega_exchB_ar + mol.omega_exchB_ra.T
        )

        rho_MO_pol[mol.ndocc_A :, : mol.ndocc_A] = rho_pol_ra
        rho_MO_pol[: mol.ndocc_A, mol.ndocc_A :] = rho_pol_ra.T

        rho_MO_exch[mol.ndocc_A :, : mol.ndocc_A] = rho_exch_ra
        rho_MO_exch[: mol.ndocc_A, mol.ndocc_A :] = rho_exch_ra.T

    if monomer == "B":
        rho_pol_sb = mol.cpscf("B")
        rho_exch_sb = 0.5 * mol.cpscf(
            "B", perturbation=mol.omega_exchA_bs + mol.omega_exchA_sb.T
        )

        rho_MO_pol[mol.ndocc_B :, : mol.ndocc_B] = rho_pol_sb
        rho_MO_pol[: mol.ndocc_B, mol.ndocc_B :] = rho_pol_sb.T

        rho_MO_exch[mol.ndocc_B :, : mol.ndocc_B] = rho_exch_sb
        rho_MO_exch[: mol.ndocc_B, mol.ndocc_B :] = rho_exch_sb.T

    if orbital_basis == "AO":
        return {
            "pol": density_mo_to_ao(mol, monomer, rho_MO_pol),
            "exch": density_mo_to_ao(mol, monomer, rho_MO_exch),
            "total": density_mo_to_ao(mol, monomer, rho_MO_pol + rho_MO_exch),
        }
    if orbital_basis == "MO":
        return {
            "pol": rho_MO_pol,
            "exch": rho_MO_exch,
            "total": rho_MO_pol + rho_MO_exch,
        }
