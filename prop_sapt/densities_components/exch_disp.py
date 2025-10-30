import numpy as np

from prop_sapt import Dimer

from .exch_disp_sinf_terms import (
    get_exch_disp_density_aa,
    get_exch_disp_density_ra,
    get_exch_disp_density_rr,
    get_exch_disp_density_bb,
    get_exch_disp_density_sb,
    get_exch_disp_density_ss,
)


def get_exch_disp_density(mol: Dimer, monomer: str) -> np.ndarray:

    rho_MO_exch_disp = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        rho_MO_exch_disp_ra = get_exch_disp_density_ra(mol)
        rho_MO_exch_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_disp_ra
        rho_MO_exch_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_disp_ra.T

        rho_MO_exch_disp[mol.slices["a"], mol.slices["a"]] = get_exch_disp_density_aa(
            mol
        )
        rho_MO_exch_disp[mol.slices["r"], mol.slices["r"]] = get_exch_disp_density_rr(
            mol
        )

    if monomer == "B":

        rho_MO_exch_disp_sb = get_exch_disp_density_sb(mol)
        rho_MO_exch_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_disp_sb
        rho_MO_exch_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_disp_sb.T

        rho_MO_exch_disp[mol.slices["b"], mol.slices["b"]] = get_exch_disp_density_bb(
            mol
        )
        rho_MO_exch_disp[mol.slices["s"], mol.slices["s"]] = get_exch_disp_density_ss(
            mol
        )

    return rho_MO_exch_disp


def get_exch_disp_s2_density(mol: Dimer, monomer: str) -> np.ndarray:

    rho_MO_exch_disp = np.zeros((mol.nmo, mol.nmo))

    # TODO
    if monomer == "A":

        rho_MO_exch_disp_ra = np.zeros((mol.sizes["r"], mol.sizes["a"]))
        rho_MO_exch_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_exch_disp_ra
        rho_MO_exch_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_exch_disp_ra.T

        rho_MO_exch_disp[mol.slices["a"], mol.slices["a"]] = np.zeros(
            (mol.sizes["a"], mol.sizes["a"])
        )
        rho_MO_exch_disp[mol.slices["r"], mol.slices["r"]] = np.zeros(
            (mol.sizes["r"], mol.sizes["r"])
        )

    # TODO
    if monomer == "B":

        rho_MO_exch_disp_sb = np.zeros((mol.sizes["s"], mol.sizes["b"]))
        rho_MO_exch_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_exch_disp_sb
        rho_MO_exch_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_exch_disp_sb.T

        rho_MO_exch_disp[mol.slices["b"], mol.slices["b"]] = np.zeros(
            (mol.sizes["b"], mol.sizes["b"])
        )
        rho_MO_exch_disp[mol.slices["s"], mol.slices["s"]] = np.zeros(
            (mol.sizes["s"], mol.sizes["s"])
        )

    return rho_MO_exch_disp
