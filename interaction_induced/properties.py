"""
Calculations of SAPT dipole moment contributions.
"""

from time import time
import psi4
import numpy as np
import pandas as pd
import opt_einsum as oe

from .molecule import Molecule
from .utils import trace_memory_peak


def perform_property_contractions(
    mol: Molecule, property_matix: np.ndarray
) -> pd.DataFrame:
    """
    Calculates property contributions.

    Returns pandas.DataFrame with values
    of separate contributions and sumed up total.
    """

    # prepare results DataFrame
    results = pd.DataFrame()

    # variables for each axis
    prop_A_aa = mol.transform_ao_to_mo(property_matix, "aa")
    prop_A_ar = mol.transform_ao_to_mo(property_matix, "ar")
    # prop_A_rr = mol.transform_ao_to_mo(property_matix, "rr")

    prop_B_bb = mol.transform_ao_to_mo(property_matix, "bb")
    prop_B_bs = mol.transform_ao_to_mo(property_matix, "bs")
    # prop_B_ss = mol.transform_ao_to_mo(property_matix, "ss")

    # relaxed amplitudes with dipole moment
    xt_A_ra = mol.cpscf("A", perturbation=prop_A_ar)
    # xt_A_ar = xt_A_ra.T
    xt_B_sb = mol.cpscf("B", perturbation=prop_B_bs)
    # xt_B_bs = xt_B_sb.T

    results["x0_A"] = np.array([2 * oe.contract("aa", prop_A_aa)])
    results["x0_B"] = np.array([2 * oe.contract("bb", prop_B_bb)])

    results["x1_pol,r"] = np.array(
        [
            4 * oe.contract("ra,ar", xt_A_ra, mol.omegaB_ar)
            + 4 * oe.contract("sb,bs", xt_B_sb, mol.omegaA_bs)
        ]
    )

    results["x1_exch,r"] = np.array(
        [
            2 * oe.contract("ra,ar", xt_A_ra, mol.omega_exchB_ar)
            + 2 * oe.contract("ra,ra", mol.omega_exchB_ra, xt_A_ra)
            + 2 * oe.contract("sb,bs", xt_B_sb, mol.omega_exchA_bs)
            + 2 * oe.contract("sb,sb", mol.omega_exchA_sb, xt_B_sb)
        ]
    )

    results["x_induced"] = results["x1_pol,r"] + results["x1_exch,r"]

    return results


def calc_induced_dipole(mol: Molecule):
    """
    Calculate interaction-induced dipole moment
    along X, Y and Z axes.
    """

    # grab nuclear dipole moment
    nuc_dipole = mol.dimer.nuclear_dipole()
    nuc_dipole = np.array([nuc_dipole[0], nuc_dipole[1], nuc_dipole[2]])

    # grab AO dipole moment matrices
    d_vec = mol.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]

    # prepare results
    results = pd.DataFrame()

    # calculate for each axis
    for d_i, i in zip(d_vec, ["X", "Y", "Z"]):
        res_i = perform_property_contractions(mol, d_i)
        res_i["axis"] = i
        results = pd.concat([results, res_i])

    del d_vec

    results.set_index("axis", inplace=True)
    results["nuc"] = nuc_dipole

    return results


@trace_memory_peak
def calc_property(mol: Molecule, prop: str | np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Calculation of interaction-induced property.
    """

    ### Start total time of calculations
    total_time = time()

    ### Results output file
    results_fname = kwargs.get("results", "results.csv")

    if prop == "dipole":
        ### Dipole moment calculations
        results = calc_induced_dipole(mol)

    else:
        results = perform_property_contractions(mol, prop)

    ### Results saving to file
    results.to_csv(results_fname)

    ### End of calculations
    total_time = time() - total_time
    psi4.core.print_out(
        f"\nInteraction-induced property calculaitons took {total_time:.2f} seconds.\n"
    )
    psi4.core.clean()

    return results
