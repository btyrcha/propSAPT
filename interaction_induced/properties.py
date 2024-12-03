"""
Calculations of SAPT one-electron properties.
"""

from time import time
import psi4
import numpy as np
import pandas as pd
import opt_einsum as oe

from .molecule import Dimer
from .utils import trace_memory_peak


def perform_property_contractions(
    mol: Dimer, property_matix: np.ndarray
) -> pd.DataFrame:
    """
    Calculates property contributions. Returns pandas.DataFrame with values of separate
    contributions and sumed up total.

    Args:
        mol (Dimer): A dimer system.
        property_matix (np.ndarray): A matrix representation of the operator corresponding
            to a property to calculate, in AO basis.

    Returns:
        pd.DataFrame: DataFrame with results.
    """

    # prepare results DataFrame
    results = pd.DataFrame()

    # variables for each axis
    prop_A_aa = mol.transform_ao_to_mo(property_matix, "aa")
    prop_A_ar = mol.transform_ao_to_mo(property_matix, "ar")
    prop_A_rr = mol.transform_ao_to_mo(property_matix, "rr")

    prop_B_bb = mol.transform_ao_to_mo(property_matix, "bb")
    prop_B_bs = mol.transform_ao_to_mo(property_matix, "bs")
    prop_B_ss = mol.transform_ao_to_mo(property_matix, "ss")

    # relaxed amplitudes with dipole moment
    xt_A_ra = mol.cpscf("A", perturbation=prop_A_ar)
    # xt_A_ar = xt_A_ra.T
    xt_B_sb = mol.cpscf("B", perturbation=prop_B_bs)
    # xt_B_bs = xt_B_sb.T

    # First-order

    results["x0_A"] = np.array([2 * oe.contract("aa", prop_A_aa)])
    results["x0_B"] = np.array([2 * oe.contract("bb", prop_B_bb)])

    results["x1_pol,r"] = np.array(
        [
            4 * oe.contract("ra,ar", xt_A_ra, mol.omegaB_ar)
            + 4 * oe.contract("sb,bs", xt_B_sb, mol.omegaA_bs)
        ]
    )

    results["x1_exch,r_S2"] = np.array(
        [
            2 * oe.contract("ra,ar", xt_A_ra, mol.omega_exchB_ar_S2)
            + 2 * oe.contract("ra,ra", mol.omega_exchB_ra_S2, xt_A_ra)
            + 2 * oe.contract("sb,bs", xt_B_sb, mol.omega_exchA_bs_S2)
            + 2 * oe.contract("sb,sb", mol.omega_exchA_sb_S2, xt_B_sb)
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

    ### Second-order

    alpha = results["x1_exch,r"] / results["x1_pol,r"]

    # TODO: verify on diagrams and test
    results["x2_ind,r"] = (
        # 2 Re <R(X)|V R(V)>
        -4 * oe.contract("ra,ac,rc", mol.get_cpscf_ra(), mol.omegaB_aa, xt_A_ra)
        + 4 * oe.contract("ra,cr,ca", mol.get_cpscf_ra(), mol.omegaB_rr, xt_A_ra)
        + 8
        * oe.contract("sb,Qar,Qbs,ra", mol.get_cpscf_sb(), mol.Qar, mol.Qbs, xt_A_ra)
        - 4 * oe.contract("sb,bc,sc", mol.get_cpscf_sb(), mol.omegaA_bb, xt_B_sb)
        + 4 * oe.contract("sb,cs,cb", mol.get_cpscf_sb(), mol.omegaA_ss, xt_B_sb)
        + 8
        * oe.contract("ra,Qar,Qbs,sb", mol.get_cpscf_ra(), mol.Qar, mol.Qbs, xt_B_sb)
        # <R(V)|d R(V)>
        - 2 * oe.contract("ac,rc,ra", prop_A_aa, mol.get_cpscf_ra(), mol.get_cpscf_ra())
        + 2 * oe.contract("cr,ca,ra", prop_A_rr, mol.get_cpscf_ra(), mol.get_cpscf_ra())
        - 2 * oe.contract("bc,sc,sb", prop_B_bb, mol.get_cpscf_sb(), mol.get_cpscf_sb())
        + 2 * oe.contract("cs,cb,sb", prop_B_ss, mol.get_cpscf_sb(), mol.get_cpscf_sb())
        + 8
        * oe.contract("ra,Qar,Qbs,sb", xt_A_ra, mol.Qar, mol.Qbs, mol.get_cpscf_sb())
        + 8
        * oe.contract("sb,Qar,Qbs,ra", xt_B_sb, mol.Qar, mol.Qbs, mol.get_cpscf_ra())
    )

    # scaled exchange-induction
    results["x2_exch-ind,r"] = alpha * results["x2_ind,r"]

    # TODO: verify on diagrams and test
    results["x2_disp"] = (
        # 2 Re <R(X)|V R(V)>
        # NOTE some response not coupled
        8 * oe.contract("rsab,Qcr,Qbs,ca", mol.t_rsab, mol.Qrr, mol.Qbs, xt_A_ra)
        - 8 * oe.contract("rsab,Qac,Qbs,rc", mol.t_rsab, mol.Qaa, mol.Qbs, xt_A_ra)
        + 8 * oe.contract("rsab,Qar,Qcs,cb", mol.t_rsab, mol.Qar, mol.Qss, xt_B_sb)
        - 8 * oe.contract("rsab,Qar,Qbc,sc", mol.t_rsab, mol.Qar, mol.Qbb, xt_B_sb)
        # <R(V)|d R(V)>
        # NOTE not coupled no response
        - 4 * oe.contract("ac,rscb,rsab", prop_A_aa, mol.t_rsab, mol.t_rsab)
        - 4 * oe.contract("bc,rsac,rsab", prop_B_bb, mol.t_rsab, mol.t_rsab)
        + 4 * oe.contract("cr,csab,rsab", prop_A_rr, mol.t_rsab, mol.t_rsab)
        + 4 * oe.contract("cs,rcab,rsab", prop_B_ss, mol.t_rsab, mol.t_rsab)
    )

    # scaled exchange-dispersion
    results["x2_exch-disp"] = alpha * results["x2_disp"]

    results["x_induced"] = (
        results["x1_pol,r"]
        + results["x1_exch,r"]
        + results["x2_ind,r"]
        + results["x2_exch-ind,r"]
        + results["x2_disp"]
        + results["x2_exch-disp"]
    )

    return results


def calc_induced_dipole(mol: Dimer) -> pd.DataFrame:
    """
    Calculate interaction-induced dipole moment along X, Y and Z axes.

    Args:
        mol (Dimer): A dimer system.

    Returns:
        pd.DataFrame: DataFrame with results.
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
def calc_property(mol: Dimer, prop: str | np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Calculation of interaction-induced property.

    Args:
        mol (Dimer): A dimer system.
        prop (str | np.ndarray): Property to calculate. Either name of already implemented
            property, e.g. dipole moment (prop="dipole") or an array with matrix representation,
            in AO basis, of the operator corresponding to a property to calculate.

    Returns:
        pd.DataFrame: DataFrame with results.
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
