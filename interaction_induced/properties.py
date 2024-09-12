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

    prop_B_bb = mol.transform_ao_to_mo(property_matix, "aa")
    prop_B_bs = mol.transform_ao_to_mo(property_matix, "ar")
    # prop_B_ss = mol.transform_ao_to_mo(property_matix, "rr")

    # relaxed amplitudes with dipole moment
    xt_A_ra = mol.cpscf("A", perturbation=prop_A_ar)
    # xt_A_ar = xt_A_ra.T
    xt_B_sb = mol.cpscf("B", perturbation=prop_B_bs)
    # xt_B_bs = xt_B_sb.T

    results["x0_A"] = 2 * oe.contract("aa", prop_A_aa)
    results["x0_B"] = 2 * oe.contract("bb", prop_B_bb)

    results["x1_ind,r"] = 4 * oe.contract(
        "ra,ar", xt_A_ra, mol.omegaB_ar
    ) + 4 * oe.contract("sb,bs", xt_B_sb, mol.omegaA_bs)

    results["x1_exch_ind,r"] = (
        ### A part
        # <V P XA>
        -2 * oe.contract("sc,db,bs,cd", mol.G_sr, mol.H_ab, mol.omegaA_bs, xt_A_ra)
        + 2 * oe.contract("ca,rd,ar,dc", mol.B_aa, mol.C_rr, mol.omegaB_ar, xt_A_ra)
        - 4
        * oe.contract(
            "ca,rd,sb,dc,Qar,Qbs",
            mol.B_aa,
            mol.C_rr,
            mol.F_sb,
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "rb,ca,sd,dc,Qar,Qbs",
            mol.I_rb,
            mol.B_aa,
            mol.G_sr,
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rc,db,sa,cd,Qar,Qbs",
            mol.C_rr,
            mol.H_ab,
            mol.J_sa,
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "ra,sc,db,cd,Qar,Qbs",
            mol.E_ra,
            mol.G_sr,
            mol.H_ab,
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
        )
        # <XA|V P>
        - 4 * oe.contract("ra,Qar,Qbs,sb", xt_A_ra, mol.Qar, mol.Qbs, mol.F_sb)
        + 2 * oe.contract("rA,aA,ra", xt_A_ra, mol.omegaB_aa, mol.E_ra)
        - 2 * oe.contract("Ra,Rr,ra", xt_A_ra, mol.omegaB_rr, mol.E_ra)
        + 2
        * oe.contract("rA,QaA,Qbs,rb,sa", xt_A_ra, mol.Qaa, mol.Qbs, mol.I_rb, mol.J_sa)
        - 2
        * oe.contract("Ra,QRr,Qbs,rb,sa", xt_A_ra, mol.Qrr, mol.Qbs, mol.I_rb, mol.J_sa)
        - 2 * oe.contract("ra,bs,rb,sa", xt_A_ra, mol.omegaA_bs, mol.I_rb, mol.J_sa)
        - 2 * oe.contract("ra,AR,Ra,rA", xt_A_ra, mol.omegaB_ar, mol.E_ra, mol.E_ra)
        - 2
        * oe.contract(
            "ra,QAR,Qbs,Ra,rb,sA",
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
            mol.E_ra,
            mol.I_rb,
            mol.J_sa,
        )
        - 2
        * oe.contract(
            "ra,QAR,Qbs,rA,Rb,sa",
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
            mol.E_ra,
            mol.I_rb,
            mol.J_sa,
        )
        + 4
        * oe.contract(
            "ra,QAR,Qbs,RA,rb,sa",
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
            mol.E_ra,
            mol.I_rb,
            mol.J_sa,
        )
        + 4
        * oe.contract(
            "ra,QAR,Qbs,Ra,rA,sb",
            xt_A_ra,
            mol.Qar,
            mol.Qbs,
            mol.E_ra,
            mol.E_ra,
            mol.F_sb,
        )
        ### B part
        # <V P XB>
        - 2 * oe.contract("rc,da,ar,cd", mol.G_rs, mol.H_ba, mol.omegaB_ar, xt_B_sb)
        + 2 * oe.contract("cb,sd,bs,dc", mol.A_bb, mol.D_ss, mol.omegaA_bs, xt_B_sb)
        - 4
        * oe.contract(
            "ra,cb,sd,dc,Qar,Qbs",
            mol.E_ra,
            mol.A_bb,
            mol.D_ss,
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
        )
        - 2
        * oe.contract(
            "cb,rd,sa,dc,Qar,Qbs",
            mol.A_bb,
            mol.G_rs,
            mol.J_sa,
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
        )
        + 2
        * oe.contract(
            "rb,sc,da,cd,Qar,Qbs",
            mol.I_rb,
            mol.D_ss,
            mol.H_ba,
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
        )
        + 4
        * oe.contract(
            "sb,rc,da,cd,Qar,Qbs",
            mol.F_sb,
            mol.G_rs,
            mol.H_ba,
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
        )
        # <XB|V P>
        - 4 * oe.contract("sb,Qar,Qbs,ra", xt_B_sb, mol.Qar, mol.Qbs, mol.E_ra)
        + 2 * oe.contract("sB,bB,sb", xt_B_sb, mol.omegaA_bb, mol.F_sb)
        - 2 * oe.contract("Sb,Ss,sb", xt_B_sb, mol.omegaA_ss, mol.F_sb)
        + 2
        * oe.contract("sB,Qar,QbB,sa,rb", xt_B_sb, mol.Qar, mol.Qbb, mol.J_sa, mol.I_rb)
        - 2
        * oe.contract("Sb,Qar,QSs,sa,rb", xt_B_sb, mol.Qar, mol.Qss, mol.J_sa, mol.I_rb)
        - 2 * oe.contract("sb,ar,sa,rb", xt_B_sb, mol.omegaB_ar, mol.J_sa, mol.I_rb)
        - 2 * oe.contract("sb,BS,Sb,sB", xt_B_sb, mol.omegaA_bs, mol.F_sb, mol.F_sb)
        - 2
        * oe.contract(
            "sb,Qar,QBS,Sb,sa,rB",
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
            mol.F_sb,
            mol.J_sa,
            mol.I_rb,
        )
        - 2
        * oe.contract(
            "sb,Qar,QBS,sB,Sa,rb",
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
            mol.F_sb,
            mol.J_sa,
            mol.I_rb,
        )
        + 4
        * oe.contract(
            "sb,Qar,QBS,SB,sa,rb",
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
            mol.F_sb,
            mol.J_sa,
            mol.I_rb,
        )
        + 4
        * oe.contract(
            "sb,Qar,QBS,Sb,sB,ra",
            xt_B_sb,
            mol.Qar,
            mol.Qbs,
            mol.F_sb,
            mol.F_sb,
            mol.E_ra,
        )
        # remove the polarizational terms from
        # S^inf expressions for <V P XA> and <V P XB> parts
    ) - 0.5 * results["x1_ind,r"]

    results["d_induced"] = (
        +results["x1_ind,r"]
        + results["x1_exch_ind,r"]
        + results["x2_ind,r"]
        + results["x2_disp"]
    )

    return results


@trace_memory_peak
def calc_property(mol: Molecule, prop: str | np.array, **kwargs) -> pd.DataFrame:
    """
    Calculation of interaction-induced property.
    """

    ### Start total time of calculations
    total_time = time()

    ### Results output file
    results_fname = kwargs.get("results", "results.csv")

    if prop == "dimer":
        ### Dipole moment calculations
        # NOTE: probalby move to a new function

        # grab nuclear dipole moment
        nuc_dipole = mol.dimer.nuclear_dipole()
        nuc_dipole = [nuc_dipole[0], nuc_dipole[1], nuc_dipole[2]]

        # grab AO dipole moment matrices
        d_vec = mol.mints.ao_dipole()
        d_vec = [np.asarray(elem) for elem in d_vec]

        # calculate for each axis
        for d_i, i in zip(d_vec, ["X", "Y", "Z"]):
            # TODO: save it properly
            perform_property_contractions(mol, d_i)

        del d_vec

        # TODO: prepare results
        results = pd.DataFrame(
            {
                "nuc": nuc_dipole,
                "d0_A": np.zeros(3),
                "d0_B": np.zeros(3),
                "d1_ind,r": np.zeros(3),
                "d1_exch_ind,r": np.zeros(3),
                "d2_ind,r": np.zeros(3),
                "d2_disp": np.zeros(3),
                "d_induced": np.zeros(3),
            },
            index=["X", "Y", "Z"],
        )

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
