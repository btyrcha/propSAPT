"""
Calculations of SAPT dipole moment contributions.
"""

import os
from time import time
import psi4
import numpy as np
import pandas as pd
import opt_einsum as oe
import tracemalloc
from helper_SAPT_DF import helper_SAPT
from sinfinity import sinfinity


def trace_memory_peak(func):
    """
    Trace peak memory usage of 'func'.
    """

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        func(*args, **kwargs)
        _, peak_size = tracemalloc.get_traced_memory()
        psi4.core.print_out(f"\nPeak meamory usage: {peak_size*1e-6:.2f} MB\n")
        tracemalloc.reset_peak()
        tracemalloc.stop()

    return wrapper


def get_dipole_moment(mol_dimer, sapt: helper_SAPT, sinf: sinfinity) -> pd.DataFrame:
    """
    Calculates dipole moment contributions
    calculated using SAPT.

    Returns pandas.DataFrame with values
    of separate contributions and sumed up
    total dipole moment.
    """
    ### Start calculations time
    psi4.core.print_out("\nStarting dipole moment calculations...\n")
    start_time = time()

    ### Dipole moment matrix elements
    d_vec = sapt.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]

    d_vec_A_aa = [sapt.transform_ao_to_mo(elem, "aa") for elem in d_vec]
    d_vec_A_ar = [sapt.transform_ao_to_mo(elem, "ar") for elem in d_vec]
    d_vec_A_rr = [sapt.transform_ao_to_mo(elem, "rr") for elem in d_vec]

    d_vec_B_bb = [sapt.transform_ao_to_mo(elem, "bb") for elem in d_vec]
    d_vec_B_bs = [sapt.transform_ao_to_mo(elem, "bs") for elem in d_vec]
    d_vec_B_ss = [sapt.transform_ao_to_mo(elem, "ss") for elem in d_vec]

    del d_vec

    ### Response amplitudes
    cphf_ra = sapt.cpscf("A")
    cphf_sb = sapt.cpscf("B")

    ### Dipole moment contributions
    start_time = time()

    ### Nuclear dipole moment
    nuc_dipole = mol_dimer.nuclear_dipole()
    nuc_dipole = [nuc_dipole[0], nuc_dipole[1], nuc_dipole[2]]

    # prepare DataFrame
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

    x = {0: "X", 1: "Y", 2: "Z"}
    for i in range(3):
        # variables for each axis
        d_A_aa = d_vec_A_aa[i]
        d_A_ar = d_vec_A_ar[i]
        d_A_rr = d_vec_A_rr[i]

        d_B_bb = d_vec_B_bb[i]
        d_B_bs = d_vec_B_bs[i]
        d_B_ss = d_vec_B_ss[i]

        # relaxed amplitudes with dipole moment
        xt_A_ra = sapt.cpscf("A", perturbation=d_A_ar)
        xt_A_ar = xt_A_ra.T
        xt_B_sb = sapt.cpscf("B", perturbation=d_B_bs)
        xt_B_bs = xt_B_sb.T

        results["d0_A"][x[i]] = 2 * oe.contract("aa", d_A_aa)
        results["d0_B"][x[i]] = 2 * oe.contract("bb", d_B_bb)

        results["d1_ind,r"][x[i]] = 4 * oe.contract(
            "ra,ar", xt_A_ra, sinf.omegaB_ar
        ) + 4 * oe.contract("sb,bs", xt_B_sb, sinf.omegaA_bs)

        results["d1_exch_ind,r"][x[i]] = (
            ### A part
            # <V P XA>
            -2
            * oe.contract("sc,db,bs,cd", sinf.G_sr, sinf.H_ab, sinf.omegaA_bs, xt_A_ra)
            + 2
            * oe.contract("ca,rd,ar,dc", sinf.B_aa, sinf.C_rr, sinf.omegaB_ar, xt_A_ra)
            - 4
            * oe.contract(
                "ca,rd,sb,dc,Qar,Qbs",
                sinf.B_aa,
                sinf.C_rr,
                sinf.F_sb,
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
            )
            - 2
            * oe.contract(
                "rb,ca,sd,dc,Qar,Qbs",
                sinf.I_rb,
                sinf.B_aa,
                sinf.G_sr,
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
            )
            + 2
            * oe.contract(
                "rc,db,sa,cd,Qar,Qbs",
                sinf.C_rr,
                sinf.H_ab,
                sinf.J_sa,
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
            )
            + 4
            * oe.contract(
                "ra,sc,db,cd,Qar,Qbs",
                sinf.E_ra,
                sinf.G_sr,
                sinf.H_ab,
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
            )
            # <XA|V P>
            - 4 * oe.contract("ra,Qar,Qbs,sb", xt_A_ra, sinf.Qar, sinf.Qbs, sinf.F_sb)
            + 2 * oe.contract("rA,aA,ra", xt_A_ra, sinf.omegaB_aa, sinf.E_ra)
            - 2 * oe.contract("Ra,Rr,ra", xt_A_ra, sinf.omegaB_rr, sinf.E_ra)
            + 2
            * oe.contract(
                "rA,QaA,Qbs,rb,sa", xt_A_ra, sinf.Qaa, sinf.Qbs, sinf.I_rb, sinf.J_sa
            )
            - 2
            * oe.contract(
                "Ra,QRr,Qbs,rb,sa", xt_A_ra, sinf.Qrr, sinf.Qbs, sinf.I_rb, sinf.J_sa
            )
            - 2
            * oe.contract("ra,bs,rb,sa", xt_A_ra, sinf.omegaA_bs, sinf.I_rb, sinf.J_sa)
            - 2
            * oe.contract("ra,AR,Ra,rA", xt_A_ra, sinf.omegaB_ar, sinf.E_ra, sinf.E_ra)
            - 2
            * oe.contract(
                "ra,QAR,Qbs,Ra,rb,sA",
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
                sinf.E_ra,
                sinf.I_rb,
                sinf.J_sa,
            )
            - 2
            * oe.contract(
                "ra,QAR,Qbs,rA,Rb,sa",
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
                sinf.E_ra,
                sinf.I_rb,
                sinf.J_sa,
            )
            + 4
            * oe.contract(
                "ra,QAR,Qbs,RA,rb,sa",
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
                sinf.E_ra,
                sinf.I_rb,
                sinf.J_sa,
            )
            + 4
            * oe.contract(
                "ra,QAR,Qbs,Ra,rA,sb",
                xt_A_ra,
                sinf.Qar,
                sinf.Qbs,
                sinf.E_ra,
                sinf.E_ra,
                sinf.F_sb,
            )
            ### B part
            # <V P XB>
            - 2
            * oe.contract("rc,da,ar,cd", sinf.G_rs, sinf.H_ba, sinf.omegaB_ar, xt_B_sb)
            + 2
            * oe.contract("cb,sd,bs,dc", sinf.A_bb, sinf.D_ss, sinf.omegaA_bs, xt_B_sb)
            - 4
            * oe.contract(
                "ra,cb,sd,dc,Qar,Qbs",
                sinf.E_ra,
                sinf.A_bb,
                sinf.D_ss,
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
            )
            - 2
            * oe.contract(
                "cb,rd,sa,dc,Qar,Qbs",
                sinf.A_bb,
                sinf.G_rs,
                sinf.J_sa,
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
            )
            + 2
            * oe.contract(
                "rb,sc,da,cd,Qar,Qbs",
                sinf.I_rb,
                sinf.D_ss,
                sinf.H_ba,
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
            )
            + 4
            * oe.contract(
                "sb,rc,da,cd,Qar,Qbs",
                sinf.F_sb,
                sinf.G_rs,
                sinf.H_ba,
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
            )
            # <XB|V P>
            - 4 * oe.contract("sb,Qar,Qbs,ra", xt_B_sb, sinf.Qar, sinf.Qbs, sinf.E_ra)
            + 2 * oe.contract("sB,bB,sb", xt_B_sb, sinf.omegaA_bb, sinf.F_sb)
            - 2 * oe.contract("Sb,Ss,sb", xt_B_sb, sinf.omegaA_ss, sinf.F_sb)
            + 2
            * oe.contract(
                "sB,Qar,QbB,sa,rb", xt_B_sb, sinf.Qar, sinf.Qbb, sinf.J_sa, sinf.I_rb
            )
            - 2
            * oe.contract(
                "Sb,Qar,QSs,sa,rb", xt_B_sb, sinf.Qar, sinf.Qss, sinf.J_sa, sinf.I_rb
            )
            - 2
            * oe.contract("sb,ar,sa,rb", xt_B_sb, sinf.omegaB_ar, sinf.J_sa, sinf.I_rb)
            - 2
            * oe.contract("sb,BS,Sb,sB", xt_B_sb, sinf.omegaA_bs, sinf.F_sb, sinf.F_sb)
            - 2
            * oe.contract(
                "sb,Qar,QBS,Sb,sa,rB",
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
                sinf.F_sb,
                sinf.J_sa,
                sinf.I_rb,
            )
            - 2
            * oe.contract(
                "sb,Qar,QBS,sB,Sa,rb",
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
                sinf.F_sb,
                sinf.J_sa,
                sinf.I_rb,
            )
            + 4
            * oe.contract(
                "sb,Qar,QBS,SB,sa,rb",
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
                sinf.F_sb,
                sinf.J_sa,
                sinf.I_rb,
            )
            + 4
            * oe.contract(
                "sb,Qar,QBS,Sb,sB,ra",
                xt_B_sb,
                sinf.Qar,
                sinf.Qbs,
                sinf.F_sb,
                sinf.F_sb,
                sinf.E_ra,
            )
            # remove the polarizational terms from
            # S^inf expressions for <V P XA> and <V P XB> parts
        ) - 0.5 * results["d1_ind,r"][x[i]]

        results["d2_ind,r"][x[i]] = (
            # 2 <X|V T>
            -4 * oe.contract("ra,ac,cr", cphf_ra, sinf.omegaB_aa, xt_A_ar)
            + 4 * oe.contract("ra,cr,ac", cphf_ra, sinf.omegaB_rr, xt_A_ar)
            + 8 * oe.contract("sb,Qar,Qbs,ar", cphf_sb, sinf.Qar, sinf.Qbs, xt_A_ar)
            - 4 * oe.contract("sb,bc,cs", cphf_sb, sinf.omegaA_bb, xt_B_bs)
            + 4 * oe.contract("sb,cs,bc", cphf_sb, sinf.omegaA_ss, xt_B_bs)
            + 8 * oe.contract("ra,Qar,Qbs,bs", cphf_ra, sinf.Qar, sinf.Qbs, xt_B_bs)
            # <T|d T>
            - 2 * oe.contract("ac,rc,ra", d_A_aa, cphf_ra, cphf_ra)
            + 2 * oe.contract("cr,ca,ra", d_A_rr, cphf_ra, cphf_ra)
            - 2 * oe.contract("bc,sc,sb", d_B_bb, cphf_sb, cphf_sb)
            + 2 * oe.contract("cs,cb,sb", d_B_ss, cphf_sb, cphf_sb)
            # Experimental: something like <V R(d) R(V)>
            # if this works it should be aded to d2_ind
            + 8 * oe.contract("ar,Qar,Qbs,sb", xt_A_ar, sinf.Qar, sinf.Qbs, cphf_sb)
            + 8 * oe.contract("bs,Qar,Qbs,ra", xt_B_bs, sinf.Qar, sinf.Qbs, cphf_ra)
        )

        results["d2_disp"][x[i]] = (
            # 2 <X|V T>
            # NOTE some response not coupled
            +8 * oe.contract("rsab,bs,ar", sinf.t_rsab, sinf.omegaA_bs, xt_A_ar)
            - 8
            * oe.contract("rsab,Qac,Qbs,cr", sinf.t_rsab, sinf.Qaa, sinf.Qbs, xt_A_ar)
            + 8
            * oe.contract("rsab,Qcr,Qbs,ac", sinf.t_rsab, sinf.Qrr, sinf.Qbs, xt_A_ar)
            + 8 * oe.contract("rsab,ar,bs", sinf.t_rsab, sinf.omegaB_ar, xt_B_bs)
            - 8
            * oe.contract("rsab,Qar,Qbc,cs", sinf.t_rsab, sinf.Qar, sinf.Qbb, xt_B_bs)
            + 8
            * oe.contract("rsab,Qar,Qcs,bc", sinf.t_rsab, sinf.Qar, sinf.Qss, xt_B_bs)
            # <T|d T>
            # NOTE not coupled no response
            - 4 * oe.contract("ac,rscb,rsab", d_A_aa, sinf.t_rsab, sinf.t_rsab)
            - 4 * oe.contract("bc,rsac,rsab", d_B_bb, sinf.t_rsab, sinf.t_rsab)
            + 4 * oe.contract("cr,csab,rsab", d_A_rr, sinf.t_rsab, sinf.t_rsab)
            + 4 * oe.contract("cs,rcab,rsab", d_B_ss, sinf.t_rsab, sinf.t_rsab)
        )

        results["d_induced"][x[i]] = (
            +results["d1_ind,r"][x[i]]
            + results["d1_exch_ind,r"][x[i]]
            + results["d2_ind,r"][x[i]]
            + results["d2_disp"][x[i]]
        )

    ### Total time
    psi4.core.print_out(
        f"\nDipole moment calculations took {time() - start_time:.2f} seconds.\n"
    )

    return results


def dipole_moment_from_wfn(wfn) -> np.ndarray:
    """
    Calculates the dipole moment based on a given
    wavefunction by calculating the one-electron RDM
    and tracing it with dipole moment operator in AO
    basis.
    """

    mints = psi4.core.MintsHelper(wfn.basisset())
    dipole_vec_ao = mints.ao_dipole()

    Ca = np.asarray(wfn.Ca())
    Ca_o = Ca[:, : wfn.nalpha()]
    Cb = np.asarray(wfn.Cb())
    Cb_o = Cb[:, : wfn.nbeta()]
    rho = Ca_o.dot(Ca_o.T) + Cb_o.dot(Cb_o.T)

    return np.array([np.trace(rho.dot(elem)) for elem in dipole_vec_ao])


def get_sup_mp2_dipole(dimer, options: dict) -> np.ndarray:
    """
    Get dipole moment from supermolecular MP2.
    """

    mp2_start = time()

    monomer_A = dimer.extract_subsets(1, 2)
    monomer_A.set_name("monomerA")
    monomer_B = dimer.extract_subsets(2, 1)
    monomer_B.set_name("monomerB")

    psi4.set_options(options)

    dipole_mp2 = np.array(psi4.properties("MP2", properties=["dipole"], molecule=dimer))
    dipole_mp2_A = np.array(
        psi4.properties("MP2", properties=["dipole"], molecule=monomer_A)
    )
    dipole_mp2_B = np.array(
        psi4.properties("MP2", properties=["dipole"], molecule=monomer_B)
    )

    # Supermolecular MP2 time
    mp2_time = time() - mp2_start
    psi4.core.print_out(f"Supermolecular MP2 took {mp2_time:.2f} seconds.\n")

    return dipole_mp2 - dipole_mp2_A - dipole_mp2_B


@trace_memory_peak
def one_point_calculation(geo_path: str, options: dict, **kwargs) -> pd.DataFrame:
    """
    Single point induced dipole moment calculations
    for dimer geometry.
    """

    sysname = geo_path.split("/")[-1].removesuffix(".xyz")
    mol = "-".join(sysname.split("-")[:2])

    ### Start total time of calculations
    total_time = time()

    ### Psi4 output file
    psi4_output_fname = sysname + "-" + options["basis"] + ".dat"
    psi4_output_path = os.path.join("outputs", mol, psi4_output_fname)

    ### Results output file
    results_fname = sysname + "-" + options["basis"] + ".csv"
    results_path = os.path.join("results", mol, results_fname)

    ### Psi4 options
    psi4.set_memory(kwargs.get("memory", "2 GB"))
    psi4.set_num_threads(kwargs.get("threads", 1))
    psi4.core.set_output_file(psi4_output_path, False)
    psi4.set_options(options)

    ### Reading the molecule geometry
    with open(geo_path, "r", encoding="utf-8") as f:
        geometry = f.read()

    ### Printing informations
    psi4.core.print_out(
        psi4.driver.p4util.message_box(f"Dipole moment calculations\nfor {sysname}")
    )
    psi4.core.print_out("\n\nSystem geometry:\n")
    psi4.core.print_out("-" * 80 + "\n")
    psi4.core.print_out(geometry + "\n")
    psi4.core.print_out("-" * 80 + "\n")
    psi4.core.print_out(f"\nBasis set: {options['basis']}\n")
    psi4.core.print_out(f"Auxiliary basis set: {options['DF_BASIS_SAPT']}\n")

    ### Psi4NumPy helper_SAPT calculations
    dimer = psi4.geometry(geometry)
    sapt_helper = helper_SAPT(
        dimer,
        df_basis=options["DF_BASIS_SAPT"],
        memory=kwargs.get("numpy_memory", 8),
    )
    sinf = sinfinity(sapt_helper, df_basis=options["DF_BASIS_SAPT"])

    ### Number of basis functions
    psi4.core.print_out("\nNumber of orbitals:\n")
    psi4.core.print_out("-" * 27 + "\n")
    psi4.core.print_out("   a    b    r    s\n")
    psi4.core.print_out("-" * 27 + "\n")
    psi4.core.print_out(
        f"{sapt_helper.sizes['a']:4d}"
        + f"{sapt_helper.sizes['b']:4d}"
        + f"{sapt_helper.sizes['r']:4d}"
        + f"{sapt_helper.sizes['s']:4d}"
        + "\n\n"
    )

    ### Dipole moment form SAPT calculations
    data = get_dipole_moment(dimer, sapt_helper, sinf)

    del sapt_helper
    del sinf

    ### Supermolecular HF calculations
    hf_start = time()
    _, wfn_hf = psi4.energy("SCF", molecule=dimer, return_wfn=True)

    dipole_hf = dipole_moment_from_wfn(wfn_hf)
    data["sup_HF"] = dipole_hf - data["d0_A"] - data["d0_B"]  # induced dipole

    # Supermolecular HF time
    hf_time = time() - hf_start
    psi4.core.print_out(f"Supermolecular HF took {hf_time:.2f} seconds.\n")

    ### Supermolecular MP2 calculations
    if kwargs.get("do_mp2", False):

        if kwargs.get("mp2_options", None) is None:
            psi4.core.clean()
            raise ValueError(
                "Aseked form supermolecular MP2 but no options were given!\n"
                + "Specify options dict with the keyword argument 'mp2_options'."
            )

        data["sup_MP2"] = get_sup_mp2_dipole(dimer, kwargs["mp2_options"])

    ### Calculating vector length
    data = data.transpose()
    data["len"] = np.sqrt(data["X"] ** 2 + data["Y"] ** 2 + data["Z"] ** 2)
    data = data.transpose()

    ### Results saving to file
    data.to_csv(results_path)

    ### End of calculations
    total_time = time() - total_time
    psi4.core.print_out(f"\nIn total calculaitons took {total_time:.2f} seconds.\n")
    psi4.core.print_out(f"Calculations for {sysname} ended.\n")
    psi4.core.clean()

    return data
