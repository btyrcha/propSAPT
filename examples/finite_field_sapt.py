import pandas as pd
import psi4
from prop_sapt import Dimer, calc_property, finite_field_sapt

# specify geometry in Psi4 format
GEO = """
symmetry c1
no_com
no_reorient
units bohr
0 1
He  -2.500000000   0.000000000   0.000000000
--
0 1
H    1.775500000   0.000000000   0.000000000
H    3.224500000   0.000000000   0.000000000
"""

# specify memory and threads
MEMORY = "2 GB"
THREADS = 2

# specify basis sets
BASIS = "aug-cc-pvtz"
DF_BASIS = "aug-cc-pvqz"

# specify field strength for finite field SAPT
FIELD_STRENGTH = 0.0001

# specify options
OPTIONS = {
    "basis": BASIS,
    # DF_BASIS_SCF is left to Psi4's default: Psi4 has no aug-cc-pVQZ-JKFIT (or
    # aug-cc-pVTZ-JKFIT) set for He, and the default falls back to def2-universal-jkfit.
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "direct",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    # Integral thresholds for higher precision
    "ints_tolerance": 1e-14,
    "screening": "schwarz",
    "cholesky_tolerance": 1e-10,
}

# specify output filename
OUTPUT_FILE_PATH = "output.dat"


def print_dipole_comparison(ff_sapt_data: pd.DataFrame, prop_sapt_data: pd.DataFrame):
    """
    Print a formatted comparison of finite field SAPT and propSAPT dipole moment results.

    Args:
        ff_sapt_data (pd.DataFrame): Finite field SAPT results
        prop_sapt_data (pd.DataFrame): propSAPT results
    """

    # Select relevant columns for comparison
    ff_columns = [
        "ELST1",
        "EXCH1",
        "IND2",
        "IND2,R",
        "EXCH-IND2,R(S^2)",
        "EXCH-IND2,R",
        "DISP2",
        "EXCH-DISP2(S^2)",
        "EXCH-DISP2",
        "TOTAL(S^2)",
        "TOTAL",
    ]
    prop_columns = [
        "x1_pol,r",
        "x1_exch,r",
        "x2_ind",
        "x2_ind,r",
        "x2_exch-ind,r_S2",
        "x2_exch-ind,r",
        "x2_disp",
        "x2_exch-disp_S2",
        "x2_exch-disp",
        "x_induced_S2",
        "x_induced",
    ]

    table_column_names = [
        "Elest",
        "Exch",
        "Ind",
        "Ind,r",
        "Exch-Ind,r S2",
        "Exch-Ind,r",
        "Disp",
        "Exch-Disp S2",
        "Exch-Disp",
        "Total S2",
        "Total",
    ]

    # Create comparison DataFrames with renamed columns for clarity
    ff_sapt_data = ff_sapt_data[ff_columns].copy()
    ff_sapt_data.columns = table_column_names
    ff_sapt_data = ff_sapt_data * 1e3  # scale to 10^{-3} e*a_0

    prop_data = prop_sapt_data[prop_columns].copy()
    prop_data.columns = table_column_names
    prop_data = prop_data * 1e3  # scale to 10^{-3} e*a_0

    table_width = (max(len(elem) for elem in table_column_names) + 2) * len(
        table_column_names
    ) + 10

    # Print formatted comparison
    print("\n" + "=" * table_width)
    print("INTERACTION-INDUCED DIPOLE MOMENT COMPARISON")
    print("=" * table_width)

    print("\nFinite Field SAPT Results (10^{-3} e*a_0):")
    print("-" * table_width)
    print(ff_sapt_data.to_string(float_format=lambda x: f"{x:> 14.6f}"))

    print("\n\npropSAPT Results (10^{-3} e*a_0):")
    print("-" * table_width)
    print(prop_data.to_string(float_format=lambda x: f"{x:> 14.6f}"))

    # Calculate and display differences
    print("\n\nDifference (propSAPT - FF-SAPT) (10^{-3} e*a_0):")
    print("-" * table_width)
    diff_data = prop_data - ff_sapt_data
    print(diff_data.to_string(float_format=lambda x: f"{x:> 14.6f}"))

    # Calculate and display relative error
    print("\n\nRelative error (propSAPT - FF-SAPT) / FF-SAPT (%), \"-\" where the dipole is zero:")
    print("-" * table_width)
    # components zero by symmetry are only numerical noise, ~1e-13 here
    nonzero_ff = ff_sapt_data.where(ff_sapt_data.abs() > 1e-6)
    relerr_data = 100 * diff_data.div(nonzero_ff)
    print(relerr_data.to_string(float_format=lambda x: f"{x:> 14.6f}", na_rep="-"))

    print("=" * table_width)


if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    ### Calculate interaction-induced dipole moment with finite field SAPT
    data_ff_sapt = finite_field_sapt(
        geometry=GEO,
        prop="dipole",
        reference="RHF",
        field_strength=FIELD_STRENGTH,
        formula="four_point",
        sapt_kwargs={"response": "both"},
    )

    ### Calculate interaction-induced dipole moment with propSAPT
    dimer = Dimer(GEO)
    data_prop_sapt = calc_property(dimer, "dipole")

    ### Compare and print results
    print_dipole_comparison(data_ff_sapt, data_prop_sapt)
