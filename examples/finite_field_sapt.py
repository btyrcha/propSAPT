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

# specify options
OPTIONS = {
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "direct",
    "save_jk": True,  # necessary option
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    # Integral thresholds for higher precision
    "ints_tolerance": 1e-14,
    "screening": "schwarz",
    "cholesky_tolerance": 1e-10,
}

# specify output filename
OUTPUT_FILE_PATH = "output.dat"

if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    ### Calculate interaction-induced dipole moment with finite field SAPT
    data_ff_sapt = finite_field_sapt(
        geometry=GEO, prop="dipole", reference="RHF", field_strength=0.001
    )

    ### Calculate interaction-induced dipole moment with propSAPT
    dimer = Dimer(GEO)
    data_prop_sapt = calc_property(dimer, "dipole")

    ### Compare results
    pd.set_option("display.precision", 6)
    pd.set_option("display.float_format", "{:.6f}".format)

    print("Finite Field SAPT Dipole Moment:")
    print(
        data_ff_sapt[
            [
                'ELST1',
                'EXCH1',
                'IND2,R',
                'EXCH-IND2,R',
                'DISP2',
                'EXCH-DISP2',
            ]
        ]
    )

    print("\npropSAPT Dipole Moment:")
    print(
        data_prop_sapt[
            [
                'x1_pol,r',
                'x1_exch,r',
                'x2_ind,r',
                'x2_exch-ind,r_S2',
                'x2_disp',
                'x2_exch-disp_S2',
            ]
        ]
    )
