"""
Example script to calculate SAPT energy using the interaction_induced package.
"""

import psi4

from interaction_induced import Dimer, calc_sapt_energy
from interaction_induced.utils import energy_printer

# Define the geometry of the dimer
GEO = """
symmetry c1
no_com
no_reorient
units bohr
0 1
He  0.000000000  0.000000000 -2.800000000
--
0 1
He  0.000000000  0.000000000  2.800000000
"""

# specify memory and threads
MEMORY = "2 GB"
THREADS = 2

# specify basis sets
BASIS = "aug-cc-pvtz"
DF_BASIS = "aug-cc-pvqz"

# specify options
OPTIONS = {
    # "option": "value",
    "reference": "RHF",
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "direct",
    "save_jk": True,  # necessary option
}

# specify output and results filenames
OUTPUT_FILE_PATH = "output.dat"
RESULTS_FILE_PATH = "results.csv"

if __name__ == "__main__":
    # Set Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    # Initialize interaction_induced.Dimer object
    dimer = Dimer(GEO)

    # Calculate SAPT energy
    sapt_results = calc_sapt_energy(dimer, results=RESULTS_FILE_PATH)

    # Print the results
    for term, value in sapt_results.items():
        energy_printer(term, value, output="stdout")

    # Compare with Psi4's SAPT energy calculation
    psi4.set_options(
        {
            "reference": "RHF",
            "basis": BASIS,
            "DF_BASIS_SCF": DF_BASIS + "-jkfit",
            "DF_BASIS_MP2": DF_BASIS + "-ri",
            "DF_BASIS_ELST": DF_BASIS + "-jkfit",
            "DF_BASIS_SAPT": DF_BASIS + "-ri",
            "scf_type": "direct",
            "SAPT_DFT_MP2_DISP_ALG": "fisapt",
            "SAPT_DFT_FUNCTIONAL": "HF",
            "DO_IND_EXCH_SINF": True,
            "DO_DISP_EXCH_SINF": True,
        }
    )
    psi4.energy("sapt(dft)")

    psi4.core.clean()
