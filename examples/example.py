"""
Minimal example of using propSAPT to calculate interaction-induced properties
and densities for a Ne-H2 dimer.
"""

import psi4
from prop_sapt import Dimer, calc_property, calc_densities
from prop_sapt.utils import CalcTimer


# specify geometry in Psi4 format
GEO = """
symmetry c1
no_com
no_reorient
units bohr
0 1
Ne  -2.500000000   0.000000000   0.000000000
--
0 1
H    1.775500000   0.000000000   0.000000000
H    3.224500000   0.000000000   0.000000000
"""

# specify memory and threads
MEMORY = "4 GB"
THREADS = 4

# specify basis sets
BASIS = "aug-cc-pvdz"
DF_BASIS = "aug-cc-pvdz"

# specify options
OPTIONS = {
    # "option": "value",
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "df",
    "save_jk": True,  # necessary option
}

# specify output and results filenames
OUTPUT_FILE_PATH = "output.dat"
RESULTS_FILE_PATH = "results.csv"

if __name__ == "__main__":

    with CalcTimer("Example propSAPT calculations"):

        ### Psi4 options
        psi4.set_memory(MEMORY)
        psi4.set_num_threads(THREADS)
        psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
        psi4.set_options(OPTIONS)

        ### Initalise prop_sapt.Dimer object
        dimer = Dimer(GEO)

        ### Calculate interaction-induced dipole moment
        data = calc_property(dimer, "dipole", results=RESULTS_FILE_PATH)

        ### Calculate interaction-induced denisty matrix
        delta_dm = calc_densities(dimer, save_cubes=True)

        ### NOTE: You can use Psi4 to perform other calculations
        # dimer_psi4 = dimer.get_psi4_molecule()
        # psi4.energy("mp2", molecule=dimer_psi4, bsse_type="cp")

        ### End calculations
        psi4.core.clean()
