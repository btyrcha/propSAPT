"""
This example calculates the interaction-induced densities of a Ne-H2 dimer and writes them
into TREXIO files, to view them in VIAMD (https://github.com/scanberg/viamd), and prints
the leading natural orbitals of the first-order polarisation of each monomer.

It needs the trexio package (pip install trexio). In VIAMD, open e.g.
trexio_ne_h2/delta_dm_total.h5 and show the Electron Density with spin Difference: the
alpha - beta density of a file is its density change.
"""

import numpy as np
import psi4
from prop_sapt import Dimer, calc_densities, density_natural_orbitals
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
}

# specify output filename and the directory of the TREXIO files
OUTPUT_FILE_PATH = "output.dat"
TREXIO_DIR = "trexio_ne_h2"

if __name__ == "__main__":

    with CalcTimer("Example propSAPT TREXIO export"):

        ### Psi4 options
        psi4.set_memory(MEMORY)
        psi4.set_num_threads(THREADS)
        psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
        psi4.set_options(OPTIONS)

        ### Initialise prop_sapt.Dimer object
        dimer = Dimer(GEO)

        ### Calculate the density matrices and write a TREXIO file of each
        densities = calc_densities(dimer, save_trexio=True, trexio_dir=TREXIO_DIR)

        ### The natural orbitals of the polarisation: a +/- pair per occupied orbital
        orbitals = density_natural_orbitals(
            densities, dimer, keys=["pol_A", "pol_B"], cutoff=1e-8
        )
        for key, natural in orbitals.items():
            occupations = np.array2string(natural.occupations, precision=6)
            print(f"{key}: {len(natural)} natural orbitals, occupations {occupations}")

        ### End calculations
        psi4.core.clean()
