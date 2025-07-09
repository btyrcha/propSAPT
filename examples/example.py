import psi4
from prop_sapt import Dimer, calc_property, calc_density_matirx


# specify geometry in Psi4 format
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
BASIS = "aug-cc-pvdz"
DF_BASIS = "aug-cc-pvtz"

# specify options
OPTIONS = {
    # "option": "value",
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "df",
    "save_jk": True,  # necessary option
}

# specify output and resultS filenames
OUTPUT_FILE_PATH = "output.dat"
RESULTS_FILE_PATH = "results.csv"

if __name__ == "__main__":

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
    delta_dm_A = calc_density_matrix(dimer, "A")
    delta_dm_B = calc_density_matrix(dimer, "B")

    delta_dm = delta_dm_A["total"] + delta_dm_B["total"]

    ### Store densities to .cube files
    dimer.save_cube(2 * delta_dm_A["total"], filename="delta_dm_A.cube")
    dimer.save_cube(2 * delta_dm_B["total"], filename="delta_dm_B.cube")
    dimer.save_cube(2 * delta_dm, filename="delta_dm.cube")

    ### Use Psi4 to perform other calculations
    dimer_psi4 = dimer.get_psi4_molecule()
    psi4.energy("mp2", molecule=dimer_psi4)

    ### End calculations
    psi4.core.clean()
