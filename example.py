import psi4
import interaction_induced as ii


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
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
    "s_tolerance": 1e-12,
    "screening": "density",
    "ints_tolerance": 1e-14,
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

    ### Initalise interaction_induced.Molecule object
    dimer = ii.Molecule(GEO)

    ### Calculate interaction-induced dipole moment
    data = ii.calc_property(dimer, "dipole", results=RESULTS_FILE_PATH)

    ### Calculate interaction-induced denisty matrix
    density_matrix = ii.calc_density(dimer)

    ### End calculations
    psi4.core.clean()
