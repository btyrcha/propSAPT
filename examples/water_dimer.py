import psi4
import interaction_induced as ii
from interaction_induced.utils import prepare_path


# specify geometry in Psi4 format
GEO = """
symmetry c1
no_com
no_reorient
--
0 1
O   -1.41606479    0.00000000    0.00000000
H   -0.52004811    0.00000000   -0.08624091
H   -1.81884655    0.00000000   -0.89989100
--
0 1
O    1.41606479    0.00000000    0.00000000
H    1.71179607   -0.76501221    0.50083629
H    1.71179607    0.76501221    0.50083629
"""

# specify memory and threads
MEMORY = "5 GB"
THREADS = 4

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

    ### Initalise interaction_induced.Molecule object
    dimer = ii.Molecule(GEO)

    ### Calculate interaction-induced dipole moment
    data = ii.calc_property(dimer, "dipole", results=RESULTS_FILE_PATH)

    ### Calculate interaction-induced denisty matrix
    delta_dm_A = ii.calc_density_matirx(dimer, "A")
    delta_dm_B = ii.calc_density_matirx(dimer, "B")

    delta_dm = delta_dm_A["total"] + delta_dm_B["total"]
    delta_dm_pol = delta_dm_A["pol"] + delta_dm_B["pol"]
    delta_dm_exch = delta_dm_A["exch"] + delta_dm_B["exch"]

    ### Store densities to .cube files
    dimer.save_cube(
        2 * delta_dm_A["pol"],
        filename=prepare_path("water-dimer-cubes/delta_dm_pol_A.cube"),
    )
    dimer.save_cube(
        2 * delta_dm_A["exch"],
        filename="water-dimer-cubes/delta_dm_exch_A.cube",
    )
    dimer.save_cube(
        2 * delta_dm_B["pol"],
        filename="water-dimer-cubes/delta_dm_pol_B.cube",
    )
    dimer.save_cube(
        2 * delta_dm_B["exch"],
        filename="water-dimer-cubes/delta_dm_exch_B.cube",
    )
    dimer.save_cube(
        2 * delta_dm_A["total"],
        filename="water-dimer-cubes/delta_dm_A.cube",
    )
    dimer.save_cube(
        2 * delta_dm_B["total"],
        filename="water-dimer-cubes/delta_dm_B.cube",
    )
    dimer.save_cube(
        2 * delta_dm,
        filename="water-dimer-cubes/delta_dm.cube",
    )
    dimer.save_cube(
        2 * delta_dm_pol,
        filename="water-dimer-cubes/delta_dm_pol.cube",
    )
    dimer.save_cube(
        2 * delta_dm_exch,
        filename="water-dimer-cubes/delta_dm_exch.cube",
    )

    ### End calculations
    psi4.core.clean()
