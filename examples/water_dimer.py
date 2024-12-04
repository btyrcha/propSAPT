import psi4
from interaction_induced import Dimer, calc_property, calc_density_matirx
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
    dimer = Dimer(GEO)

    ### Calculate interaction-induced dipole moment
    data = calc_property(dimer, "dipole", results=RESULTS_FILE_PATH)

    ### Calculate interaction-induced denisty matrix
    delta_dm_A = calc_density_matirx(dimer, "A")
    delta_dm_B = calc_density_matirx(dimer, "B")

    delta_dm = delta_dm_A["total"] + delta_dm_B["total"]
    delta_dm_pol = delta_dm_A["pol"] + delta_dm_B["pol"]
    delta_dm_exch = delta_dm_A["exch"] + delta_dm_B["exch"]

    ### Store densities to .cube files
    dimer.save_cube(
        [
            2 * delta_dm_A["pol"],
            2 * delta_dm_A["exch"],
            2 * delta_dm_B["pol"],
            2 * delta_dm_B["exch"],
            2 * delta_dm_A["total"],
            2 * delta_dm_B["total"],
            2 * delta_dm,
            2 * delta_dm_pol,
            2 * delta_dm_exch,
        ],
        ["density"] * 9,
        [
            prepare_path("water-dimer-cubes/delta_dm_pol_A.cube"),
            "water-dimer-cubes/delta_dm_exch_A.cube",
            "water-dimer-cubes/delta_dm_pol_B.cube",
            "water-dimer-cubes/delta_dm_exch_B.cube",
            "water-dimer-cubes/delta_dm_A.cube",
            "water-dimer-cubes/delta_dm_B.cube",
            "water-dimer-cubes/delta_dm.cube",
            "water-dimer-cubes/delta_dm_pol.cube",
            "water-dimer-cubes/delta_dm_exch.cube",
        ],
    )

    ### End calculations
    psi4.core.clean()
