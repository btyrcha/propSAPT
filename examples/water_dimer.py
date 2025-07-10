import psi4
from prop_sapt import Dimer, calc_property, calc_density_matrix
from prop_sapt.utils import prepare_path


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

CUBES_DIR = prepare_path("water-dimer-cubes/")

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
    delta_dm_pol = delta_dm_A["pol"] + delta_dm_B["pol"]
    delta_dm_exch = delta_dm_A["exch"] + delta_dm_B["exch"]
    delta_dm_ind = delta_dm_A["ind"] + delta_dm_B["ind"]
    delta_dm_disp = delta_dm_A["disp"] + delta_dm_B["disp"]

    ### Store densities to .cube files
    densities_to_save = [2 * rho for rho in delta_dm_A.values()]
    densities_to_save += [2 * rho for rho in delta_dm_B.values()]
    densities_to_save += [
        2 * rho
        for rho in [delta_dm, delta_dm_pol, delta_dm_exch, delta_dm_ind, delta_dm_disp]
    ]

    cube_filenames = [CUBES_DIR + f"delta_dm_{key}_A.cube" for key in delta_dm_A.keys()]
    cube_filenames += [
        CUBES_DIR + f"delta_dm_{key}_B.cube" for key in delta_dm_B.keys()
    ]
    cube_filenames += [
        CUBES_DIR + f"delta_dm{key}.cube"
        for key in ["", "_pol", "_exch", "_ind", "_disp"]
    ]

    dimer.save_cube(
        densities_to_save,
        ["density"] * len(densities_to_save),
        cube_filenames,
    )

    ### End calculations
    psi4.core.clean()
