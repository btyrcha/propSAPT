import os
from interaction_induced.cube_utils import (
    read_cube_file,
    save_cube_file,
    subtract_cubes,
)


if __name__ == "__main__":

    cubes_dir = "water-dimer-cubes"

    delta_rho = read_cube_file(os.path.join(cubes_dir, "delta_dm.cube"))
    delta_rho_pol = read_cube_file(os.path.join(cubes_dir, "delta_dm_pol.cube"))

    # save_cube_file(delta_rho, "saving_test.cube")

    diff = subtract_cubes(delta_rho, delta_rho_pol)

    diff["comment1"] = (
        "This is the result of subtraction of two cube files: `delta_dm.cube` - `delta_dm_pol.cube`."
    )
    diff["comment2"] = (
        "Volumetric data should be in agreement with `delta_dm_exch.cube`."
    )

    save_cube_file(diff, "saving_test.cube")
