"""
Cube arithmetic on the density cubes that `water_dimer.py` writes to `water-dimer-cubes/`.
Run `water_dimer.py` first.
"""

import os

from prop_sapt.cubes import read_cube_file

if __name__ == "__main__":

    cubes_dir = "water-dimer-cubes"

    delta_rho = read_cube_file(os.path.join(cubes_dir, "delta_dm_total.cube"))
    delta_rho_pol = read_cube_file(os.path.join(cubes_dir, "delta_dm_pol.cube"))

    # the grids must match; same as subtract_cubes(delta_rho, delta_rho_pol)
    diff = delta_rho - delta_rho_pol

    # the total density is pol + exch + Ind + Disp, so this is everything but polarization
    diff.comment1 = "delta_dm_total.cube - delta_dm_pol.cube"
    diff.comment2 = "exchange, induction and dispersion parts: exch + Ind + Disp"

    print(f"Charge moved by everything but polarization: {diff.moved_charge():.6f} e")

    diff.save("saving_test.cube")
