"""
This example creates cube files from saved interaction-induced density matrices.
Use the example "save_mo_matrices.py" to calculate and save the density matrices
before running this example.

Saved matrices are always in the AO basis, also when they were calculated in the MO
basis, in one bundle with the geometry, the basis set and the conventions, which
`prop_sapt.load_densities` reads and checks.
"""

import psi4

from prop_sapt import load_densities
from prop_sapt.cubes import make_cube

MATRICES_DIR = "matrices_MO"


if __name__ == "__main__":

    psi4.core.set_output_file("output_cubes.dat", False)

    densities = load_densities(MATRICES_DIR)

    # the dimer with every atom real, in the frame of the propSAPT calculation
    mol = psi4.geometry(densities.metadata["geometry_input"])
    mol.reset_point_group("c1")
    mol.fix_orientation(True)
    mol.fix_com(True)
    mol.update_geometry()

    # the monomer basis sets span the whole dimer (dimer-centered basis set)
    basisset = densities.wavefunction("A").basisset()

    for key in ("total_A", "total_B"):
        cube = make_cube(
            mol,
            densities.total(key),  # total density: twice the stored one-spin matrix
            "density",
            basisset=basisset,  # necesary in this case
        )
        cube.save(f"{MATRICES_DIR}/delta_dm_{key}.cube")
