"""
This example creates cube files from interaction-induced density matrices in MO basis.
Use the example "save_mo_matrices.py" to calculate and save the density matrices
before running this example.
"""

import numpy as np
import psi4

from prop_sapt.cubes import make_cube

WFN_A_FILE_PATH = "matrices_MO/wfn_mon_A.npy"
WFN_B_FILE_PATH = "matrices_MO/wfn_mon_B.npy"

MATRIX_A_FILE_PATH = "matrices_MO/delta_dm_total_A.npy"
MATRIX_B_FILE_PATH = "matrices_MO/delta_dm_total_B.npy"


if __name__ == "__main__":

    wfn_A = psi4.core.Wavefunction.from_file(WFN_A_FILE_PATH)
    wfn_B = psi4.core.Wavefunction.from_file(WFN_B_FILE_PATH)

    C_A = wfn_A.Ca().to_array()
    C_B = wfn_B.Ca().to_array()

    dm_mo_A = np.load(MATRIX_A_FILE_PATH)
    dm_ao_A = C_A.dot(dm_mo_A).dot(C_A.T)
    dm_ao_A *= 2.0  # factor 2 for RHF spin

    cube_A = make_cube(
        wfn_A.molecule(),
        dm_ao_A,
        "density",
        basisset=wfn_A.basisset(),  # necesary in this case
    )
    cube_A.save(MATRIX_A_FILE_PATH.replace(".npy", ".cube"))

    dm_mo_B = np.load(MATRIX_B_FILE_PATH)
    dm_ao_B = C_B.dot(dm_mo_B).dot(C_B.T)
    dm_ao_B *= 2.0  # factor 2 for RHF spin

    cube_B = make_cube(
        wfn_B.molecule(),
        dm_ao_B,
        "density",
        basisset=wfn_B.basisset(),  # necesary in this case
    )
    cube_B.save(MATRIX_B_FILE_PATH.replace(".npy", ".cube"))
