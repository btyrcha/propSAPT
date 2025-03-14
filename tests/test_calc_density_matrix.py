import pytest

import numpy as np

import psi4

from interaction_induced import Dimer
from interaction_induced import calc_density_matirx


def test_induction_from_density_matrix():

    threshold = 1.0e-9
    excpected_result = 0.0010501236802531397

    geo = """
symmetry c1
no_com
no_reorient
units bohr
0 1
He  -2.500000000   0.000000000   0.000000000
--
0 1
H    1.775500000   0.000000000   0.000000000
H    3.224500000   0.000000000   0.000000000
"""

    psi4.set_memory("2 GB")
    psi4.core.set_output_file("output.dat", False)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DF_BASIS_SAPT": "aug-cc-pvqz-ri",
            "scf_type": "direct",
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            "save_jk": True,  # necessary option
        }
    )
    dimer = Dimer(geo)
    rho_A = calc_density_matirx(dimer, "A", orbital_basis="AO")
    rho_B = calc_density_matirx(dimer, "B", orbital_basis="AO")

    rho_ind = rho_A["ind"] + rho_B["ind"]

    d_vec = dimer.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]
    d_X = d_vec[0]

    result = 2 * np.trace(rho_ind @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)
