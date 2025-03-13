import pytest

import psi4

from interaction_induced import calc_property
from interaction_induced import Dimer


def test_calc_dipole():

    threshold = 5.0e-4

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
    property_type = "dipole"
    expected_results = {
        "x1_pol,r": -3.694805e-04,
        "x1_exch,r": -1.581842e-02,
        "x2_ind,r": 1.042533e-03,
        "x2_disp": 2.195424e-03,
        # "x2_exch-ind,r_S2": -2.096264e-03,
        # "x2_exch-disp_S2": -3.902404e-04,
    }

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
    results = calc_property(dimer, property_type)
    results = results.loc["X"]

    for key, value in expected_results.items():
        assert value == pytest.approx(results[key], abs=threshold)
