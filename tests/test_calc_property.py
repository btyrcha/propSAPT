import pytest

from prepare_dimer import prepare_dimer
from prop_sapt import calc_property


def test_calc_dipole(prepare_dimer):

    threshold = 5.0e-4

    property_type = "dipole"
    expected_results = {
        "x1_pol,r": -3.694805e-04,
        "x1_exch,r": -1.581842e-02,
        "x2_ind,r": 1.042533e-03,
        "x2_disp": 2.195424e-03,
        "x2_exch-ind,r_S2": -2.096264e-03,
        "x2_exch-disp_S2": -3.902404e-04,
    }

    results = calc_property(prepare_dimer, property_type)
    results = results.loc["X"]

    for key, value in expected_results.items():
        assert value == pytest.approx(results[key], abs=threshold)
