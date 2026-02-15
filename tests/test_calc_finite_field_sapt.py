import pytest

from prepare_dimer import prepare_dimer
from prop_sapt import finite_field_sapt


def test_calc_ff_sapt_dipole(prepare_dimer):

    threshold = 1.0e-8

    field_strength = 0.0001
    property_type = "dipole"
    expected_results = {
        "ELST1": -0.369553 * 1e-3,
        "EXCH1": -15.816954 * 1e-3,
        "IND2,R": 1.042484 * 1e-3,
        "EXCH-IND2,R(S^2)": -2.095958 * 1e-3,
        "EXCH-IND2,R": -2.105035 * 1e-3,
        "DISP2": 2.195426 * 1e-3,
        "EXCH-DISP2(S^2)": -0.390241 * 1e-3,
        "EXCH-DISP2": -0.390747 * 1e-3,
    }

    results = finite_field_sapt(
        prepare_dimer.geometry, property_type, field_strength=field_strength
    )
    results = results.loc["X"]

    for key, value in expected_results.items():
        assert value == pytest.approx(results[key], abs=threshold)
