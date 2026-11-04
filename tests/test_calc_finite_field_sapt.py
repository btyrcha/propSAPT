import pytest

from prop_sapt import finite_field_sapt
from prop_sapt.finite_field import calculate_ff_sapt_dipole_along_axis

from .he_h2 import HE_H2

# every field builds its own dimer, from the global Psi4 options
pytestmark = pytest.mark.usefixtures("he_h2_settings")

FIELD_STRENGTH = 0.0001

# Interaction-induced dipole along X for the He-H2 dimer, in a.u.
# Validated against Psi4's native SAPT implementation.
EXPECTED_X = {
    "ELST1": -0.369553 * 1e-3,
    "EXCH1": -15.816954 * 1e-3,
    "IND2,R": 1.042484 * 1e-3,
    "EXCH-IND2,R(S^2)": -2.095958 * 1e-3,
    "EXCH-IND2,R": -2.105035 * 1e-3,
    "DISP2": 2.195426 * 1e-3,
    "EXCH-DISP2(S^2)": -0.390241 * 1e-3,
    "EXCH-DISP2": -0.390747 * 1e-3,
}

# Same quantity from the four-point stencil, converged in the step size: the
# values below shift by less than 1e-11 between h = 1e-4 and h = 2e-4.
EXPECTED_FOUR_POINT_X = {
    "ELST1": -0.369558 * 1e-3,
    "EXCH1": -15.816933 * 1e-3,
    "IND2,R": 1.042481 * 1e-3,
    "EXCH-IND2,R(S^2)": -2.095952 * 1e-3,
    "EXCH-IND2,R": -2.105030 * 1e-3,
    "DISP2": 2.195424 * 1e-3,
    "EXCH-DISP2(S^2)": -0.390240 * 1e-3,
    "EXCH-DISP2": -0.390746 * 1e-3,
}


def test_calc_ff_sapt_dipole():

    threshold = 1.0e-8

    property_type = "dipole"

    results = finite_field_sapt(HE_H2, property_type, field_strength=FIELD_STRENGTH)
    results = results.loc["X"]

    for key, value in EXPECTED_X.items():
        assert value == pytest.approx(results[key], abs=threshold)


def test_calc_ff_sapt_dipole_four_point():
    """Four-point finite field SAPT reproduces the interaction-induced dipole.

    Only the X axis is computed: it is the axis the reference values cover, and
    four-point costs four SAPT runs per axis instead of two.

    Two thresholds are used on purpose. Against EXPECTED_FOUR_POINT_X the usual
    1e-8 applies. Against EXPECTED_X — obtained with the two-point stencil — the
    tolerance has to absorb that stencil's own O(h^2) truncation error, which at
    h = 1e-4 reaches 2.1e-8 on EXCH1; the four-point values are the accurate ones.
    """

    results = calculate_ff_sapt_dipole_along_axis(
        HE_H2,
        axis="X",
        field_strength=FIELD_STRENGTH,
        formula="four_point",
    )

    for key, value in EXPECTED_FOUR_POINT_X.items():
        assert value == pytest.approx(results[key], abs=1.0e-8)

    for key, value in EXPECTED_X.items():
        assert value == pytest.approx(results[key], abs=1.0e-7)


def test_ff_sapt_four_point_is_fourth_order():
    """The four-point result must be converged with respect to the step size.

    Halving h changes an O(h^4) estimate by ~16x less than an O(h^2) one. The
    two-point values move by 4.6e-8 between h and 2h; four-point moves by less
    than 1e-10, which a mis-weighted stencil would not reproduce.
    """

    results = {
        h: calculate_ff_sapt_dipole_along_axis(
            HE_H2,
            axis="X",
            field_strength=h,
            formula="four_point",
        )
        for h in (FIELD_STRENGTH, 2 * FIELD_STRENGTH)
    }

    deviation = (results[FIELD_STRENGTH] - results[2 * FIELD_STRENGTH]).abs()

    assert deviation.max() < 1.0e-10, f"four-point not converged in h:\n{deviation}"
