import warnings

import pytest

import numpy as np
import psi4

from prop_sapt import reset_propsapt_options, set_propsapt_options
from prop_sapt.checks import (
    InvariantWarning,
    check_charge_closure,
    check_partition_of_unity,
    check_symmetric,
)
from prop_sapt.partition import becke_fragment_weights, molecular_grid


@pytest.fixture
def symmetric():
    """A symmetric 4x4 matrix."""

    matrix = np.random.default_rng(0).normal(size=(4, 4))
    return matrix + matrix.T


@pytest.fixture
def options():
    """Restores the default propSAPT options after the test."""

    yield
    reset_propsapt_options()


def _skewed(matrix, deviation):
    """`matrix` with element [0, 1] raised by `deviation`, so max|D - D^T| = deviation."""

    matrix = matrix.copy()
    matrix[0, 1] += deviation
    return matrix


def test_symmetric_matrices_pass(symmetric):
    """Symmetric matrices, single or by name, give no warning."""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_symmetric({"pol": symmetric, "exch": 2 * symmetric}) == {}
        assert check_symmetric(symmetric) == {}


def test_warns_once_naming_every_offender(symmetric):
    """One warning lists every matrix above the tolerance, with its deviation."""

    matrices = {
        "pol": symmetric,
        "exch-ind_S2": _skewed(symmetric, 2.5e-5),
        "exch-disp_S2": _skewed(symmetric, 4.0e-6),
    }

    with pytest.warns(InvariantWarning) as record:
        deviations = check_symmetric(matrices, where="monomer A")

    assert len(record) == 1
    message = str(record[0].message)
    for text in ("monomer A", "'exch-ind_S2'", "2.5e-05", "'exch-disp_S2'", "4.0e-06"):
        assert text in message
    assert "'pol'" not in message
    assert "check_symmetry_atol" in message
    assert set(deviations) == {"exch-ind_S2", "exch-disp_S2"}
    assert deviations["exch-ind_S2"] == pytest.approx(2.5e-5)


def test_single_matrix_and_caller_location(symmetric):
    """A single matrix is named "matrix", and the warning points at the calling line."""

    with pytest.warns(InvariantWarning) as record:
        deviations = check_symmetric(_skewed(symmetric, 1e-6))

    assert set(deviations) == {"matrix"}
    assert record[0].filename == __file__


def test_nan_fails(symmetric):
    """A NaN deviation is not within any tolerance."""

    with pytest.warns(InvariantWarning):
        assert "matrix" in check_symmetric(_skewed(symmetric, np.nan))


def test_tolerance_from_option_and_argument(symmetric, options):
    """The tolerance is the option unless given; the argument wins over the option."""

    matrix = _skewed(symmetric, 1e-8)

    with pytest.warns(InvariantWarning):
        check_symmetric(matrix)  # default 1e-10

    set_propsapt_options({"check_symmetry_atol": 1e-6})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_symmetric(matrix) == {}

    with pytest.warns(InvariantWarning):
        check_symmetric(matrix, atol=1e-9)


def test_option_none_turns_the_check_off(symmetric, options):
    """With the option set to None nothing is checked."""

    set_propsapt_options({"check_symmetry_atol": None})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_symmetric(_skewed(symmetric, 1.0)) == {}


def test_warning_can_be_made_an_error(symmetric):
    """The dedicated category lets callers turn the warning into an error."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        with pytest.raises(InvariantWarning):
            check_symmetric(_skewed(symmetric, 1.0))


@pytest.mark.parametrize("atol", [-1e-10, np.inf, np.nan, True, "1e-10"])
def test_rejects_bad_tolerance(symmetric, atol):

    with pytest.raises(ValueError, match="tolerance"):
        check_symmetric(symmetric, atol=atol)


def test_rejects_bad_option(symmetric, options):

    set_propsapt_options({"check_symmetry_atol": -1.0})

    with pytest.raises(ValueError, match="check_symmetry_atol"):
        check_symmetric(symmetric)


@pytest.mark.parametrize("shape", [(3, 4), (4,), (2, 4, 4)])
def test_rejects_non_square(shape):

    with pytest.raises(ValueError, match="square"):
        check_symmetric({"bad": np.zeros(shape)})


# partition of unity, on a small grid of He-H2 in cc-pVDZ (no SCF needed)


@pytest.fixture(scope="module")
def small_basis():
    """cc-pVDZ basis set of He-H2."""

    molecule = psi4.core.Molecule.from_string("""
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
        """)
    molecule.update_geometry()

    return psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")


@pytest.fixture(scope="module")
def small_grid(small_basis):
    """A small molecular grid and the Becke weights of He and H2 on it."""

    grid = molecular_grid(small_basis, radial_points=20, spherical_points=50)
    weights = becke_fragment_weights(grid, {"A": [0], "B": [1, 2]})

    return grid.points, grid.weights, weights


def test_partition_exact_weights_pass(small_grid, small_basis):

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_partition_of_unity(*small_grid, small_basis) is None


def test_partition_ignores_points_out_of_reach(small_grid, small_basis):
    """Weights off by 1e-5 only where no basis function reaches pass, as LISA weights do."""

    points, weights, fragment_weights = small_grid
    far = np.array([[60.0, 0.0, 0.0], [0.0, -60.0, 0.0], [0.0, 0.0, 80.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            check_partition_of_unity(
                np.concatenate([points, far]),
                np.concatenate([weights, np.full(3, 1.0e3)]),
                {
                    "A": np.concatenate(
                        [fragment_weights["A"], np.full(3, 0.5 + 1e-5)]
                    ),
                    "B": np.concatenate([fragment_weights["B"], np.full(3, 0.5)]),
                },
                small_basis,
            )
            is None
        )


def test_partition_negative_quadrature_weight(small_grid, small_basis):
    """With a negative quadrature weight the defect is computed; exact weights pass."""

    points, weights, fragment_weights = small_grid
    weights = weights.copy()
    weights[0] = -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            check_partition_of_unity(points, weights, fragment_weights, small_basis)
            is None
        )


def test_partition_missing_fragment_warns(small_grid, small_basis):
    """Weights of He alone miss H2: one warning, an O(1) defect, from the calling line."""

    points, weights, fragment_weights = small_grid

    with pytest.warns(InvariantWarning) as record:
        defect = check_partition_of_unity(
            points, weights, {"A": fragment_weights["A"]}, small_basis, where="the test"
        )

    assert len(record) == 1
    assert record[0].filename == __file__
    message = str(record[0].message)
    for text in (
        "fragments A of the test",
        "check_partition_atol",
        "partition_of_unity=False",
    ):
        assert text in message
    assert defect > 0.1


def test_partition_tolerance(small_grid, small_basis, options):
    """Weights off by 1e-6 everywhere fail the default 1e-8, pass a looser tolerance, and
    pass anything with the option set to None."""

    points, weights, fragment_weights = small_grid
    shifted = {"A": fragment_weights["A"] + 1e-6, "B": fragment_weights["B"]}

    with pytest.warns(InvariantWarning):
        defect = check_partition_of_unity(points, weights, shifted, small_basis)
    assert 1e-7 < defect <= 1.01e-6

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            check_partition_of_unity(points, weights, shifted, small_basis, atol=1e-5)
            is None
        )
        set_propsapt_options({"check_partition_atol": None})
        assert (
            check_partition_of_unity(points, weights, {"A": 0 * weights}, small_basis)
            is None
        )


def test_partition_rejects_bad_shapes(small_grid, small_basis):

    points, weights, fragment_weights = small_grid

    with pytest.raises(ValueError, match="weights"):
        check_partition_of_unity(points, weights[:-1], fragment_weights, small_basis)
    with pytest.raises(ValueError, match="fragment `A`"):
        check_partition_of_unity(
            points, weights, {"A": fragment_weights["A"][:-1]}, small_basis
        )
    with pytest.raises(ValueError, match="nonempty"):
        check_partition_of_unity(points, weights, {}, small_basis)


# charge closure


def test_closure_scalar():

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        closure = check_charge_closure({"A": 9.9, "B": 0.1 + 1e-9}, 10.0)
    assert closure == pytest.approx(1e-9, abs=1e-15)

    with pytest.warns(InvariantWarning, match=r"\+1\.0e-05 e") as record:
        check_charge_closure({"A": 9.9, "B": 0.1 + 1e-5}, 10.0, where="water dimer")

    assert len(record) == 1
    assert record[0].filename == __file__
    for text in ("fragments A, B of water dimer", "check_closure_atol", "Tr(D S)"):
        assert text in str(record[0].message)


def test_closure_several_densities():
    """One warning lists the densities that fail, by index, and the largest error."""

    total = np.zeros(5)
    populations = {"A": np.array([1e-3, -2e-3, 0.0, 5e-4, 1e-2]), "B": np.zeros(5)}
    populations["B"] = -populations["A"] + np.array([0.0, 3e-6, 1e-9, -8e-6, 0.0])

    with pytest.warns(InvariantWarning) as record:
        closure = check_charge_closure(populations, total)

    assert len(record) == 1
    message = str(record[0].message)
    assert "for 2 of 5 densities (indices 1, 3)" in message
    assert "up to -8.0e-06 e" in message
    assert closure.shape == (5,)
    np.testing.assert_allclose(closure, [0.0, 3e-6, 1e-9, -8e-6, 0.0], atol=1e-18)


def test_closure_nan_fails():

    with pytest.warns(InvariantWarning, match=r"for 1 of 2 densities \(index 1\)"):
        check_charge_closure({"A": np.array([1.0, np.nan])}, np.ones(2))


def test_closure_option_none(options):

    set_propsapt_options({"check_closure_atol": None})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_charge_closure({"A": 1.0}, 2.0) is None


def test_closure_rejects_bad_input():

    with pytest.raises(ValueError, match="shape"):
        check_charge_closure({"A": np.zeros(3)}, np.zeros(2))
    with pytest.raises(ValueError, match="nonempty"):
        check_charge_closure({}, 0.0)
