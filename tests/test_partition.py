import warnings

import pytest

import numpy as np
import psi4
from psi4.driver import p4util

from prop_sapt import reset_propsapt_options, set_propsapt_options
from prop_sapt.checks import InvariantWarning
from prop_sapt.evaluate import evaluate_density
from prop_sapt.partition import (
    FragmentOperators,
    QuadratureGrid,
    becke_fragment_weights,
    hirshfeld_fragment_weights,
    molecular_grid,
)

FRAGMENT_ATOMS = {"A": [0], "B": [1, 2]}  # He | H2, as in `prepare_dimer`
QUADRATURE_ATOL = 1e-6  # quadrature error of the default grid, 5e-7 measured


@pytest.fixture(scope="module")
def basisset(prepare_dimer):
    """Dimer basis set."""

    return prepare_dimer.get_psi4_basisset()


@pytest.fixture(scope="module")
def overlap(basisset):
    """Analytic AO overlap matrix."""

    return np.asarray(psi4.core.MintsHelper(basisset).ao_overlap())


@pytest.fixture(scope="module")
def densities(prepare_dimer):
    """Unperturbed monomer A and B density matrices in AO basis."""

    return {
        "A": 2 * np.asarray(prepare_dimer.wfnA.Da()),  # RHF spin
        "B": 2 * np.asarray(prepare_dimer.wfnB.Da()),
    }


@pytest.fixture(scope="module")
def grid(basisset):
    """Default molecular quadrature grid of the dimer."""

    return molecular_grid(basisset)


@pytest.fixture(scope="module")
def becke_weights(grid):
    """Becke fragment weights of He and H2."""

    return becke_fragment_weights(grid, FRAGMENT_ATOMS)


@pytest.fixture(scope="module")
def becke_ops(grid, becke_weights, basisset):
    """Fragment operators of the Becke partition; an invariant warning fails."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        return FragmentOperators.from_grid(
            grid.points, grid.weights, becke_weights, basisset
        )


@pytest.fixture(scope="module")
def hirshfeld_weights(grid, densities, basisset):
    """Hirshfeld weights from the monomer densities."""

    return hirshfeld_fragment_weights(grid.points, densities, basisset)


@pytest.fixture(scope="module")
def hirshfeld_ops(grid, hirshfeld_weights, basisset):
    """Fragment operators of the Hirshfeld partition; an invariant warning fails."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        return FragmentOperators.from_grid(
            grid.points, grid.weights, hirshfeld_weights, basisset
        )


@pytest.fixture
def no_invariant_warnings():
    """An invariant warning in the test fails it."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        yield


@pytest.fixture(scope="module")
def ghost_basisset():
    """Basis set of He-H2 with a ghost He between the monomers, a third fragment."""

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
        --
        0 1
        @He  0.000000000   0.000000000   0.000000000
        """)
    molecule.update_geometry()

    return psi4.core.BasisSet.build(molecule, "ORBITAL", "cc-pvdz")


def test_molecular_grid(grid, basisset):
    """Points, weights and parent atoms of every atom of the dimer."""

    n_points = len(grid.weights)

    assert isinstance(grid, QuadratureGrid)
    assert grid.points.shape == (n_points, 3)
    assert grid.atom_index.shape == (n_points,)
    assert set(grid.atom_index.tolist()) == {0, 1, 2}
    assert np.all(grid.weights > 0.0)


def test_molecular_grid_keeps_psi4_options(basisset):
    """The radial extent is set only while the grid is built."""

    saved_options = p4util.OptionsState(["DFT_BS_RADIUS_ALPHA"])
    try:
        default = psi4.core.get_global_option("DFT_BS_RADIUS_ALPHA")
        molecular_grid(basisset, radial_points=20, spherical_points=110)
        assert psi4.core.get_global_option("DFT_BS_RADIUS_ALPHA") == default
        assert not psi4.core.has_global_option_changed("DFT_BS_RADIUS_ALPHA")

        psi4.set_options({"DFT_BS_RADIUS_ALPHA": 1.5})
        molecular_grid(basisset, radial_points=20, spherical_points=110)
        assert psi4.core.get_global_option("DFT_BS_RADIUS_ALPHA") == 1.5
        assert psi4.core.has_global_option_changed("DFT_BS_RADIUS_ALPHA")
    finally:
        saved_options.restore()


def test_molecular_grid_options(basisset):
    """The `quadrature_*` options are the default grid sizes and extent."""

    explicit = molecular_grid(
        basisset, radial_points=20, spherical_points=110, bs_radius_alpha=1.5
    )

    set_propsapt_options(
        {
            "quadrature_radial_points": 20,
            "quadrature_spherical_points": 110,
            "quadrature_bs_radius_alpha": 1.5,
        }
    )
    try:
        from_options = molecular_grid(basisset)
    finally:
        reset_propsapt_options(
            [
                "quadrature_radial_points",
                "quadrature_spherical_points",
                "quadrature_bs_radius_alpha",
            ]
        )

    np.testing.assert_array_equal(from_options.points, explicit.points)
    np.testing.assert_array_equal(from_options.weights, explicit.weights)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radial_points": 0},
        {"spherical_points": 2.5},
        {"bs_radius_alpha": 0.0},
        {"bs_radius_alpha": True},
    ],
)
def test_molecular_grid_rejects_bad_sizes(basisset, kwargs):
    with pytest.raises(ValueError):
        molecular_grid(basisset, **kwargs)


def test_becke_weights_are_a_partition(becke_weights):
    """Every point belongs to exactly one fragment."""

    total = becke_weights["A"] + becke_weights["B"]

    assert set(np.unique(becke_weights["A"]).tolist()) == {0.0, 1.0}
    np.testing.assert_array_equal(total, 1.0)


@pytest.mark.parametrize("ops_name", ["becke_ops", "hirshfeld_ops"])
def test_operators_sum_to_overlap(request, ops_name, overlap):
    """W_A + W_B is the overlap matrix, up to the quadrature error."""

    ops = request.getfixturevalue(ops_name)

    assert ops.fragments == ("A", "B")
    assert ops.nbf == overlap.shape[0]
    np.testing.assert_allclose(
        ops.overlap("A") + ops.overlap("B"), overlap, rtol=0, atol=QUADRATURE_ATOL
    )


def test_charge_matches_direct_quadrature(
    becke_ops, becke_weights, grid, densities, basisset
):
    """-Tr(D W_X) equals -sum_q w_q w_X(r_q) rho(r_q) on the same grid."""

    rho = evaluate_density(densities["A"], grid.points, basisset)

    for fragment in ("A", "B"):
        expected = -np.sum(grid.weights * becke_weights[fragment] * rho)
        assert becke_ops.charge(fragment, densities["A"]) == pytest.approx(
            expected, rel=1e-11, abs=1e-14
        )


@pytest.mark.parametrize("ops_name", ["becke_ops", "hirshfeld_ops"])
def test_monomer_populations(request, ops_name, densities):
    """The He density puts its 2 electrons on the grid, nearly all on He."""

    ops = request.getfixturevalue(ops_name)

    on_A = ops.population("A", densities["A"])
    on_B = ops.population("B", densities["A"])

    assert on_A + on_B == pytest.approx(2.0, abs=QUADRATURE_ATOL)
    assert on_A > 1.9
    assert ops.charge("A", densities["A"]) == -on_A


def test_several_matrices(becke_ops, densities):
    """A list and a stacked array give one value per matrix, as single calls do."""

    matrices = [densities["A"], densities["B"]]
    single = np.array([becke_ops.population("A", m) for m in matrices])
    single_dipoles = np.array([becke_ops.dipole("A", m) for m in matrices])

    from_list = becke_ops.population("A", matrices)
    from_stack = becke_ops.charge("A", np.stack(matrices))
    dipoles = becke_ops.dipole("A", matrices)

    assert from_list.shape == (2,)
    assert dipoles.shape == (2, 3)
    np.testing.assert_allclose(from_list, single, rtol=1e-14)
    np.testing.assert_allclose(from_stack, -single, rtol=1e-14)
    np.testing.assert_allclose(dipoles, single_dipoles, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize("ops_name", ["becke_ops", "hirshfeld_ops"])
def test_dipoles_sum_to_electronic_dipole(request, ops_name, basisset, densities):
    """At the coordinate origin, the fragment dipoles sum to Tr(D ao_dipole)."""

    ops = request.getfixturevalue(ops_name)
    ao_dipole = np.array(
        [np.asarray(m) for m in psi4.core.MintsHelper(basisset).ao_dipole()]
    )

    for density in densities.values():
        expected = np.einsum("ij,kji->k", density, ao_dipole)
        total = ops.dipole("A", density) + ops.dipole("B", density)

        np.testing.assert_allclose(total, expected, rtol=0, atol=QUADRATURE_ATOL)


def test_dipole_origin(becke_ops, densities):
    """Moving the origin by O adds O times the population."""

    origin = np.array([0.3, -1.2, 2.0])

    for fragment in ("A", "B"):
        shifted = becke_ops.dipole(fragment, densities["A"], origin=origin)
        expected = becke_ops.dipole(fragment, densities["A"]) + origin * (
            becke_ops.population(fragment, densities["A"])
        )

        np.testing.assert_allclose(shifted, expected, rtol=1e-13, atol=1e-13)

    with pytest.raises(ValueError):
        becke_ops.dipole("A", densities["A"], origin=[0.0, 0.0])


def test_chunk_size_and_points_shape(grid, becke_weights, basisset):
    """The operators do not depend on the chunk size or on the shape of the points."""

    subset = slice(0, 600)
    weights = {name: w[subset] for name, w in becke_weights.items()}
    reference = FragmentOperators.from_grid(
        grid.points[subset], grid.weights[subset], weights, basisset
    )

    chunked = FragmentOperators.from_grid(
        grid.points[subset], grid.weights[subset], weights, basisset, chunk_size=7
    )
    reshaped = FragmentOperators.from_grid(
        grid.points[subset].reshape(20, 30, 3),
        grid.weights[subset].reshape(20, 30),
        {name: w.reshape(20, 30) for name, w in weights.items()},
        basisset,
    )

    for ops in (chunked, reshaped):
        for fragment in ("A", "B"):
            np.testing.assert_allclose(
                ops.overlap(fragment),
                reference.overlap(fragment),
                rtol=1e-12,
                atol=1e-16,
            )
            np.testing.assert_allclose(
                ops.first_moment(fragment),
                reference.first_moment(fragment),
                rtol=1e-12,
                atol=1e-16,
            )


def test_operators_are_read_only(becke_ops):
    with pytest.raises(ValueError):
        becke_ops.overlap("A")[0, 0] = 1.0
    with pytest.raises(ValueError):
        becke_ops.first_moment("A")[0, 0, 0] = 1.0


def test_without_dipole(
    grid, becke_weights, basisset, densities, no_invariant_warnings
):
    """Operators built with `dipole=False` give populations but no dipoles."""

    subset = slice(0, 100)  # a part of space: not a partition of unity
    ops = FragmentOperators.from_grid(
        grid.points[subset],
        grid.weights[subset],
        {name: w[subset] for name, w in becke_weights.items()},
        basisset,
        dipole=False,
        partition_of_unity=False,
    )

    ops.population("A", densities["A"])
    with pytest.raises(ValueError):
        ops.dipole("A", densities["A"])
    with pytest.raises(ValueError):
        ops.first_moment("A")


def test_hirshfeld_weights(hirshfeld_weights, densities, basisset):
    """The weights lie in [0, 1], sum to 1, and are split equally far away."""

    total = hirshfeld_weights["A"] + hirshfeld_weights["B"]

    assert np.all((hirshfeld_weights["A"] >= 0.0) & (hirshfeld_weights["A"] <= 1.0))
    np.testing.assert_allclose(total, 1.0, rtol=0, atol=1e-14)

    far = hirshfeld_fragment_weights(np.array([[1.0e4, 0.0, 0.0]]), densities, basisset)
    assert far["A"][0] == far["B"][0] == 0.5


def test_hirshfeld_identity(hirshfeld_ops, densities):
    """With w_X = rho_X / (rho_A + rho_B), the integral of rho_A w_A - rho_B w_B
    is that of rho_A - rho_B, i.e. N_A - N_B = 0 for He and H2."""

    difference = hirshfeld_ops.population("A", densities["A"]) - (
        hirshfeld_ops.population("B", densities["B"])
    )

    assert difference == pytest.approx(0.0, abs=QUADRATURE_ATOL)


def test_becke_rejects_unassigned_atoms(grid, ghost_basisset):
    """Every atom with grid points, ghost atoms included, needs a fragment."""

    with pytest.raises(ValueError, match="belong to no fragment"):
        becke_fragment_weights(grid, {"A": [0], "B": [1]})

    ghost_grid = molecular_grid(ghost_basisset, radial_points=20, spherical_points=110)
    assert set(ghost_grid.atom_index.tolist()) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="belong to no fragment"):
        becke_fragment_weights(ghost_grid, FRAGMENT_ATOMS)

    weights = becke_fragment_weights(ghost_grid, {"A": [0, 3], "B": [1, 2]})
    np.testing.assert_array_equal(weights["A"] + weights["B"], 1.0)


@pytest.mark.parametrize(
    "fragment_atoms",
    [{"A": [0, 1], "B": [1, 2]}, {"A": [0.0], "B": [1, 2]}, {"A": [True], "B": [1, 2]}],
    ids=["shared-atom", "float-index", "bool-index"],
)
def test_becke_rejects_bad_atoms(grid, fragment_atoms):
    with pytest.raises(ValueError):
        becke_fragment_weights(grid, fragment_atoms)


def test_from_grid_rejects_bad_input(grid, becke_weights, basisset, prepare_dimer):
    subset = slice(0, 50)
    points = grid.points[subset]
    weights = grid.weights[subset]
    fragment_weights = {name: w[subset] for name, w in becke_weights.items()}

    with pytest.raises(ValueError):
        FragmentOperators.from_grid(points, weights[:-1], fragment_weights, basisset)
    with pytest.raises(ValueError):
        FragmentOperators.from_grid(
            points, weights, {"A": fragment_weights["A"][:-1]}, basisset
        )
    with pytest.raises(ValueError):
        FragmentOperators.from_grid(
            points, weights, {"A": np.full(50, np.nan)}, basisset
        )
    with pytest.raises(ValueError):
        FragmentOperators.from_grid(points, weights, {}, basisset)
    with pytest.raises(TypeError):
        FragmentOperators.from_grid(
            points, weights, list(fragment_weights.values()), basisset
        )
    with pytest.raises(TypeError):
        FragmentOperators.from_grid(
            points, weights, fragment_weights, prepare_dimer.get_psi4_molecule()
        )


def test_rejects_unknown_fragment_and_bad_matrix(becke_ops, densities):
    nbf = becke_ops.nbf

    with pytest.raises(KeyError):
        becke_ops.population("C", densities["A"])
    with pytest.raises(ValueError):
        becke_ops.population("A", np.eye(nbf - 1))
    with pytest.raises(ValueError):
        becke_ops.dipole("A", [np.eye(nbf), np.eye(nbf - 1)])


def test_constructor_rejects_inconsistent_matrices():
    with pytest.raises(ValueError):
        FragmentOperators({"A": np.eye(3), "B": np.eye(4)})
    with pytest.raises(ValueError):
        FragmentOperators({"A": np.eye(3)}, {"B": np.zeros((3, 3, 3))})
    with pytest.raises(ValueError):
        FragmentOperators({"A": np.eye(3)}, {"A": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="ao_overlap"):
        FragmentOperators({"A": np.eye(3)}, ao_overlap=np.eye(4))


@pytest.mark.parametrize("weights", ["becke", "hirshfeld"])
def test_dimer_fragment_operators(
    prepare_dimer, weights, becke_ops, hirshfeld_ops, densities
):
    """The `Dimer` method equals the explicit route on the default grid."""

    expected = {"becke": becke_ops, "hirshfeld": hirshfeld_ops}[weights]

    ops = prepare_dimer.fragment_operators(weights)

    assert ops.fragments == ("A", "B")
    for fragment in ("A", "B"):
        np.testing.assert_allclose(
            ops.overlap(fragment), expected.overlap(fragment), rtol=1e-14, atol=1e-16
        )
        np.testing.assert_allclose(
            ops.dipole(fragment, densities["B"]),
            expected.dipole(fragment, densities["B"]),
            rtol=1e-14,
            atol=1e-16,
        )


def test_dimer_fragment_operators_with_arrays(prepare_dimer, grid, becke_weights):
    """Weights given as arrays need their grid; grid options need no grid."""

    ops = prepare_dimer.fragment_operators(becke_weights, grid=grid, dipole=False)
    assert ops.fragments == ("A", "B")
    assert ops.ao_overlap is not None

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        partial = prepare_dimer.fragment_operators(
            {"A": becke_weights["A"]},
            grid=grid,
            dipole=False,
            partition_of_unity=False,
        )
    assert partial.fragments == ("A",)
    assert partial.ao_overlap is None

    with pytest.raises(ValueError):
        prepare_dimer.fragment_operators(becke_weights)
    with pytest.raises(TypeError):
        prepare_dimer.fragment_operators(grid=grid, radial_points=20)
    with pytest.raises(ValueError):
        prepare_dimer.fragment_operators("lisa")


# invariant checks: partition of unity and charge closure


@pytest.fixture
def options():
    """Restores the default propSAPT options after the test."""

    yield
    reset_propsapt_options()


def test_closure_error(becke_ops, overlap, densities):
    """sum_X Tr(D W_X) - Tr(D S), within the quadrature error, one value per matrix."""

    assert becke_ops.ao_overlap.shape == overlap.shape
    np.testing.assert_allclose(becke_ops.ao_overlap, overlap, rtol=0, atol=1e-15)

    for density in densities.values():
        expected = (
            becke_ops.population("A", density)
            + becke_ops.population("B", density)
            - np.trace(density @ overlap)
        )
        closure = becke_ops.closure_error(density)
        assert closure == pytest.approx(expected, abs=1e-13)  # cancellation of ~2 e
        assert abs(closure) < QUADRATURE_ATOL

    several = becke_ops.closure_error(list(densities.values()))
    assert several.shape == (2,)


def test_population_and_charge_warn_on_closure(becke_ops, densities, options):
    """Below the quadrature error every population and charge call warns once, from the
    calling line, listing the densities that fail."""

    set_propsapt_options({"check_closure_atol": 1e-14})

    with pytest.warns(InvariantWarning, match="check_closure_atol") as record:
        becke_ops.population("A", densities["A"])
    assert len(record) == 1
    assert record[0].filename == __file__

    with pytest.warns(InvariantWarning, match="for 2 of 2 densities") as record:
        becke_ops.charge("B", [densities["A"], densities["B"]])
    assert len(record) == 1
    assert record[0].filename == __file__

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        becke_ops.dipole("A", densities["A"])  # not checked


def test_partial_fragments(grid, becke_weights, basisset, densities):
    """One fragment alone warns by default and works with `partition_of_unity=False`."""

    only_A = {"A": becke_weights["A"]}

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        ops = FragmentOperators.from_grid(
            grid.points,
            grid.weights,
            only_A,
            basisset,
            dipole=False,
            partition_of_unity=False,
        )
        on_A = ops.population("A", densities["A"])

    assert on_A > 1.9
    assert ops.ao_overlap is None
    with pytest.raises(ValueError, match="partition of unity"):
        ops.closure_error(densities["A"])

    with pytest.warns(InvariantWarning, match="check_partition_atol") as record:
        ops = FragmentOperators.from_grid(
            grid.points, grid.weights, only_A, basisset, dipole=False
        )
    assert record[0].filename == __file__

    with pytest.warns(InvariantWarning, match="check_closure_atol"):
        ops.population("A", densities["B"])
