import pytest

import numpy as np
import psi4

from prop_sapt import reset_propsapt_options, set_propsapt_options
from prop_sapt.evaluate import (
    evaluate_basis_functions,
    evaluate_density,
    evaluate_esp,
    evaluate_orbital,
)

N_POINTS = 23  # not a multiple of the chunk sizes used below
HE_POSITION = np.array([-2.5, 0.0, 0.0])  # bohr, as in `prepare_dimer`


@pytest.fixture(scope="module")
def basisset(prepare_dimer):
    """Dimer basis set."""

    return prepare_dimer.get_psi4_basisset()


@pytest.fixture(scope="module")
def density_A(prepare_dimer):
    """Monomer A density matrix in AO basis."""

    return 2 * np.asarray(prepare_dimer.wfnA.Da())  # RHF spin


@pytest.fixture(scope="module")
def coeffs_A(prepare_dimer):
    """Monomer A MO coefficients in AO basis, orbitals as columns."""

    return np.asarray(prepare_dimer.wfnA.Ca())


@pytest.fixture(scope="module")
def points():
    """Random points around the He-H2 dimer, in bohr."""

    rng = np.random.default_rng(7)
    return rng.uniform([-6.0, -3.0, -3.0], [6.0, 3.0, 3.0], size=(N_POINTS, 3))


@pytest.fixture(scope="module")
def phi_ref(basisset, points):
    """Basis function values from Psi4's per-point `compute_phi`."""

    return np.array([np.asarray(basisset.compute_phi(*p)).reshape(-1) for p in points])


@pytest.fixture(scope="module")
def quadrature_grid(basisset):
    """Points and weights of Psi4's molecular DFT quadrature grid."""

    functional = psi4.driver.dft.build_superfunctional("svwn", True)[0]
    vbase = psi4.core.VBase.build(basisset, functional, "RV")
    vbase.initialize()
    x, y, z, w = (np.asarray(v) for v in vbase.get_np_xyzw())

    return np.column_stack([x, y, z]), w


def _psi4_esp(basisset, density, points):
    """Electronic ESP at `points` straight from Psi4 analytic integrals."""

    mints = psi4.core.MintsHelper(basisset)
    esp = mints.electrostatic_potential_value(
        psi4.core.Vector.from_array(np.ones(len(points))),
        psi4.core.Matrix.from_array(np.asarray(points, dtype=float)),
        psi4.core.Matrix.from_array(density),
    )

    return np.asarray(esp)


def test_basis_functions_match_compute_phi(basisset, points, phi_ref):
    """Batched values equal `compute_phi`, also across chunk boundaries."""

    phi = evaluate_basis_functions(points, basisset, chunk_size=7)

    assert phi.shape == (N_POINTS, basisset.nbf())
    np.testing.assert_allclose(phi, phi_ref, rtol=0, atol=1e-14)


def test_points_shape_is_kept(basisset, points, phi_ref, density_A):
    """Points of shape (..., 3) give values of shape (...)."""

    grid_points = points[:20].reshape(4, 5, 3)

    phi = evaluate_basis_functions(grid_points, basisset)
    rho = evaluate_density(density_A, grid_points, basisset)

    assert phi.shape == (4, 5, basisset.nbf())
    assert rho.shape == (4, 5)
    np.testing.assert_allclose(phi.reshape(20, -1), phi_ref[:20], rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        rho.reshape(20), evaluate_density(density_A, points[:20], basisset), rtol=1e-14
    )


def test_density_matches_reference(basisset, density_A, points, phi_ref):
    """rho(r) = phi(r)^T D phi(r) at every point."""

    expected = np.einsum("pi,ij,pj->p", phi_ref, density_A, phi_ref)

    rho = evaluate_density(density_A, points, basisset, chunk_size=7)

    assert rho.shape == (N_POINTS,)
    np.testing.assert_allclose(rho, expected, rtol=1e-12, atol=1e-14)


def test_density_several_matrices(prepare_dimer, basisset, density_A, points):
    """A list and a stacked array give one row per matrix, as single calls do."""

    density_B = 2 * np.asarray(prepare_dimer.wfnB.Da())
    single = np.array(
        [evaluate_density(m, points, basisset) for m in (density_A, density_B)]
    )

    from_list = evaluate_density([density_A, density_B], points, basisset)
    from_stack = evaluate_density(
        np.stack([density_A, density_B]), points, basisset, chunk_size=5
    )

    assert from_list.shape == (2, N_POINTS)
    np.testing.assert_array_equal(from_list, single)
    np.testing.assert_allclose(from_stack, single, rtol=1e-14, atol=1e-16)


def test_density_integrates_to_electron_count(basisset, density_A, quadrature_grid):
    """The He density integrates to its 2 electrons on a quadrature grid."""

    grid_points, weights = quadrature_grid

    n_elec = np.sum(weights * evaluate_density(density_A, grid_points, basisset))

    assert n_elec == pytest.approx(2.0, abs=1e-6)


def test_orbital_shapes_and_values(basisset, coeffs_A, points, phi_ref):
    """Orbitals are the columns of the MO coefficients, one output row each."""

    one = evaluate_orbital(coeffs_A[:, 0], points, basisset)
    column = evaluate_orbital(coeffs_A[:, [0]], points, basisset)
    several = evaluate_orbital(coeffs_A[:, :3], points, basisset, chunk_size=7)

    assert one.shape == (N_POINTS,)
    assert column.shape == (1, N_POINTS)
    assert several.shape == (3, N_POINTS)
    np.testing.assert_array_equal(column[0], one)
    np.testing.assert_allclose(
        several, (phi_ref @ coeffs_A[:, :3]).T, rtol=1e-12, atol=1e-14
    )


def test_occupied_orbital_is_normalized(basisset, coeffs_A, quadrature_grid):
    """The occupied He orbital integrates to 1 on a quadrature grid."""

    grid_points, weights = quadrature_grid

    psi = evaluate_orbital(coeffs_A[:, 0], grid_points, basisset)

    assert np.sum(weights * psi**2) == pytest.approx(1.0, abs=1e-6)


def test_esp_matches_psi4(basisset, density_A, points):
    """The electronic ESP equals Psi4's analytic potential, also for several matrices."""

    expected = _psi4_esp(basisset, density_A, points)

    esp = evaluate_esp(density_A, points, basisset)
    esp_pair = evaluate_esp([density_A, 0.5 * density_A], points, basisset)

    assert esp.shape == (N_POINTS,)
    np.testing.assert_allclose(esp, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(esp_pair, [expected, 0.5 * expected], rtol=0, atol=1e-12)


def test_esp_nuclei_skips_ghost_atoms(prepare_dimer, density_A, points):
    """In monomer A's basis set the ghost H atoms add no nuclear potential."""

    basis_A = prepare_dimer.wfnA.basisset()

    electronic = evaluate_esp(density_A, points, basis_A)
    total = evaluate_esp(density_A, points, basis_A, nuclei=True)

    expected = 2.0 / np.linalg.norm(points - HE_POSITION, axis=1)
    np.testing.assert_allclose(total - electronic, expected, rtol=1e-12, atol=0)


def test_esp_nuclei_skips_nucleus_on_point(prepare_dimer, basisset, density_A):
    """A nucleus lying on a point is skipped there, so the value stays finite."""

    on_nucleus = HE_POSITION[None]

    esp = evaluate_esp(density_A, on_nucleus, basisset, nuclei=True)

    mol = prepare_dimer.get_psi4_molecule()
    h_atoms = np.asarray(mol.geometry())[1:]
    expected = _psi4_esp(basisset, density_A, on_nucleus) + np.sum(
        1.0 / np.linalg.norm(h_atoms - HE_POSITION, axis=1)
    )
    np.testing.assert_allclose(esp, expected, rtol=1e-12, atol=0)


def test_dimer_methods_match_module_functions(
    prepare_dimer, basisset, density_A, coeffs_A, points
):
    """`Dimer.evaluate_*` use the dimer basis set."""

    np.testing.assert_array_equal(
        prepare_dimer.evaluate_density(density_A, points),
        evaluate_density(density_A, points, basisset),
    )
    np.testing.assert_array_equal(
        prepare_dimer.evaluate_orbital(coeffs_A[:, :2], points),
        evaluate_orbital(coeffs_A[:, :2], points, basisset),
    )
    np.testing.assert_array_equal(
        prepare_dimer.evaluate_esp(density_A, points, nuclei=True),
        evaluate_esp(density_A, points, basisset, nuclei=True),
    )


def test_chunk_size_option(basisset, points):
    """The `point_eval_chunk_size` option is the default chunk size."""

    set_propsapt_options({"point_eval_chunk_size": 0})
    try:
        with pytest.raises(ValueError):
            evaluate_basis_functions(points, basisset)
    finally:
        reset_propsapt_options("point_eval_chunk_size")


@pytest.mark.parametrize("chunk_size", [0, -1, 2.5, True])
def test_rejects_bad_chunk_size(basisset, points, chunk_size):
    with pytest.raises(ValueError):
        evaluate_basis_functions(points, basisset, chunk_size=chunk_size)


@pytest.mark.parametrize(
    "shape", [(5, 2), (), (5, 3, 2)], ids=["xy-only", "scalar", "last-axis-2"]
)
def test_rejects_points_without_xyz_axis(basisset, density_A, shape):
    with pytest.raises(ValueError):
        evaluate_density(density_A, np.zeros(shape), basisset)


def test_rejects_matrix_of_another_basis(basisset, points):
    nbf = basisset.nbf()

    with pytest.raises(ValueError):
        evaluate_density(np.eye(nbf - 1), points, basisset)
    with pytest.raises(ValueError):
        evaluate_density([np.eye(nbf), np.eye(nbf - 1)], points, basisset)
    with pytest.raises(ValueError):
        evaluate_esp(np.eye(nbf - 1), points, basisset)


def test_rejects_empty_matrix_list(basisset, points):
    with pytest.raises(ValueError):
        evaluate_density([], points, basisset)


def test_rejects_orbitals_as_rows(basisset, coeffs_A, points):
    """Rows of a (k, nbf) array are not taken as orbitals."""

    with pytest.raises(ValueError):
        evaluate_orbital(coeffs_A[:, :3].T, points, basisset)


def test_requires_psi4_basisset(prepare_dimer, density_A, points):
    with pytest.raises(TypeError):
        evaluate_density(density_A, points, prepare_dimer.get_psi4_molecule())
