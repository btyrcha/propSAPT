import warnings

import pytest

import numpy as np

import prop_sapt.densities as densities_module
import prop_sapt.densities_components.exch_disp as exch_disp_module
from prop_sapt import calc_densities, calc_density_matrix, DENSITY_COMPONENTS
from prop_sapt.checks import InvariantWarning
from prop_sapt.densities_components import (
    get_exch_disp_s2_density,
    get_exch_ind_density,
    get_exch_ind_s2_density,
)

# Each expected value is the corresponding analytic property term of
# tests/test_calc_property.py, along X for the He-H2 dimer in a.u.: a density component
# contracted with the dipole matrix must reproduce the property term of the same name,
# and those terms are now the field derivative of the SAPT energies. Components that do
# not yet reproduce theirs are marked xfail and need the same corrections the property
# path got - the one-body part of [W, C - C^dagger] in the Fock derivative, the two-body
# part in the coupled amplitudes, and no terms contracting the CPHF amplitudes into the
# dispersion ones.


@pytest.fixture(scope="module")
def prepare_dipole_along_x(prepare_dimer):
    """Prepare dipole moment AO matrix along X axis."""

    d_vec = prepare_dimer.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]
    d_X = d_vec[0]

    return d_X


@pytest.fixture(scope="module")
def get_density_matrices(prepare_dimer):
    """Get interaction-induced density matrices for monomers A and B; an invariant
    warning fails every test of the module."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", InvariantWarning)
        rho_A = calc_density_matrix(prepare_dimer, "A", orbital_basis="AO")
        rho_B = calc_density_matrix(prepare_dimer, "B", orbital_basis="AO")

    return rho_A, rho_B


def test_component_keys(get_density_matrices):
    """The components are keyed and ordered by `DENSITY_COMPONENTS`."""

    for rho in get_density_matrices:
        assert tuple(rho) == DENSITY_COMPONENTS


def test_components_are_symmetric(get_density_matrices):
    """Every component is a symmetric matrix, the exchange ones included: their oo and vv
    blocks are built as the symmetric part of a product of two different amplitudes."""

    for rho in get_density_matrices:
        for key, matrix in rho.items():
            assert np.max(np.abs(matrix - matrix.T)) <= 1e-14, key


def test_non_symmetric_component_warns(prepare_dimer, monkeypatch):
    """A non-symmetric component warns, naming it and the sums it enters, and the
    warning points at the caller."""

    def skewed(build):
        def builder(mol, monomer):
            rho = yield from build(mol, monomer)
            rho[0, 1] += 1e-6
            return rho

        return builder

    components = tuple(
        (key, skewed(build) if key == "exch-disp_S2" else build, name)
        for key, build, name in densities_module.BATCHED_COMPONENTS
    )
    monkeypatch.setattr(densities_module, "BATCHED_COMPONENTS", components)

    with pytest.warns(InvariantWarning, match="monomer A") as record:
        calc_density_matrix(prepare_dimer, "A", orbital_basis="MO")

    assert len(record) == 1
    message = str(record[0].message)
    for key in ("'exch-disp_S2'", "'Disp'", "'total'"):
        assert key in message
    assert "'pol'" not in message
    assert record[0].filename == __file__


@pytest.fixture
def empty_theta_cache(prepare_dimer, monkeypatch):
    """The dimer with no cached theta amplitudes; its own cache is restored afterwards."""

    monkeypatch.setattr(prepare_dimer, "theta_amplitudes", {})

    return prepare_dimer


@pytest.mark.parametrize(
    "builder, key",
    [(get_exch_ind_s2_density, "exch-ind_S2"), (get_exch_ind_density, "exch-ind")],
)
def test_exch_ind_theta_solved_once(empty_theta_cache, monkeypatch, builder, key):
    """The first monomer's call solves for the theta amplitudes of both monomers and
    keeps them; the other monomer's call solves only for its density."""

    mol = empty_theta_cache
    mol.get_cpscf_ra()
    mol.get_cpscf_sb()

    solves = []
    cpscf_batch = mol.cpscf_batch

    def counted_cpscf_batch(monomer, perturbations, **kwargs):
        solves.extend([monomer] * len(perturbations))
        return cpscf_batch(monomer, perturbations, **kwargs)

    monkeypatch.setattr(mol, "cpscf_batch", counted_cpscf_batch)

    builder(mol, "A")
    assert sorted(solves) == ["A", "A", "B"]

    solves.clear()
    builder(mol, "B")
    assert solves == ["B"]

    assert not any(theta.flags.writeable for theta in mol.theta_amplitudes[key])


def test_density_matrix_batches_cpscf(empty_theta_cache, monkeypatch):
    """calc_density_matrix solves the CPSCF equations of its components in batches.
    The first monomer: its six components with the theta amplitudes of both monomers,
    then its two exchange-induction components, which need them. The second monomer,
    whose theta amplitudes are kept: its six components at once."""

    mol = empty_theta_cache
    mol.get_cpscf_ra()
    mol.get_cpscf_sb()

    batches = []
    cpscf_batch = mol.cpscf_batch

    def counted_cpscf_batch(monomer, perturbations, **kwargs):
        batches.append((monomer, len(perturbations)))
        return cpscf_batch(monomer, perturbations, **kwargs)

    monkeypatch.setattr(mol, "cpscf_batch", counted_cpscf_batch)

    calc_density_matrix(mol, "A")
    assert batches == [("A", 6), ("B", 2), ("A", 2)]

    batches.clear()
    calc_density_matrix(mol, "B")
    assert batches == [("B", 6)]


@pytest.mark.parametrize("monomer", ["A", "B"])
def test_batched_components_match_single_solves(empty_theta_cache, monomer):
    """The batched solves give each component the amplitudes of its own equations, as
    solving them one at a time does."""

    mol = empty_theta_cache
    rho = calc_density_matrix(mol, monomer, orbital_basis="MO")

    single = {
        "exch-ind_S2": get_exch_ind_s2_density(mol, monomer),
        "exch-ind": get_exch_ind_density(mol, monomer) - rho["ind"],
        "exch-disp_S2": get_exch_disp_s2_density(mol, monomer),
    }
    for key, matrix in single.items():
        np.testing.assert_allclose(rho[key], matrix, rtol=0, atol=1e-12, err_msg=key)


def test_exch_disp_s2_theta_built_once(empty_theta_cache, monkeypatch):
    """The exchange-dispersion theta amplitudes are built once for both monomers."""

    mol = empty_theta_cache

    builds = []
    build = exch_disp_module._build_exch_disp_s2_theta

    def counted_build(dimer):
        builds.append(dimer)
        return build(dimer)

    monkeypatch.setattr(exch_disp_module, "_build_exch_disp_s2_theta", counted_build)

    get_exch_disp_s2_density(mol, "A")
    get_exch_disp_s2_density(mol, "B")

    assert len(builds) == 1
    assert not mol.theta_amplitudes["exch-disp_S2"].flags.writeable


def test_calc_densities_frees_theta_amplitudes(empty_theta_cache, monkeypatch):
    """calc_densities frees the theta amplitudes once both monomers are done, and
    also when a monomer fails."""

    mol = empty_theta_cache
    fail = False

    def fill_cache(mol, monomer, orbital_basis):
        mol.theta_amplitudes[monomer] = np.zeros(1)
        if monomer == "B" and fail:
            raise RuntimeError("monomer B failed")
        return {}

    monkeypatch.setattr(densities_module, "calc_density_matrix", fill_cache)

    calc_densities(mol, orbital_basis="MO")
    assert mol.theta_amplitudes == {}

    fail = True
    with pytest.raises(RuntimeError, match="monomer B failed"):
        calc_densities(mol, orbital_basis="MO")
    assert mol.theta_amplitudes == {}


def test_polarization_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = -3.695584988e-04  # x1_pol,r

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_pol = rho_A["pol"] + rho_B["pol"]

    result = 2 * np.trace(rho_pol @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_exchange_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = -1.581693343e-02  # x1_exch,r

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch = rho_A["exch"] + rho_B["exch"]

    result = 2 * np.trace(rho_exch @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_induction_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = +1.042482216e-03  # x2_ind,r
    # expected_result = 0.00104248625  # from MOLPRO finite-field SAPT calculation

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_ind = rho_A["ind"] + rho_B["ind"]

    result = 2 * np.trace(rho_ind @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_dispersion_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = +2.195424318e-03  # x2_disp

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_disp = rho_A["disp"] + rho_B["disp"]

    result = 2 * np.trace(rho_disp @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_exch_ind_s2_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = -2.095951859e-03  # x2_exch-ind,r_S2

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind_s2 = rho_A["exch-ind_S2"] + rho_B["exch-ind_S2"]

    result = 2 * np.trace(rho_exch_ind_s2 @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_exch_ind_sinf_from_density_matrix(
    get_density_matrices, prepare_dipole_along_x
):

    threshold = 1.0e-9
    expected_result = -2.105029271e-03  # x2_exch-ind,r

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind = rho_A["exch-ind"] + rho_B["exch-ind"]

    result = 2 * np.trace(rho_exch_ind @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


def test_exch_disp_s2_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    expected_result = -3.902033407e-04  # x2_exch-disp_S2

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind_s2 = rho_A["exch-disp_S2"] + rho_B["exch-disp_S2"]

    result = 2 * np.trace(rho_exch_ind_s2 @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)


@pytest.mark.xfail(
    reason="the exch-disp S^inf density is not implemented; its property counterpart "
    "x2_exch-disp is itself still under xfail in test_calc_property.py."
)
def test_exch_disp_sinf_from_density_matrix(
    get_density_matrices, prepare_dipole_along_x
):

    threshold = 1.0e-9
    expected_result = 0.0  # NOTE: update when value known

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_disp = rho_A["exch-disp"] + rho_B["exch-disp"]

    result = 2 * np.trace(rho_exch_disp @ d_X)

    assert expected_result == pytest.approx(result, abs=threshold)
