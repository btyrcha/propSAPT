import pytest

import numpy as np

from prepare_dimer import prepare_dimer
from prop_sapt import calc_density_matrix


@pytest.fixture(scope="module")
def prepare_dipole_along_x(prepare_dimer):
    """Prepare dipole moment AO matrix along X axis."""

    d_vec = prepare_dimer.mints.ao_dipole()
    d_vec = [np.asarray(elem) for elem in d_vec]
    d_X = d_vec[0]

    return d_X


@pytest.fixture(scope="module")
def get_density_matrices(prepare_dimer):
    """Get interaction-induced density matrices for monomers A and B."""

    rho_A = calc_density_matrix(prepare_dimer, "A", orbital_basis="AO")
    rho_B = calc_density_matrix(prepare_dimer, "B", orbital_basis="AO")

    return rho_A, rho_B


def test_polarization_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = -0.00036955856060192586  # propSAPT calculation

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_pol = rho_A["pol"] + rho_B["pol"]

    result = 2 * np.trace(rho_pol @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_exchange_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = -0.01581687187738625  # propSAPT calculation

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch = rho_A["exch"] + rho_B["exch"]

    result = 2 * np.trace(rho_exch @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_induction_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = 0.0010500679691781473  # propSAPT calculation
    # excpected_result = 0.0010501236802531397  # propSAPT before updating to JK
    # excpected_result = 0.00104248625  # from MOLPRO finite-field SAPT calculation

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_ind = rho_A["ind"] + rho_B["ind"]

    result = 2 * np.trace(rho_ind @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_dispersion_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = 0.0022285332273810207

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_disp = rho_A["disp"] + rho_B["disp"]

    result = 2 * np.trace(rho_disp @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_exch_ind_s2_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = -0.0020955309351026385

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind_s2 = rho_A["exch-ind_S2"] + rho_B["exch-ind_S2"]

    result = 2 * np.trace(rho_exch_ind_s2 @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_exch_ind_sinf_from_density_matrix(
    get_density_matrices, prepare_dipole_along_x
):

    threshold = 1.0e-9
    excpected_result = -0.002014582241925037

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind = rho_A["exch-ind"] + rho_B["exch-ind"]

    result = 2 * np.trace(rho_exch_ind @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


def test_exch_disp_s2_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = -0.00040088163332375364

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_ind_s2 = rho_A["exch-disp_S2"] + rho_B["exch-disp_S2"]

    result = 2 * np.trace(rho_exch_ind_s2 @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)


@pytest.mark.xfail(reason="Exch-disp S^inf density not yet implemented")
def test_exch_disp_sinf_from_density_matrix(
    get_density_matrices, prepare_dipole_along_x
):

    threshold = 1.0e-9
    excpected_result = -0.00034642059973987

    d_X = prepare_dipole_along_x

    rho_A, rho_B = get_density_matrices
    rho_exch_disp = rho_A["exch-disp"] + rho_B["exch-disp"]

    result = 2 * np.trace(rho_exch_disp @ d_X)

    assert excpected_result == pytest.approx(result, abs=threshold)
