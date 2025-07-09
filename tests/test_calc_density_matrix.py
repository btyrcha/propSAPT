import pytest

import numpy as np

from prepare_dimer import prepare_dimer
from prop_sapt import Dimer, calc_density_matirx


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

    rho_A = calc_density_matirx(prepare_dimer, "A", orbital_basis="AO")
    rho_B = calc_density_matirx(prepare_dimer, "B", orbital_basis="AO")

    return rho_A, rho_B


def test_induction_from_density_matrix(get_density_matrices, prepare_dipole_along_x):

    threshold = 1.0e-9
    excpected_result = 0.0010501236802531397

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
