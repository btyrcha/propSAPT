"""Density components reproduce their property terms on a dimer whose monomers have
several occupied orbitals.

The He-H2 tests cannot see a wrong sum over occupied orbitals: He and H2 have one
occupied orbital each, so every such sum has a single term.
"""

import numpy as np
import psi4
import pytest

from .he_h2 import OPTIONS
from prop_sapt import Dimer, calc_density_matrix, calc_property

# water dimer as in examples/water_dimer.py, five occupied orbitals per monomer
WATER_DIMER = """
symmetry c1
no_com
no_reorient
--
0 1
O   -1.41606479    0.00000000    0.00000000
H   -0.52004811    0.00000000   -0.08624091
H   -1.81884655    0.00000000   -0.89989100
--
0 1
O    1.41606479    0.00000000    0.00000000
H    1.71179607   -0.76501221    0.50083629
H    1.71179607    0.76501221    0.50083629
"""

# each density component and the property term it reproduces
PROPERTY_TERMS = {
    "pol": "x1_pol,r",
    "exch": "x1_exch,r",
    "ind": "x2_ind,r",
    "exch-ind_S2": "x2_exch-ind,r_S2",
    "exch-ind": "x2_exch-ind,r",
    "disp": "x2_disp",
    "exch-disp_S2": "x2_exch-disp_S2",
}


@pytest.fixture(scope="module")
def water_dimer(prepare_dimer):
    """Water dimer in cc-pVDZ. `prepare_dimer` is requested for the Psi4 memory and
    output file it sets up; its options are restored for the test modules after this
    one."""

    psi4.set_options({**OPTIONS, "basis": "cc-pvdz", "DF_BASIS_SAPT": "cc-pvdz-ri"})

    yield Dimer(WATER_DIMER)

    psi4.set_options(OPTIONS)


@pytest.fixture(scope="module")
def operator(water_dimer):
    """A random symmetric AO matrix. The property terms are linear in the operator, so
    this probes every element of the density matrices at once."""

    nbf = np.asarray(water_dimer.mints.ao_overlap()).shape[0]
    matrix = np.random.default_rng(2026).standard_normal((nbf, nbf))

    return matrix + matrix.T


@pytest.fixture(scope="module")
def property_terms(water_dimer, operator, tmp_path_factory):
    """Analytic property terms of the operator."""

    results = tmp_path_factory.mktemp("water") / "results.csv"

    return calc_property(water_dimer, operator, results=str(results)).iloc[0]


@pytest.fixture(scope="module")
def density_matrices(water_dimer):
    """AO density matrices of both monomers."""

    return {
        monomer: calc_density_matrix(water_dimer, monomer, orbital_basis="AO")
        for monomer in ("A", "B")
    }


@pytest.mark.parametrize("component, term", PROPERTY_TERMS.items())
def test_density_reproduces_property(
    density_matrices, property_terms, operator, component, term
):
    """The trace of a density component with the operator is its property term."""

    rho = density_matrices["A"][component] + density_matrices["B"][component]

    assert 2 * np.trace(rho @ operator) == pytest.approx(
        property_terms[term], abs=1.0e-9
    )
