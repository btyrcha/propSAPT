"""Fixtures shared by the test modules."""

import psi4
import pytest

from prop_sapt import Dimer

from .he_h2 import HE_H2, OPTIONS


def _apply_he_h2_settings(append: bool):
    psi4.set_memory("2 GB")
    psi4.core.set_output_file("output.dat", append)
    psi4.set_options(OPTIONS)


@pytest.fixture(scope="session")
def prepare_dimer():
    """The He-H2 dimer, built once per test session."""

    _apply_he_h2_settings(append=False)

    return Dimer(HE_H2)


@pytest.fixture
def he_h2_settings():
    """Applies the Psi4 settings of `prepare_dimer`, for tests that build their own
    He-H2 dimers: the shared one is built only once, so other test modules may have
    changed the settings since."""

    _apply_he_h2_settings(append=True)
