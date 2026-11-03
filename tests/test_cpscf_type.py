"""The `cpscf_type` option: the CPSCF solves of the density matrices and the properties
on another J/K object, while the SAPT energies keep the monomer SCF's."""

import pytest

from prop_sapt import get_propsapt_options, reset_propsapt_options, set_propsapt_options
from prop_sapt.options import check_cpscf_type


@pytest.fixture
def options():
    """Resets the propSAPT options after the test."""

    yield
    reset_propsapt_options()


@pytest.mark.parametrize(
    "value, expected", [(None, None), ("df", "DF"), ("Direct", "DIRECT")]
)
def test_cpscf_type_values(options, value, expected):
    """`cpscf_type` takes None, "DIRECT" or "DF" in any case."""

    assert check_cpscf_type(value) == expected

    set_propsapt_options({"cpscf_type": value})

    assert get_propsapt_options()["cpscf_type"] == value


@pytest.mark.parametrize(
    "new_options, match",
    [
        ({"cpscf_type": "PK"}, "cpscf_type"),
        ({"cpscf_type": "mem_df"}, "cpscf_type"),
        ({"cpscf_type": 1}, "cpscf_type"),
        ({"r_convergence": 1e-8, "cpscf_type": "CD"}, "cpscf_type"),
        ({"df_basis_cpscf": ""}, "df_basis_cpscf"),
        ({"df_basis_cpscf": 5}, "df_basis_cpscf"),
    ],
)
def test_invalid_cpscf_options_rejected(options, new_options, match):
    """Invalid values are rejected when they are set, and no option changes."""

    before = get_propsapt_options()

    with pytest.raises(ValueError, match=match):
        set_propsapt_options(new_options)

    assert get_propsapt_options() == before
