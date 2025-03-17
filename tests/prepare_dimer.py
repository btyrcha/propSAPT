import pytest

import psi4

from interaction_induced import Dimer


@pytest.fixture(scope="session")
def prepare_dimer():
    """Prepare dimer object with the He-H2 geometry."""

    geo = """
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
"""

    psi4.set_memory("2 GB")
    psi4.core.set_output_file("output.dat", False)
    psi4.set_options(
        {
            "basis": "aug-cc-pvtz",
            "DF_BASIS_SAPT": "aug-cc-pvqz-ri",
            "scf_type": "direct",
            "e_convergence": 1.0e-10,
            "d_convergence": 1.0e-10,
            "save_jk": True,  # necessary option
        }
    )

    return Dimer(geo)
