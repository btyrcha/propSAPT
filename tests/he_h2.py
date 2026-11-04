"""The He-H2 test system: the geometry and Psi4 options of the shared `prepare_dimer`
fixture (`conftest.py`)."""

HE_H2 = """
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

OPTIONS = {
    "basis": "aug-cc-pvtz",
    "DF_BASIS_SAPT": "aug-cc-pvqz-ri",
    "scf_type": "direct",
    "e_convergence": 1.0e-10,
    "d_convergence": 1.0e-10,
}
