import pytest

import pandas as pd

from prepare_dimer import prepare_dimer
from prop_sapt.sapt_driver import calc_sapt_energy
from prop_sapt.sapt_components.ind2 import calc_ind2_r_energy

# Common threshold for all SAPT component tests
THRESHOLD = 1.0e-8

# Expected results for SAPT components calculated with Psi4 SAPT(DFT) code
EXPECTED_RESULTS = {
    "ELST1": -0.21265981e-3,
    "EXCH1": 1.19144311e-3,
    "EXCH1(S^2)": 1.19038145e-3,
    "IND2_A": -0.01855105e-3,
    "IND2_B": -0.05165709e-3,
    "IND2": -0.07020814e-3,
    "EXCH-IND2(S^2)_A": 0.00230711e-3,
    "EXCH-IND2(S^2)_B": 0.05013906e-3,
    "EXCH-IND2(S^2)": 0.05244617e-3,
    "EXCH-IND2_A": 0.00235597e-3,
    "EXCH-IND2_B": 0.05031871e-3,
    "EXCH-IND2": 0.05267468e-3,
    "IND2,R_A": -0.02283883e-3,
    "IND2,R_B": -0.05913678e-3,
    "IND2,R": -0.08197561e-3,
    "EXCH-IND2,R(S^2)_A": 0.00321188e-3,
    "EXCH-IND2,R(S^2)_B": 0.06315685e-3,
    "EXCH-IND2,R(S^2)": 0.06636873e-3,
    "EXCH-IND2,R_A": 0.00327278e-3,
    "EXCH-IND2,R_B": 0.06337536e-3,
    "EXCH-IND2,R": 0.06664814e-3,
    "DISP2": -0.433150373e-3,
    "EXCH-DISP2(S^2)": 0.03655862e-3,
    "EXCH-DISP2": 0.036593945e-3,
}


@pytest.fixture(scope="module")
def sapt_results(prepare_dimer):
    """Calculate SAPT energy components once for all tests."""
    dimer = prepare_dimer
    results = calc_sapt_energy(dimer, results=False, response=False)
    response_results = calc_ind2_r_energy(dimer)
    results = pd.concat([results, response_results])

    return results


def test_elst1(sapt_results):
    """Test ELST1 component."""
    expected = EXPECTED_RESULTS["ELST1"]
    assert expected == pytest.approx(sapt_results["ELST1"], abs=THRESHOLD)


def test_exch1(sapt_results):
    """Test EXCH1 component."""
    expected = EXPECTED_RESULTS["EXCH1"]
    assert expected == pytest.approx(sapt_results["EXCH1"], abs=THRESHOLD)


def test_exch1_s2(sapt_results):
    """Test EXCH1(S^2) component."""
    expected = EXPECTED_RESULTS["EXCH1(S^2)"]
    assert expected == pytest.approx(sapt_results["EXCH1(S^2)"], abs=THRESHOLD)


def test_ind2_a(sapt_results):
    """Test IND2_A component."""
    expected = EXPECTED_RESULTS["IND2_A"]
    assert expected == pytest.approx(sapt_results["IND2_A"], abs=THRESHOLD)


def test_ind2_b(sapt_results):
    """Test IND2_B component."""
    expected = EXPECTED_RESULTS["IND2_B"]
    assert expected == pytest.approx(sapt_results["IND2_B"], abs=THRESHOLD)


def test_ind2(sapt_results):
    """Test IND2 component."""
    expected = EXPECTED_RESULTS["IND2"]
    assert expected == pytest.approx(sapt_results["IND2"], abs=THRESHOLD)


def test_ind2_r_a(sapt_results):
    """Test IND2,R_A component."""
    expected = EXPECTED_RESULTS["IND2,R_A"]
    assert expected == pytest.approx(sapt_results["IND2,R_A"], abs=THRESHOLD)


def test_ind2_r_b(sapt_results):
    """Test IND2,R_B component."""
    expected = EXPECTED_RESULTS["IND2,R_B"]
    assert expected == pytest.approx(sapt_results["IND2,R_B"], abs=THRESHOLD)


def test_ind2_r(sapt_results):
    """Test IND2,R component."""
    expected = EXPECTED_RESULTS["IND2,R"]
    assert expected == pytest.approx(sapt_results["IND2,R"], abs=THRESHOLD)


def test_exch_ind2_s2_a(sapt_results):
    """Test EXCH-IND2(S^2)_A component."""
    expected = EXPECTED_RESULTS["EXCH-IND2(S^2)_A"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2(S^2)_A"], abs=THRESHOLD)


def test_exch_ind2_s2_b(sapt_results):
    """Test EXCH-IND2(S^2)_B component."""
    expected = EXPECTED_RESULTS["EXCH-IND2(S^2)_B"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2(S^2)_B"], abs=THRESHOLD)


def test_exch_ind2_s2(sapt_results):
    """Test EXCH-IND2(S^2) component."""
    expected = EXPECTED_RESULTS["EXCH-IND2(S^2)"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2(S^2)"], abs=THRESHOLD)


def test_exch_ind2_a(sapt_results):
    """Test EXCH-IND2_A component."""
    expected = EXPECTED_RESULTS["EXCH-IND2_A"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2_A"], abs=THRESHOLD)


def test_exch_ind2_b(sapt_results):
    """Test EXCH-IND2_B component."""
    expected = EXPECTED_RESULTS["EXCH-IND2_B"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2_B"], abs=THRESHOLD)


def test_exch_ind2(sapt_results):
    """Test EXCH-IND2 component."""
    expected = EXPECTED_RESULTS["EXCH-IND2"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2"], abs=THRESHOLD)


def test_exch_ind2_r_a(sapt_results):
    """Test EXCH-IND2,R_A component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R_A"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R_A"], abs=THRESHOLD)


def test_exch_ind2_r_b(sapt_results):
    """Test EXCH-IND2,R_B component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R_B"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R_B"], abs=THRESHOLD)


def test_exch_ind2_r(sapt_results):
    """Test EXCH-IND2,R component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R"], abs=THRESHOLD)


def test_exch_ind2_r_s2_a(sapt_results):
    """Test EXCH-IND2,R(S^2)_A component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R(S^2)_A"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R(S^2)_A"], abs=THRESHOLD)


def test_exch_ind2_r_s2_b(sapt_results):
    """Test EXCH-IND2,R(S^2)_B component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R(S^2)_B"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R(S^2)_B"], abs=THRESHOLD)


def test_exch_ind2_r_s2(sapt_results):
    """Test EXCH-IND2,R(S^2) component."""
    expected = EXPECTED_RESULTS["EXCH-IND2,R(S^2)"]
    assert expected == pytest.approx(sapt_results["EXCH-IND2,R(S^2)"], abs=THRESHOLD)


def test_disp2(sapt_results):
    """Test DISP2 component."""
    expected = EXPECTED_RESULTS["DISP2"]
    assert expected == pytest.approx(sapt_results["DISP2"], abs=THRESHOLD)


def test_exch_disp2(sapt_results):
    """Test EXCH-DISP2 component."""
    expected = EXPECTED_RESULTS["EXCH-DISP2"]
    assert expected == pytest.approx(sapt_results["EXCH-DISP2"], abs=THRESHOLD)


def test_exch_disp2_s2(sapt_results):
    """Test EXCH-DISP2(S^2) component."""
    expected = EXPECTED_RESULTS["EXCH-DISP2(S^2)"]
    assert expected == pytest.approx(sapt_results["EXCH-DISP2(S^2)"], abs=THRESHOLD)
