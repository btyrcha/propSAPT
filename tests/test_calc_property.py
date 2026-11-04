import numpy as np
import psi4
import pytest

from .he_h2 import HE_H2, OPTIONS
from prop_sapt import Dimer, calc_property, properties
from prop_sapt.properties import calc_induced_dipole, perform_property_contractions

from .matched_integrals import use_integrals

# Interaction-induced dipole along X for the He-H2 dimer, in a.u., as produced by
# the analytic path. Regression values: the calculation is deterministic (two runs
# agree bit-for-bit) and stable to ~1e-14 across the option sets used by tests/ and
# examples/, so the threshold below is a real constraint rather than a formality.
EXPECTED_X = {
    "x1_pol,r": -3.695584988e-04,
    "x1_exch,r_S2": -1.579072727e-02,
    "x1_exch,r": -1.581693343e-02,
    "x2_ind": +9.357202287e-04,
    "x2_ind,r": +1.042482216e-03,
    "x2_exch-ind,r_S2": -2.095951859e-03,
    "x2_exch-ind,r": -2.105029271e-03,
    "x2_exch-ind_S2": -1.590278890e-03,
    "x2_exch-ind": -1.597539343e-03,
    "x2_disp": +2.195424318e-03,
    "x2_exch-disp_S2": -3.902033407e-04,
    "x2_exch-disp": -4.623252849e-04,
    "x_induced_S2": -1.543474060e-02,
}

# The same quantity from four-point finite field SAPT under this directory's fixture
# options, with the energy path's J/K built from the DF_BASIS_SAPT tensors that the
# analytic contractions use ("matched" integrals, tests/matched_integrals.py).
# Differentiating the SAPT energy numerically and evaluating the analytic derivative
# on the same integrals must give the same number, so these are the values the
# contractions are aiming at. Produced with propSAPT's internal finite-field
# validation harness.
FF_REFERENCE_MATCHED_X = {
    "ELST1": -3.699548326e-04,
    "EXCH1(S^2)": -1.579092780e-02,
    "EXCH1": -1.581713451e-02,
    "IND2": +9.357350916e-04,
    "IND2,R": +1.042537304e-03,
    "EXCH-IND2,R(S^2)": -2.096263470e-03,
    "EXCH-IND2,R": -2.105341740e-03,
    "EXCH-IND2(S^2)": -1.590516803e-03,
    "EXCH-IND2": -1.597777875e-03,
    "DISP2": +2.195424362e-03,
    "EXCH-DISP2(S^2)": -3.902403715e-04,
    "EXCH-DISP2": -3.907828526e-04,
}

# The finite field reference for propSAPT as shipped, where the energies take their J/K
# from Psi4's JK object (DirectJK under these options, so exact). Only the first-order
# terms are kept: their exchange potentials are built with the same JK object, while
# the second-order contractions still take some J/K-type integrals from DF_BASIS_SAPT.
# Same command as above, with `--integrals production`.
FF_REFERENCE_X = {
    "ELST1": -3.695584514e-04,
    "EXCH1(S^2)": -1.579072807e-02,
    "EXCH1": -1.581693423e-02,
}

# Agreement expected between the analytic and the finite field value of each term on
# matched integrals, as (analytic key, finite field key, tolerance in a.u.).
#
# Both paths run on the same integrals, so every pair is expected to agree: the analytic
# contractions are the field derivative of the energy expressions, and a deviation here is
# them failing to be that. The floor is the SCF and CPSCF, which keep Psi4's own integrals
# while the second-order contractions see them through the fitting tensors (see
# tests/matched_integrals.py), hence 1e-9 at first order and 1e-8 at second.
FF_AGREEMENT = [
    # first-order electrostatics; measured deviation 5.0e-11
    ("x1_pol,r", "ELST1", 1.0e-9),
    # first-order exchange; measured deviation 8.3e-10 for both flavours, set by the SCF
    ("x1_exch,r_S2", "EXCH1(S^2)", 1.0e-9),
    ("x1_exch,r", "EXCH1", 1.0e-9),
    # second-order induction; measured deviations 1.8e-12, 7.5e-11, 2.0e-11 and 2.0e-11
    ("x2_ind", "IND2", 1.0e-8),
    ("x2_ind,r", "IND2,R", 1.0e-8),
    ("x2_exch-ind,r_S2", "EXCH-IND2,R(S^2)", 1.0e-8),
    ("x2_exch-ind,r", "EXCH-IND2,R", 1.0e-8),
    # second-order exchange-induction with the uncoupled amplitudes (response=False);
    # measured deviations 1.2e-11 for both flavours
    ("x2_exch-ind_S2", "EXCH-IND2(S^2)", 1.0e-8),
    ("x2_exch-ind", "EXCH-IND2", 1.0e-8),
    # second-order dispersion; measured deviations 4.4e-11 and 4.3e-11
    ("x2_disp", "DISP2", 1.0e-8),
    ("x2_exch-disp_S2", "EXCH-DISP2(S^2)", 1.0e-8),
    # the one term whose analytic expression is not the derivative of its energy: the S^inf
    # exchange-dispersion property has not had the treatment its S^2 counterpart got
    pytest.param(
        "x2_exch-disp",
        "EXCH-DISP2",
        1.0e-8,
        marks=pytest.mark.xfail(
            reason="analytic x2_exch-disp deviates from the finite field derivative by "
            "7.2e-5 a.u.; remove the mark once calc_exch_disp2_sinf_property is the "
            "derivative of calc_exch_disp2_sinf_energy",
            strict=True,
        ),
    ),
]

# The first-order terms again, on propSAPT as shipped. Before the exchange potentials
# were moved onto the JK object, first-order exchange missed here by 6.1e-8 a.u., the
# fitting error of the property.
SHIPPED_AGREEMENT = [
    # measured deviation 4.7e-11
    ("x1_pol,r", "ELST1", 1.0e-9),
    # measured deviation 8.0e-10 for both flavours
    ("x1_exch,r_S2", "EXCH1(S^2)", 1.0e-9),
    ("x1_exch,r", "EXCH1", 1.0e-9),
]

# second-order induction columns that only one kind of induction amplitudes gives:
# the coupled (CPSCF) ones for response=True, the uncoupled ones for response=False
COUPLED_ONLY = {
    "x2_ind,r_A",
    "x2_ind,r_B",
    "x2_ind,r",
    "x2_exch-ind,r_S2",
    "x2_exch-ind,r",
}
UNCOUPLED_ONLY = {"x2_exch-ind_S2", "x2_exch-ind"}


@pytest.fixture(scope="module")
def dipole_results(prepare_dimer):
    """Interaction-induced dipole moment along X, computed once for this module, with
    both kinds of induction amplitudes."""

    return calc_property(prepare_dimer, "dipole", response="both").loc["X"]


@pytest.fixture(scope="module")
def matched_dipole_results(prepare_dimer):
    """The same dipole with both paths on the DF_BASIS_SAPT integrals.

    It needs a Dimer of its own: the matched integrals change J_A and J_B, and every
    omega built from them, when the Dimer is constructed. `prepare_dimer` is requested
    for the Psi4 memory and output file it sets up, and the options are set again
    because other test modules change them.
    """

    psi4.set_options(OPTIONS)
    with use_integrals("matched"):
        return calc_property(Dimer(HE_H2), "dipole", response="both").loc["X"]


def test_calc_dipole(dipole_results):
    """Every dipole contribution reproduces its stored value."""

    threshold = 1.0e-8

    for key, value in EXPECTED_X.items():
        assert value == pytest.approx(dipole_results[key], abs=threshold)


PROPERTY_COMPONENTS = (
    "calc_exch_ind2_resp_s2_property",
    "calc_exch_ind2_resp_sinf_property",
    "calc_exch_disp2_s2_property",
    "calc_exch_disp2_sinf_property",
)


def test_calc_property_from_matrix(prepare_dimer, tmp_path, monkeypatch):
    """An AO matrix gives the same terms as the "dipole" shortcut along that axis, as
    one row of floats. The property components return floats, as the energy ones do."""

    threshold = 1.0e-8
    d_X = np.asarray(prepare_dimer.mints.ao_dipole()[0])

    returned = {}
    for name in PROPERTY_COMPONENTS:

        def spy(*args, _component=getattr(properties, name), _name=name, **kwargs):
            value = _component(*args, **kwargs)
            returned[_name] = type(value)
            return value

        monkeypatch.setattr(properties, name, spy)

    results = calc_property(prepare_dimer, d_X, results=str(tmp_path / "results.csv"))

    assert returned == dict.fromkeys(PROPERTY_COMPONENTS, float)
    assert results.shape[0] == 1
    assert (results.dtypes == "float64").all()

    results = results.iloc[0]

    # the default is the coupled induction amplitudes, with the columns it always had
    assert UNCOUPLED_ONLY.isdisjoint(results.index)
    for key, value in EXPECTED_X.items():
        if key not in UNCOUPLED_ONLY:
            assert value == pytest.approx(results[key], abs=threshold)


def test_uncoupled_property(prepare_dimer, tmp_path, monkeypatch):
    """response=False gives the terms with the uncoupled induction amplitudes only.

    No CPSCF equations are solved for those amplitudes or their derivatives: the only
    solves left are the responses of the two monomers to the property operator.
    """

    threshold = 1.0e-8
    d_X = np.asarray(prepare_dimer.mints.ao_dipole()[0])

    def no_cpscf_amplitudes():
        raise AssertionError("response=False asked for the CPSCF induction amplitudes")

    solves = []
    cpscf_batch = prepare_dimer.cpscf_batch

    def counted_cpscf_batch(monomer, perturbations, **kwargs):
        solves.extend([monomer] * len(perturbations))
        return cpscf_batch(monomer, perturbations, **kwargs)

    monkeypatch.setattr(prepare_dimer, "get_cpscf_ra", no_cpscf_amplitudes)
    monkeypatch.setattr(prepare_dimer, "get_cpscf_sb", no_cpscf_amplitudes)
    monkeypatch.setattr(prepare_dimer, "cpscf_batch", counted_cpscf_batch)

    results = calc_property(
        prepare_dimer, d_X, response=False, results=str(tmp_path / "results.csv")
    )
    results = results.iloc[0]

    assert sorted(solves) == ["A", "B"]
    assert COUPLED_ONLY.isdisjoint(results.index)
    assert UNCOUPLED_ONLY <= set(results.index)

    # x_induced_S2 is stored for the default, coupled induction
    for key, value in EXPECTED_X.items():
        if key not in COUPLED_ONLY and key != "x_induced_S2":
            assert value == pytest.approx(results[key], abs=threshold)

    # the totals are made of the uncoupled induction terms
    rest = (
        results["x1_pol,r"]
        + results["x1_exch,r"]
        + results["x2_ind"]
        + results["x2_disp"]
    )
    assert results["x_induced"] == pytest.approx(
        rest + results["x2_exch-ind"] + results["x2_exch-disp"], abs=1.0e-15
    )
    assert results["x_induced_S2"] == pytest.approx(
        rest + results["x2_exch-ind_S2"] + results["x2_exch-disp_S2"], abs=1.0e-15
    )


@pytest.mark.parametrize(
    "response, expected",
    [
        (True, [("A", 3), ("B", 3), ("A", 3), ("B", 3)]),
        (False, [("A", 3), ("B", 3)]),
    ],
)
def test_dipole_batches_cpscf(prepare_dimer, monkeypatch, response, expected):
    """The dipole solves the CPSCF equations of its three axes together: the responses
    of each monomer to the dipole operators, then, with the coupled induction
    amplitudes, the derivatives of those amplitudes."""

    prepare_dimer.get_cpscf_ra()
    prepare_dimer.get_cpscf_sb()

    batches = []
    cpscf_batch = prepare_dimer.cpscf_batch

    def counted_cpscf_batch(monomer, perturbations, **kwargs):
        batches.append((monomer, len(perturbations)))
        return cpscf_batch(monomer, perturbations, **kwargs)

    monkeypatch.setattr(prepare_dimer, "cpscf_batch", counted_cpscf_batch)

    calc_induced_dipole(prepare_dimer, response=response)

    assert batches == expected


def test_batched_dipole_matches_single_axes(prepare_dimer):
    """Every term of every axis is what the axis alone gives."""

    batched = calc_induced_dipole(prepare_dimer, response="both")
    for axis, d_i in zip("XYZ", prepare_dimer.mints.ao_dipole()):
        single = perform_property_contractions(
            prepare_dimer, np.asarray(d_i), response="both"
        ).iloc[0]
        for key, value in single.items():
            assert batched.loc[axis, key] == pytest.approx(value, rel=0, abs=1e-12), (
                axis,
                key,
            )


def test_calc_property_invalid_response(prepare_dimer):
    """A response other than True, False or "both" is rejected."""

    with pytest.raises(ValueError, match="response"):
        calc_property(prepare_dimer, "dipole", response="yes")


def test_calc_property_unknown_name(prepare_dimer):
    """A property name without an implementation is rejected."""

    with pytest.raises(ValueError, match="not implemented"):
        calc_property(prepare_dimer, "quadrupole")


@pytest.mark.parametrize("prop_key, ff_key, tolerance", FF_AGREEMENT)
def test_dipole_matches_finite_field(
    matched_dipole_results, prop_key, ff_key, tolerance
):
    """The analytic derivative agrees with the finite field one term by term.

    Both paths run on the same integrals, so a deviation here is the contractions
    failing to be the derivative of the energy expressions and nothing else.

    S^inf exchange-induction is included; S^inf exchange-dispersion is the one term
    still expected to fail, and carries an xfail saying so.
    """

    assert matched_dipole_results[prop_key] == pytest.approx(
        FF_REFERENCE_MATCHED_X[ff_key], abs=tolerance
    )


@pytest.mark.parametrize("prop_key, ff_key, tolerance", SHIPPED_AGREEMENT)
def test_shipped_dipole_matches_finite_field(
    dipole_results, prop_key, ff_key, tolerance
):
    """The property propSAPT reports is the derivative of the energy it reports."""

    assert dipole_results[prop_key] == pytest.approx(
        FF_REFERENCE_X[ff_key], abs=tolerance
    )
