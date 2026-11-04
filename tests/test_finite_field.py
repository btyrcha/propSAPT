"""Minimal tests for finite_field module - core functionality only."""

import math

import pandas as pd
import pytest
from unittest.mock import Mock, patch

from prop_sapt.finite_field import (
    finite_field_property,
    calc_ff_dipole,
    calculate_ff_dipole_along_axis,
    finite_field_sapt,
    calc_ff_sapt_dipole,
    calculate_ff_sapt_dipole_along_axis,
)
from prop_sapt.finite_field.utils import (
    get_field_vector,
    calculate_perturbed_energy,
    calculate_perturbed_sapt_energies,
    apply_finite_difference_formula,
    build_result_series,
    print_finite_field_banner,
)


class TestFieldVectorGeneration:
    """Test field vector generation."""

    def test_get_field_vector_all_axes(self):
        """Test field vector generation for all axes."""
        for axis, expected in [
            ("X", [0.001, 0.0, 0.0]),
            ("Y", [0.0, 0.001, 0.0]),
            ("Z", [0.0, 0.0, 0.001]),
        ]:
            assert get_field_vector(axis, 0.001) == expected

    def test_get_field_vector_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        with pytest.raises(ValueError, match="Invalid axis"):
            get_field_vector("invalid", 0.001)


class TestPerturbedEnergyCalculation:
    """Test perturbed energy calculations."""

    @patch("psi4.energy")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_calculate_perturbed_energy_returns_energy(
        self, mock_revoke, mock_set_options, mock_print, mock_energy
    ):
        """Test that calculate_perturbed_energy returns energy value."""
        mock_molecule = Mock()
        mock_energy.return_value = 1.005

        result = calculate_perturbed_energy(
            method="HF",
            molecule=mock_molecule,
            field_vector=[0.001, 0.0, 0.0],
            axis="X",
        )

        assert result == pytest.approx(1.005)
        mock_energy.assert_called_once()
        mock_revoke.assert_any_call("PERTURB_H")
        mock_revoke.assert_any_call("PERTURB_DIPOLE")


class TestFiniteDifferenceFormulas:
    """Test finite difference formula calculations."""

    @patch("psi4.core.print_out")
    def test_two_point_formula(self, mock_print):
        """Test two-point finite difference formula."""
        energies = {"plus_h": 1.005, "minus_h": 0.995}
        result = apply_finite_difference_formula(energies, "two_point", 0.001)
        # (1.005 - 0.995) / (2 * 0.001) = 5.0
        assert result == pytest.approx(5.0)

    @patch("psi4.core.print_out")
    def test_four_point_formula(self, mock_print):
        """Test four-point finite difference formula."""
        energies = {
            "plus_2h": 1.010,
            "plus_h": 1.005,
            "minus_h": 0.995,
            "minus_2h": 0.990,
        }
        result = apply_finite_difference_formula(energies, "four_point", 0.001)
        # (-1.010 + 8*1.005 - 8*0.995 + 0.990) / (12 * 0.001) = 5.0
        assert result == pytest.approx(5.0)

    @patch("psi4.core.print_out")
    def test_formula_missing_energies_raises_error(self, mock_print):
        """Test that missing energies raises ValueError."""
        with pytest.raises(ValueError):
            apply_finite_difference_formula({"plus_h": 1.005}, "two_point", 0.001)


class TestResultSeriesConstruction:
    """Test result series building."""

    def test_build_result_series_two_point(self):
        """Test build_result_series for two-point formula."""
        energies = {"plus_h": 1.005, "minus_h": 0.995}
        result = build_result_series(
            derivative=5.0,
            label="HF",
            energies=energies,
            formula="two_point",
            field_strength=0.001,
        )

        assert result["HF"] == 5.0
        assert len(result) == 3
        assert "HF_+0.001" in result.index

    def test_build_result_series_four_point(self):
        """Test build_result_series for four-point formula."""
        energies = {
            "plus_2h": 1.010,
            "plus_h": 1.005,
            "minus_h": 0.995,
            "minus_2h": 0.990,
        }
        result = build_result_series(
            derivative=5.0,
            label="DFT",
            energies=energies,
            formula="four_point",
            field_strength=0.001,
        )

        assert result["DFT"] == 5.0
        assert len(result) == 5


class TestBannerPrinting:
    """Test banner printing."""

    @patch("psi4.core.print_out")
    def test_print_finite_field_banner(self, mock_print):
        """Test print_finite_field_banner calls psi4.core.print_out."""
        print_finite_field_banner(
            "TEST BANNER",
            {"Method": "HF", "Field strength": 0.001},
        )

        assert mock_print.call_count >= 5
        calls_str = str(mock_print.call_args_list)
        assert "TEST BANNER" in calls_str


class TestDipoleTwoPointFormula:
    """Test two-point finite field dipole calculation."""

    @patch("psi4.geometry")
    @patch("psi4.energy")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_calculate_ff_dipole_two_point(
        self, mock_revoke, mock_set_options, mock_print, mock_energy, mock_geometry
    ):
        """Test two-point dipole calculation."""
        mock_molecule = Mock()
        mock_geometry.return_value = mock_molecule
        mock_energy.side_effect = [1.005, 0.995]

        result = calculate_ff_dipole_along_axis(
            geometry="dummy",
            axis="X",
            method="HF",
            field_strength=0.001,
            formula="two_point",
        )

        assert result["HF"] == pytest.approx(5.0)
        assert len(result) == 3
        assert mock_energy.call_count == 2


class TestDipoleFourPointFormula:
    """Test four-point finite field dipole calculation."""

    @patch("psi4.geometry")
    @patch("psi4.energy")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_calculate_ff_dipole_four_point(
        self, mock_revoke, mock_set_options, mock_print, mock_energy, mock_geometry
    ):
        """Test four-point dipole calculation."""
        mock_molecule = Mock()
        mock_geometry.return_value = mock_molecule
        mock_energy.side_effect = [1.010, 1.005, 0.995, 0.990]

        result = calculate_ff_dipole_along_axis(
            geometry="dummy",
            axis="X",
            method="HF",
            field_strength=0.001,
            formula="four_point",
        )

        assert result["HF"] == pytest.approx(5.0)
        assert len(result) == 5
        assert mock_energy.call_count == 4


class TestDipoleWithCPCorrection:
    """Test dipole calculation with CP correction."""

    @patch("psi4.geometry")
    @patch("psi4.energy")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_calculate_ff_dipole_with_cp_correction(
        self, mock_revoke, mock_set_options, mock_print, mock_energy, mock_geometry
    ):
        """Test that CP correction is passed to psi4.energy."""
        mock_molecule = Mock()
        mock_geometry.return_value = mock_molecule
        mock_energy.side_effect = [1.005, 0.995]

        calculate_ff_dipole_along_axis(
            geometry="dummy",
            axis="X",
            method="HF",
            cp_correction=True,
            field_strength=0.001,
            formula="two_point",
        )

        # Verify bsse_type was passed to psi4.energy
        calls = mock_energy.call_args_list
        assert any("bsse_type" in str(call) for call in calls)


class TestFiniteFieldPropertyValidation:
    """Test finite_field_property function."""

    @patch("psi4.core.print_out")
    @patch("psi4.core.clean")
    def test_finite_field_property_invalid_property(self, mock_clean, mock_print):
        """Test that invalid property raises ValueError."""
        with pytest.raises(ValueError, match="not implemented"):
            finite_field_property(
                geometry="dummy",
                method="HF",
                prop="invalid_property",
                field_strength=0.001,
            )


class TestFiniteFieldSAPT:
    """Test finite_field_sapt function."""

    @patch("psi4.core.print_out")
    @patch("psi4.core.clean")
    def test_finite_field_sapt_invalid_property(self, mock_clean, mock_print):
        """Test that invalid property raises ValueError."""
        with pytest.raises(ValueError, match="not implemented"):
            finite_field_sapt(
                geometry="dummy",
                prop="invalid_property",
                field_strength=0.001,
            )


class TestFiniteDifferenceFormulaOnSeries:
    """Test finite difference formulas applied to pd.Series (SAPT decomposition)."""

    @patch("psi4.core.print_out")
    def test_two_point_formula_on_series(self, mock_print):
        """Two-point formula differentiates each SAPT term independently."""
        energies = {
            "plus_h": pd.Series({"ELST1": 1.005, "EXCH1": 2.010}),
            "minus_h": pd.Series({"ELST1": 0.995, "EXCH1": 1.990}),
        }
        result = apply_finite_difference_formula(energies, "two_point", 0.001)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["ELST1", "EXCH1"]
        assert result["ELST1"] == pytest.approx(5.0)
        assert result["EXCH1"] == pytest.approx(10.0)

    @patch("psi4.core.print_out")
    def test_four_point_formula_on_series(self, mock_print):
        """Four-point formula differentiates each SAPT term independently."""
        energies = {
            "plus_2h": pd.Series({"ELST1": 1.010, "EXCH1": 2.020}),
            "plus_h": pd.Series({"ELST1": 1.005, "EXCH1": 2.010}),
            "minus_h": pd.Series({"ELST1": 0.995, "EXCH1": 1.990}),
            "minus_2h": pd.Series({"ELST1": 0.990, "EXCH1": 1.980}),
        }
        result = apply_finite_difference_formula(energies, "four_point", 0.001)

        assert isinstance(result, pd.Series)
        assert list(result.index) == ["ELST1", "EXCH1"]
        assert result["ELST1"] == pytest.approx(5.0)
        assert result["EXCH1"] == pytest.approx(10.0)

    @patch("psi4.core.print_out")
    def test_four_point_is_higher_order_than_two_point(self, mock_print):
        """Four-point must converge as h^4 where two-point converges as h^2.

        Uses E(F) = exp(F), whose exact derivative at F = 0 is 1.
        """
        # h must stay well above the roundoff floor of the four-point stencil,
        # which is ~1e-16 / (12 h); at h = 1e-3 it already pollutes the h^4 ratio.
        errors = {}
        for h in (1e-1, 1e-2):
            energies = {
                "plus_2h": math.exp(2 * h),
                "plus_h": math.exp(h),
                "minus_h": math.exp(-h),
                "minus_2h": math.exp(-2 * h),
            }
            errors[h] = {
                formula: abs(
                    apply_finite_difference_formula(energies, formula, h) - 1.0
                )
                for formula in ("two_point", "four_point")
            }

        # At a given step size, four-point is far more accurate
        for h in errors:
            assert errors[h]["four_point"] < errors[h]["two_point"]

        # Shrinking h by 10x reduces the error by ~10^2 and ~10^4 respectively
        assert errors[1e-1]["two_point"] / errors[1e-2]["two_point"] == pytest.approx(
            1e2, rel=0.05
        )
        assert errors[1e-1]["four_point"] / errors[1e-2]["four_point"] == pytest.approx(
            1e4, rel=0.05
        )


class TestPerturbedSAPTEnergies:
    """Test the SAPT energy helper used by the finite field SAPT driver."""

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_perturbation_is_applied_and_revoked(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """The field is set before the SAPT calculation and revoked after."""
        mock_sapt.return_value = pd.Series({"ELST1": 1.0})

        result = calculate_perturbed_sapt_energies(
            geometry="dummy",
            field_vector=[0.001, 0.0, 0.0],
            axis="X",
        )

        options = mock_set_options.call_args[0][0]
        assert options["PERTURB_H"] is True
        assert options["PERTURB_DIPOLE"] == [0.001, 0.0, 0.0]
        mock_revoke.assert_any_call("PERTURB_H")
        mock_revoke.assert_any_call("PERTURB_DIPOLE")
        assert result["ELST1"] == 1.0

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_opposite_fields_use_distinct_results_files(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """+h and -h must not write their SAPT results to the same CSV."""
        mock_sapt.return_value = pd.Series({"ELST1": 1.0})

        filenames = []
        for magnitude in (0.001, -0.001, 0.002, -0.002):
            calculate_perturbed_sapt_energies(
                geometry="dummy",
                field_vector=[magnitude, 0.0, 0.0],
                axis="X",
            )
            filenames.append(mock_sapt.call_args.kwargs["results"])

        assert len(set(filenames)) == 4, f"colliding results files: {filenames}"

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_results_filename_reports_applied_field(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """The results filename must name the field strength that was applied."""
        mock_sapt.return_value = pd.Series({"ELST1": 1.0})

        calculate_perturbed_sapt_energies(
            geometry="dummy",
            field_vector=[0.001, 0.0, 0.0],
            axis="X",
        )

        assert "0.001" in mock_sapt.call_args.kwargs["results"]


def _fake_sapt_from_field(mock_set_options, terms):
    """Build a calc_sapt_energy stub returning an analytic function of the applied field.

    Each SAPT term n is modelled as E_n(F) = n * exp(F), so the exact derivative
    dE_n/dF at F = 0 is n. The field is read back from the psi4.set_options call
    that the helper just made, so this also verifies that each finite difference
    dictionary key receives the energy computed at the matching field strength.
    """

    def fake_sapt(dimer, **kwargs):
        field_vector = mock_set_options.call_args[0][0]["PERTURB_DIPOLE"]
        field = sum(field_vector)
        return pd.Series(
            {term: (n + 1) * math.exp(field) for n, term in enumerate(terms)}
        )

    return fake_sapt


class TestSAPTDipoleFourPointFormula:
    """Test four-point finite field SAPT dipole calculation."""

    TERMS = ["ELST1", "EXCH1", "IND2,R"]

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_four_point_runs_four_sapt_calculations(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """Four-point requires SAPT at -2h, -h, +h and +2h."""
        mock_sapt.side_effect = _fake_sapt_from_field(mock_set_options, self.TERMS)

        calculate_ff_sapt_dipole_along_axis(
            geometry="dummy",
            axis="X",
            field_strength=0.001,
            formula="four_point",
        )

        assert mock_sapt.call_count == 4

        applied_fields = sorted(
            call[0][0]["PERTURB_DIPOLE"][0] for call in mock_set_options.call_args_list
        )
        assert applied_fields == pytest.approx([-0.002, -0.001, 0.001, 0.002])

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_two_point_runs_two_sapt_calculations(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """Two-point requires SAPT at -h and +h only."""
        mock_sapt.side_effect = _fake_sapt_from_field(mock_set_options, self.TERMS)

        calculate_ff_sapt_dipole_along_axis(
            geometry="dummy",
            axis="X",
            field_strength=0.001,
            formula="two_point",
        )

        assert mock_sapt.call_count == 2

        applied_fields = sorted(
            call[0][0]["PERTURB_DIPOLE"][0] for call in mock_set_options.call_args_list
        )
        assert applied_fields == pytest.approx([-0.001, 0.001])

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_four_point_differentiates_every_sapt_term(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """Every SAPT term is differentiated, and the decomposition index survives.

        With E_n(F) = n * exp(F) the exact derivatives are 1, 2, 3; the four-point
        truncation error at h = 1e-3 is O(h^4) ~ 1e-12.
        """
        mock_sapt.side_effect = _fake_sapt_from_field(mock_set_options, self.TERMS)

        result = calculate_ff_sapt_dipole_along_axis(
            geometry="dummy",
            axis="X",
            field_strength=0.001,
            formula="four_point",
        )

        assert isinstance(result, pd.Series)
        assert list(result.index) == self.TERMS
        assert result["ELST1"] == pytest.approx(1.0, abs=1e-9)
        assert result["EXCH1"] == pytest.approx(2.0, abs=1e-9)
        assert result["IND2,R"] == pytest.approx(3.0, abs=1e-9)

    @patch("prop_sapt.finite_field.utils.calc_sapt_energy")
    @patch("prop_sapt.finite_field.utils.Dimer")
    @patch("psi4.core.print_out")
    @patch("psi4.set_options")
    @patch("psi4.core.revoke_global_option_changed")
    def test_four_point_beats_two_point_on_same_model(
        self, mock_revoke, mock_set_options, mock_print, mock_dimer, mock_sapt
    ):
        """On an identical model the four-point derivative is closer to exact."""
        derivatives = {}
        for formula in ("two_point", "four_point"):
            mock_set_options.reset_mock()
            mock_sapt.side_effect = _fake_sapt_from_field(mock_set_options, self.TERMS)
            derivatives[formula] = calculate_ff_sapt_dipole_along_axis(
                geometry="dummy",
                axis="X",
                field_strength=0.01,
                formula=formula,
            )

        two_point_error = abs(derivatives["two_point"]["ELST1"] - 1.0)
        four_point_error = abs(derivatives["four_point"]["ELST1"] - 1.0)

        assert four_point_error < two_point_error

    @patch("psi4.core.print_out")
    @patch("psi4.core.clean")
    def test_invalid_formula_raises_before_any_calculation(
        self, mock_clean, mock_print
    ):
        """An unknown formula is rejected before any expensive SAPT run starts."""
        with pytest.raises(ValueError, match="Unknown finite difference formula"):
            calculate_ff_sapt_dipole_along_axis(
                geometry="dummy",
                axis="X",
                field_strength=0.001,
                formula="three_point",
            )
