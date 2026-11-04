"""
Finite field property calculations and finite field SAPT.
"""

from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
import psi4

from ..utils import CalcTimer
from .utils import (
    get_field_vector,
    calculate_perturbed_energy,
    calculate_perturbed_sapt_energies,
    apply_finite_difference_formula,
    build_result_series,
    print_finite_field_banner,
    apply_to_cartesian_axes,
)

from ..options import PROPSAPT_OPTIONS


def validate_property(prop: str | np.ndarray) -> None:
    """Validate that the requested property is supported.

    Args:
        prop (str | np.ndarray): Property to validate.

    Raises:
        ValueError: If the property is not supported.
    """
    if isinstance(prop, np.ndarray):
        psi4.core.clean()
        raise ValueError("Property for general external operator is not implemented.")

    if prop != "dipole":
        psi4.core.clean()
        raise ValueError(
            f"Property {prop} is not implemented. "
            "Please provide a valid property name or a property matrix."
        )


def finite_field_property(
    geometry: str,
    method: str,
    prop: str | np.ndarray,
    cp_correction: bool = False,
    field_strength: float = 0.001,
    formula: str = "two_point",
    **kwargs,
) -> pd.DataFrame:
    """Calculate the interaction-induced property using finite field method.

    The energies of `method` (anything `psi4.energy` runs, e.g. "scf", a functional,
    "mp2" or "ccsd") are computed in a uniform field along each axis (Psi4's `PERTURB_H`)
    and differentiated, so the result includes the orbital relaxation.

    Args:
        geometry (str): Geometry of the molecule in Psi4 format.
        method (str): Electronic structure method, as named for `psi4.energy`.
        prop (str | np.ndarray): Property to calculate (e.g., "dipole").
        cp_correction (bool, optional): Differentiate the counterpoise-corrected interaction
            energy (`bsse_type="cp"`), giving the interaction-induced property of a dimer.
            Without it the total energy is differentiated; Psi4's field acts on the
            electrons only, so that gives the electronic part of the dipole, which depends
            on the origin. Defaults to False.
        field_strength (float, optional): Strength of the external electric field in atomic units. Defaults to 0.001.
        formula (str, optional): Finite difference formula to use ("two_point" or "four_point"). Defaults to "two_point".

    Keyword Args:
        results (str): Path of the CSV file the results are saved to. Defaults to
            "results-ff-<method>.csv".

    Raises:
        ValueError: If the property is not implemented.
        ValueError: If the calculation fails.

    Returns:
        pd.DataFrame: DataFrame containing the results of the finite field property calculation.
    """

    # Banner
    print_finite_field_banner(
        "FINITE FIELD PROPERTY CALCULATIONS",
        {
            "Method": method,
            "Property": str(prop),
            "Field strength": field_strength,
            "Formula": formula,
        },
    )

    with CalcTimer("Finite field property calculations"):
        ### Validate property
        validate_property(prop)

        ### Results output file
        results_fname = kwargs.get("results", f"results-ff-{method}.csv")

        ### Dipole moment calculations
        results = calc_ff_dipole(
            geometry,
            method=method,
            cp_correction=cp_correction,
            field_strength=field_strength,
            formula=formula,
        )

        ### Results saving to file
        results.to_csv(results_fname)

    return results


def calc_ff_dipole(
    geometry: str,
    method: str,
    cp_correction: bool = False,
    field_strength: float = 0.001,
    formula: str = "two_point",
) -> pd.DataFrame:
    """Calculate the finite field dipole moment.

    Args:
        geometry (str): Geometry of the molecule in Psi4 format.
        method (str): Electronic structure method to use (e.g., "HF", "DFT").
        prop (str | np.ndarray): Property to calculate (e.g., "dipole").
        cp_correction (bool, optional): Whether to apply counterpoise correction. Defaults to False.
        field_strength (float, optional): Strength of the external electric field in atomic units. Defaults to 0.001.
        formula (str, optional): Finite difference formula to use ("two_point" or "four_point"). Defaults to "two_point".
    Returns:
        pd.DataFrame: DataFrame containing the finite field dipole moment results.
    """

    # prepare results
    return apply_to_cartesian_axes(
        calculate_ff_dipole_along_axis,
        geometry,
        method=method,
        cp_correction=cp_correction,
        field_strength=field_strength,
        formula=formula,
    )


def calculate_ff_dipole_along_axis(
    geometry: str,
    axis: str,
    method: str,
    cp_correction: bool = False,
    field_strength: float = 0.001,
    formula: str = "two_point",
) -> pd.Series:

    # Validate formula
    if formula not in ["two_point", "four_point"]:
        psi4.core.clean()
        raise ValueError(f"Unknown finite difference formula: {formula}")

    molecule = psi4.geometry(geometry)
    psi4_energy_kwargs = {}
    if cp_correction:
        psi4_energy_kwargs["bsse_type"] = "cp"

    psi4.core.print_out(
        f"\nCalculating finite field dipole moment along {axis}-axis with field strength {field_strength:.6f} a.u.\n"
    )

    # Build energies dict using helper utilities
    energies: Dict[str, float] = {}

    if formula == "four_point":
        energies["plus_2h"] = calculate_perturbed_energy(
            method,
            molecule,
            get_field_vector(axis, 2 * field_strength),
            axis,
            **psi4_energy_kwargs,
        )

    energies["plus_h"] = calculate_perturbed_energy(
        method,
        molecule,
        get_field_vector(axis, field_strength),
        axis,
        **psi4_energy_kwargs,
    )

    energies["minus_h"] = calculate_perturbed_energy(
        method,
        molecule,
        get_field_vector(axis, -field_strength),
        axis,
        **psi4_energy_kwargs,
    )

    if formula == "four_point":
        energies["minus_2h"] = calculate_perturbed_energy(
            method,
            molecule,
            get_field_vector(axis, -2 * field_strength),
            axis,
            **psi4_energy_kwargs,
        )

    # Compute derivative and build results Series
    dipole_moment = apply_finite_difference_formula(energies, formula, field_strength)

    results = build_result_series(
        cast(float, dipole_moment), method.upper(), energies, formula, field_strength
    )

    psi4.core.print_out(
        f"Finite field dipole moment calculation along {axis}-axis completed.\n"
    )

    return results


def finite_field_sapt(
    geometry: str,
    prop: str | np.ndarray,
    reference: str = "RHF",
    field_strength: float = 0.001,
    formula: str = "two_point",
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Calculate the interaction-induced property using finite field SAPT.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        prop (str | np.ndarray): Property to calculate (e.g., "dipole").
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        formula (str, optional): Finite difference formula to use ("two_point" or "four_point"). Defaults to "two_point".
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Raises:
        ValueError: If the property is not implemented.
        ValueError: If the SAPT calculation fails.

    Returns:
        pd.DataFrame: DataFrame containing the results of the finite field SAPT calculation.
    """

    # Banner
    print_finite_field_banner(
        "FINITE FIELD SAPT PROPERTY CALCULATIONS",
        {
            "Property": str(prop),
            "Field strength": field_strength,
            "Reference": reference,
            "Formula": formula,
        },
    )

    with CalcTimer("Finite field SAPT property calculations"):
        ### Validate property
        validate_property(prop)

        ### Results output file
        results_fname = kwargs.get("results", "results-ff.csv")

        ### Dipole moment calculations
        results = calc_ff_sapt_dipole(
            geometry,
            reference=reference,
            field_strength=field_strength,
            formula=formula,
            sapt_kwargs=sapt_kwargs,
            **kwargs,
        )

        ### Results saving to file
        results.to_csv(results_fname)

    return results


def calc_ff_sapt_dipole(
    geometry: str,
    reference: str = "RHF",
    field_strength: float = 0.001,
    formula: str = "two_point",
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Calculate the finite field SAPT dipole moment.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        reference (str, optional): Reference wavefunction type. Defaults to "RHF".
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        formula (str, optional): Finite difference formula to use ("two_point" or "four_point"). Defaults to "two_point".
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the results of the finite field SAPT calculation.
    """

    # prepare results
    return apply_to_cartesian_axes(
        calculate_ff_sapt_dipole_along_axis,
        geometry,
        reference=reference,
        field_strength=field_strength,
        sapt_kwargs=sapt_kwargs,
        formula=formula,
        **kwargs,
    )


def calculate_ff_sapt_dipole_along_axis(
    geometry: str,
    axis: str,
    reference: str = "RHF",
    field_strength: float = 0.001,
    formula: str = "two_point",
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.Series:
    """Calculate the finite field SAPT dipole moment along a specified axis.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        axis (str): Axis along which to calculate the dipole moment (e.g., "X", "Y", "Z").
        reference (str, optional): Reference wavefunction type. Defaults to "RHF".
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        formula (str, optional): Finite difference formula to use ("two_point" or "four_point"). Defaults to "two_point".
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Returns:
        pd.Series: Series containing the results of the finite field SAPT calculation.
    """

    # Validate formula
    if formula not in ["two_point", "four_point"]:
        psi4.core.clean()
        raise ValueError(f"Unknown finite difference formula: {formula}")

    psi4.core.print_out(
        f"\nCalculating finite field SAPT dipole moment along {axis}-axis with field strength {field_strength:.6f} a.u.\n"
    )

    # Prepare dimer kwargs
    dimer_kwargs = {
        "reference": reference,
        "functional": kwargs.get("functional", PROPSAPT_OPTIONS["default_functional"]),
        "grac_A": kwargs.get("grac_A"),
        "grac_B": kwargs.get("grac_B"),
    }

    if sapt_kwargs is None:
        sapt_kwargs = {}

    # Build energies dict using helper utilities
    sapt_energies: Dict[str, pd.Series] = {}

    ### Double positive field
    if formula == "four_point":
        sapt_energies["plus_2h"] = calculate_perturbed_sapt_energies(
            geometry,
            get_field_vector(axis, 2 * field_strength),
            axis,
            dimer_kwargs,
            sapt_kwargs,
        )

    ### Positive field
    sapt_energies["plus_h"] = calculate_perturbed_sapt_energies(
        geometry,
        get_field_vector(axis, field_strength),
        axis,
        dimer_kwargs,
        sapt_kwargs,
    )

    ### Negative field
    sapt_energies["minus_h"] = calculate_perturbed_sapt_energies(
        geometry,
        get_field_vector(axis, -field_strength),
        axis,
        dimer_kwargs,
        sapt_kwargs,
    )

    ### Double negative field
    if formula == "four_point":
        sapt_energies["minus_2h"] = calculate_perturbed_sapt_energies(
            geometry,
            get_field_vector(axis, -2 * field_strength),
            axis,
            dimer_kwargs,
            sapt_kwargs,
        )

    ### Calculate the finite field difference
    data = apply_finite_difference_formula(sapt_energies, formula, field_strength)

    psi4.core.print_out(
        f"Finite field SAPT dipole moment calculation along {axis}-axis completed.\n"
    )

    return cast(pd.Series, data)
