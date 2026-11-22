"""
Utilities for finite field property calculations and finite field SAPT.
"""

import logging
from typing import Dict, Callable, Optional

import psi4
import pandas as pd

from numpy.linalg import vector_norm

from ..molecule import Dimer
from ..sapt_driver import calc_sapt_energy

logger = logging.getLogger(__name__)


def get_field_vector(axis: str, magnitude: float) -> list:
    """Generate a field vector for a given axis and magnitude.

    Args:
        axis (str): Cartesian axis ("X", "Y", or "Z").
        magnitude (float): Magnitude of the field.

    Returns:
        list: Field vector [x, y, z] with magnitude in the specified axis.

    Raises:
        ValueError: If axis is not "X", "Y", or "Z".
    """
    if axis not in ["X", "Y", "Z"]:
        raise ValueError(f"Invalid axis: {axis}. Must be 'X', 'Y', or 'Z'.")

    field_vectors = {
        "X": [magnitude, 0.0, 0.0],
        "Y": [0.0, magnitude, 0.0],
        "Z": [0.0, 0.0, magnitude],
    }
    return field_vectors[axis]


def calculate_perturbed_energy(
    method: str,
    molecule,
    field_vector: list,
    axis: str,
    **energy_kwargs,
) -> float:
    """Calculate energy with an external electric field perturbation.

    Args:
        method (str): Electronic structure method (e.g., "HF", "DFT").
        molecule: Psi4 molecule object.
        field_vector (list): Electric field vector [x, y, z].
        axis (str): Cartesian axis for logging.
        **energy_kwargs: Additional keyword arguments for psi4.energy().

    Returns:
        float: Energy computed with the perturbation.
    """
    # Determine field sign and magnitude for output
    magnitude = vector_norm(field_vector)

    # NOTE: even if we're wrong now, with -field_vector we'll get opposite sign
    sign = "+" if sum(field_vector) >= 0 else "-"

    try:
        psi4.set_options(
            {
                "PERTURB_H": True,
                "PERTURB_DIPOLE": field_vector,
            }
        )

        logger.info(
            f"Computing energy with {sign} field ({magnitude:.6f}) along {axis}-axis..."
        )

        energy = psi4.energy(method, molecule=molecule, **energy_kwargs)

    finally:
        psi4.core.revoke_global_option_changed("PERTURB_H")
        psi4.core.revoke_global_option_changed("PERTURB_DIPOLE")

    return energy


def calculate_perturbed_sapt_energies(
    geometry: str,
    field_vector: list,
    axis: str,
    dimer_kwargs: Optional[Dict] = None,
    sapt_kwargs: Optional[Dict] = None,
) -> pd.Series:
    """Calculate SAPT energies with an external electric field perturbation.

    Args:
        geometry (str): Psi4 geometry string.
        field_vector (list): Electric field vector [x, y, z].
        axis (str): Cartesian axis for logging.
        dimer_kwargs (Optional[Dict], optional): Additional keyword arguments for prop_sapt.Dimer().
        sapt_kwargs (Optional[Dict], optional):  Additional keyword arguments for prop_sapt.calc_sapt_energy().

    Returns:
        pd.Series: SAPT energies computed with the perturbation.
    """

    if sapt_kwargs is None:
        sapt_kwargs = {}

    if dimer_kwargs is None:
        dimer_kwargs = {}

    # Determine field sign and magnitude for output
    magnitude = vector_norm(field_vector)

    # NOTE: even if we're wrong now, with -field_vector we'll get opposite sign
    sign = "+" if sum(field_vector) >= 0 else "-"

    try:
        psi4.set_options(
            {
                "PERTURB_H": True,
                "PERTURB_DIPOLE": field_vector,
            }
        )

        logger.info(
            f"Computing SAPT energy with {sign} field ({magnitude:.6f}) along {axis}-axis..."
        )

        dimer = Dimer(
            geometry,
            **{key: value for key, value in dimer_kwargs.items() if value is not None},
        )

        results = calc_sapt_energy(
            dimer,
            results=f"results-{axis}-{sign}{magnitude}.csv",
            **sapt_kwargs,
        )

    finally:
        psi4.core.revoke_global_option_changed("PERTURB_H")
        psi4.core.revoke_global_option_changed("PERTURB_DIPOLE")

    return results


def apply_finite_difference_formula(
    energies: Dict[str, float] | Dict[str, pd.Series],
    formula: str,
    field_strength: float,
) -> float | pd.Series:
    """Apply finite difference formula to compute property derivative.

    Args:
        energies (Dict[str, float] | Dict[str, pd.Series]): Dictionary of energies with keys:
            - "plus_h": Energy at +h
            - "minus_h": Energy at -h
            - "plus_2h": Energy at +2h (for four_point)
            - "minus_2h": Energy at -2h (for four_point)
        formula (str): Finite difference formula ("two_point" or "four_point").
        field_strength (float): Field strength h.

    Returns:
        float | pd.Series: Computed derivative (property value).

    Raises:
        ValueError: If formula is invalid or required energies missing.
    """

    if formula == "two_point":
        if "plus_h" not in energies or "minus_h" not in energies:
            raise ValueError(
                "two_point formula requires 'plus_h' and 'minus_h' energies"
            )

        logger.info("Using two-point finite difference formula.")
        derivative = (energies["plus_h"] - energies["minus_h"]) / (2 * field_strength)

    elif formula == "four_point":
        required_keys = ["plus_2h", "plus_h", "minus_h", "minus_2h"]
        if not all(key in energies for key in required_keys):
            raise ValueError(f"four_point formula requires all of: {required_keys}")

        logger.info("Using four-point finite difference formula.")
        derivative = (
            -energies["plus_2h"]
            + 8 * energies["plus_h"]
            - 8 * energies["minus_h"]
            + energies["minus_2h"]
        ) / (12 * field_strength)

    else:
        raise ValueError(f"Unknown finite difference formula: {formula}")

    return derivative


def build_result_series(
    derivative: float,
    label: str,
    energies: Dict[str, float],
    formula: str,
    field_strength: float,
) -> pd.Series:
    """Build a result series from derivative and energy values.

    Args:
        derivative (float): Computed derivative (property value).
        label (str): Label for the derivative column (e.g., "HF", "DFT").
        energies (Dict[str, float]): Dictionary of energies.
        formula (str): Finite difference formula used ("two_point" or "four_point").
        field_strength (float): Field strength h.

    Returns:
        pd.Series: Series containing derivative and energy values.
    """
    result_dict = {label: derivative}

    if formula == "two_point":
        result_dict[f"{label}_+{field_strength}"] = energies["plus_h"]
        result_dict[f"{label}_-{field_strength}"] = energies["minus_h"]

    elif formula == "four_point":
        result_dict[f"{label}_+{2*field_strength}"] = energies["plus_2h"]
        result_dict[f"{label}_+{field_strength}"] = energies["plus_h"]
        result_dict[f"{label}_-{field_strength}"] = energies["minus_h"]
        result_dict[f"{label}_-{2*field_strength}"] = energies["minus_2h"]

    else:
        raise ValueError(f"Unknown finite difference formula: {formula}")

    return pd.Series(result_dict)


def print_finite_field_banner(title: str, details: Dict[str, str | float]) -> None:
    """Log a formatted banner with title and details for finite field calculations.

    Args:
        title (str): Title to display in the banner (e.g., "FINITE FIELD PROPERTY CALCULATIONS").
        details (Dict[str, str | float]): Dictionary of detail key-value pairs to display.
            Keys will be formatted as "  Key: value".
    """
    rule = "=" * 70
    lines = [rule, f"  {title}", rule]
    for key, value in details.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6f} a.u.")
        else:
            lines.append(f"  {key}: {value}")
    lines.append(rule)

    logger.info("\n".join(lines))


def apply_to_cartesian_axes(
    func: Callable, *func_args, axes=("X", "Y", "Z"), **func_kwargs
) -> pd.DataFrame:
    """Apply a function to X, Y, Z axes and aggregate results into a DataFrame.

    The provided function `func` must accept an `axis` keyword argument and return a
    pandas Series for that axis. `func_args` and `func_kwargs` are forwarded to `func`.
    """
    results = pd.DataFrame()

    for axis in axes:
        res_i = func(*func_args, axis=axis, **func_kwargs)
        res_i["axis"] = axis
        results = pd.concat([results, res_i.to_frame().T])

    results.set_index("axis", inplace=True)
    return results
