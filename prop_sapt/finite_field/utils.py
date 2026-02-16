"""
Utilities for finite field property calculations and finite field SAPT.
"""

from typing import Dict, Callable

import psi4
import pandas as pd


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
    magnitude = sum([abs(v) for v in field_vector])
    sign = "+" if magnitude >= 0 else "-"

    psi4.set_options(
        {
            "PERTURB_H": True,
            "PERTURB_DIPOLE": field_vector,
        }
    )

    psi4.core.print_out(
        f"Computing energy with {sign} field ({magnitude:.6f}) along {axis}-axis...\n"
    )

    energy = psi4.energy(method, molecule=molecule, **energy_kwargs)

    psi4.core.revoke_global_option_changed("PERTURB_H")
    psi4.core.revoke_global_option_changed("PERTURB_DIPOLE")

    return energy


def apply_finite_difference_formula(
    energies: Dict[str, float],
    formula: str,
    field_strength: float,
) -> float:
    """Apply finite difference formula to compute property derivative.

    Args:
        energies (Dict[str, float]): Dictionary of energies with keys:
            - "plus_h": Energy at +h
            - "minus_h": Energy at -h
            - "plus_2h": Energy at +2h (for four_point)
            - "minus_2h": Energy at -2h (for four_point)
        formula (str): Finite difference formula ("two_point" or "four_point").
        field_strength (float): Field strength h.

    Returns:
        float: Computed derivative (property value).

    Raises:
        ValueError: If formula is invalid or required energies missing.
    """

    if formula == "two_point":
        if "plus_h" not in energies or "minus_h" not in energies:
            raise ValueError(
                "two_point formula requires 'plus_h' and 'minus_h' energies"
            )

        psi4.core.print_out("Using two-point finite difference formula.\n")
        derivative = (energies["plus_h"] - energies["minus_h"]) / (2 * field_strength)

    elif formula == "four_point":
        required_keys = ["plus_2h", "plus_h", "minus_h", "minus_2h"]
        if not all(key in energies for key in required_keys):
            raise ValueError(f"four_point formula requires all of: {required_keys}")

        psi4.core.print_out("Using four-point finite difference formula.\n")
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
    """Print a formatted banner with title and details for finite field calculations.

    Args:
        title (str): Title to display in the banner (e.g., "FINITE FIELD PROPERTY CALCULATIONS").
        details (Dict[str, str | float]): Dictionary of detail key-value pairs to display.
            Keys will be formatted as "  Key: value".
    """
    psi4.core.print_out("\n" + "=" * 70 + "\n")
    psi4.core.print_out(f"  {title}\n")
    psi4.core.print_out("=" * 70 + "\n")

    for key, value in details.items():
        if isinstance(value, float):
            psi4.core.print_out(f"  {key}: {value:.6f} a.u.\n")
        else:
            psi4.core.print_out(f"  {key}: {value}\n")

    psi4.core.print_out("=" * 70 + "\n")


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
