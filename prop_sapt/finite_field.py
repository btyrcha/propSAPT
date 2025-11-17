from typing import Dict, Optional

import numpy as np
import pandas as pd
import psi4

from prop_sapt import Dimer, calc_sapt_energy
from .utils import CalcTimer


def finite_field_sapt(
    geometry: str,
    prop: str | np.ndarray,
    reference: str = "RHF",
    field_strength: float = 0.001,
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Calculate the interaction-induced property using finite field SAPT.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        prop (str | np.ndarray): Property to calculate (e.g., "dipole").
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Raises:
        ValueError: If the property is not implemented.
        ValueError: If the SAPT calculation fails.

    Returns:
        pd.DataFrame: DataFrame containing the results of the finite field SAPT calculation.
    """

    # Banner
    psi4.core.print_out("\n" + "=" * 70 + "\n")
    psi4.core.print_out("  FINITE FIELD SAPT PROPERTY CALCULATIONS\n")
    psi4.core.print_out("=" * 70 + "\n")
    psi4.core.print_out(f"  Property: {prop}\n")
    psi4.core.print_out(f"  Field strength: {field_strength:.6f} a.u.\n")
    psi4.core.print_out(f"  Reference: {reference}\n")
    psi4.core.print_out("=" * 70 + "\n")

    with CalcTimer("Finite field SAPT property calculations"):
        ### Results output file
        results_fname = kwargs.get("results", "results-ff.csv")

        if prop == "dipole":
            ### Dipole moment calculations
            results = calc_ff_sapt_dipole(
                geometry,
                reference=reference,
                field_strength=field_strength,
                sapt_kwargs=sapt_kwargs,
                **kwargs,
            )

        elif isinstance(prop, np.ndarray):
            ### Property matrix is given
            psi4.core.clean()
            raise ValueError(
                "Property for general external operator is not implemented."
            )

        else:
            psi4.core.clean()
            raise ValueError(
                f"Property {prop} is not implemented. "
                "Please provide a valid property name or a property matrix."
            )

        ### Results saving to file
        results.to_csv(results_fname)

    return results


def calc_ff_sapt_dipole(
    geometry: str,
    reference: str = "RHF",
    field_strength: float = 0.001,
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """Calculate the finite field SAPT dipole moment.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        reference (str, optional): Reference wavefunction type. Defaults to "RHF".
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the results of the finite field SAPT calculation.
    """

    # prepare results
    results = pd.DataFrame()

    # calculate for each axis
    for i in ["X", "Y", "Z"]:
        res_i = calculate_ff_sapt_dipole_along_axis(
            geometry,
            axis=i,
            reference=reference,
            field_strength=field_strength,
            sapt_kwargs=sapt_kwargs,
            **kwargs,
        )
        res_i["axis"] = i
        results = pd.concat([results, res_i.to_frame().T])

    results.set_index("axis", inplace=True)

    return results


def calculate_ff_sapt_dipole_along_axis(
    geometry: str,
    axis: str,
    reference: str = "RHF",
    field_strength: float = 0.001,
    sapt_kwargs: Optional[Dict] = None,
    **kwargs,
) -> pd.Series:
    """Calculate the finite field SAPT dipole moment along a specified axis.

    Args:
        geometry (str): Geometry of the dimer in Psi4 format.
        axis (str): Axis along which to calculate the dipole moment (e.g., "X", "Y", "Z").
        reference (str, optional): Reference wavefunction type. Defaults to "RHF".
        field_strength (float, optional): Strength of the external electric field. Defaults to 0.001.
        sapt_kwargs (Optional[Dict], optional): Additional arguments for SAPT calculations. Defaults to None.

    Returns:
        pd.Series: Series containing the results of the finite field SAPT calculation.
    """

    psi4.core.print_out(
        f"\nCalculating finite field SAPT dipole moment along {axis}-axis with field strength {field_strength:.6f} a.u.\n"
    )

    ### Postitive field
    field_vector_positive = {
        "X": [field_strength, 0.0, 0.0],
        "Y": [0.0, field_strength, 0.0],
        "Z": [0.0, 0.0, field_strength],
    }

    psi4.set_options(
        {
            "PERTURB_H": True,
            "PERTURB_DIPOLE": field_vector_positive[axis],
        }
    )

    psi4.core.print_out(
        f"Computing SAPT energy with positive field (+{field_strength:.6f}) along {axis}-axis...\n"
    )

    dimer_kwargs = {
        "reference": reference,
        "functional": kwargs.get("functional"),
        "grac_A": kwargs.get("grac_A"),
        "grac_B": kwargs.get("grac_B"),
    }

    if sapt_kwargs is None:
        sapt_kwargs = {}

    data_positive = calc_sapt_energy(
        Dimer(
            geometry,
            **{key: value for key, value in dimer_kwargs.items() if value is not None},
        ),
        results=f"results-{axis}-{field_strength}.csv",
        **sapt_kwargs,
    )

    psi4.core.revoke_global_option_changed("PERTURB_H")
    psi4.core.revoke_global_option_changed("PERTURB_DIPOLE")

    ### Negative field
    field_vector_negative = {
        "X": [-field_strength, 0.0, 0.0],
        "Y": [0.0, -field_strength, 0.0],
        "Z": [0.0, 0.0, -field_strength],
    }

    psi4.set_options(
        {
            "PERTURB_H": True,
            "PERTURB_DIPOLE": field_vector_negative[axis],
        }
    )

    psi4.core.print_out(
        f"Computing SAPT energy with negative field (-{field_strength:.6f}) along {axis}-axis...\n"
    )

    if sapt_kwargs is None:
        sapt_kwargs = {}

    data_negative = calc_sapt_energy(
        Dimer(
            geometry,
            **{key: value for key, value in dimer_kwargs.items() if value is not None},
        ),
        results=f"results-{axis}-{-field_strength}.csv",
        **sapt_kwargs,
    )

    psi4.core.revoke_global_option_changed("PERTURB_H")
    psi4.core.revoke_global_option_changed("PERTURB_DIPOLE")

    ### Calculate the finite field difference
    data = (data_positive - data_negative) / (2 * field_strength)

    psi4.core.print_out(
        f"Finite field SAPT dipole moment calculation along {axis}-axis completed.\n"
    )

    return data
