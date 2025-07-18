import pandas as pd
import psi4
import numpy as np
from prop_sapt import Dimer, calc_property, finite_field_sapt

# specify geometry in Psi4 format
GEO = """
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

# specify memory and threads
MEMORY = "2 GB"
THREADS = 2

# specify basis sets
BASIS = "aug-cc-pvdz"
DF_BASIS = "aug-cc-pvtz"

# specify options with tight convergence
OPTIONS = {
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "df",
    "save_jk": True,  # necessary option
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "ints_tolerance": 1e-14,
    "screening": "schwarz",
    "cholesky_tolerance": 1e-8,
}

# Field strengths for Richardson extrapolation (geometric progression)
BASE_FIELD = 0.002
FIELD_RATIOS = [1.0, 0.5, 0.25]  # h, h/2, h/4
FIELD_STRENGTHS = [BASE_FIELD * ratio for ratio in FIELD_RATIOS]

# specify output filename
OUTPUT_FILE_PATH = "richardson_extrapolation.dat"


def richardson_extrapolation(f_h, f_h2, f_h4):
    """
    Perform Richardson extrapolation to estimate the zero-field limit.

    For a quantity that behaves as f(h) = f(0) + a*h^2 + b*h^4 + ...
    Richardson extrapolation gives: f(0) ≈ (16*f(h/4) - f(h)) / 15
    or the more stable: f(0) ≈ f(h/4) + (f(h/4) - f(h/2)) / 3

    Args:
        f_h: Value at field strength h
        f_h2: Value at field strength h/2
        f_h4: Value at field strength h/4

    Returns:
        Extrapolated value and estimated error
    """

    # First-order Richardson extrapolation using h/4 and h/2
    f_extrap_1 = f_h4 + (f_h4 - f_h2) / 3

    # Second-order Richardson extrapolation using all three points
    # f(0) ≈ (64*f(h/4) - 20*f(h/2) + f(h)) / 45
    f_extrap_2 = (64 * f_h4 - 20 * f_h2 + f_h) / 45

    # Estimate error from difference between extrapolations
    error_estimate = abs(f_extrap_2 - f_extrap_1)

    return f_extrap_2, error_estimate


def finite_field_with_richardson():
    """
    Calculate finite field SAPT with Richardson extrapolation.
    """

    print("=" * 80)
    print("FINITE FIELD SAPT WITH RICHARDSON EXTRAPOLATION")
    print("=" * 80)
    print(f"Base field strength: {BASE_FIELD:.6f}")
    print(f"Field strengths: {FIELD_STRENGTHS}")
    print()

    # Calculate propSAPT reference
    print("Calculating propSAPT reference...")
    dimer = Dimer(GEO)
    prop_sapt_ref = calc_property(dimer, "dipole")
    print("Done.\n")

    # Store finite field results
    ff_results = {}

    # Calculate for each field strength
    for i, field_strength in enumerate(FIELD_STRENGTHS):
        print(f"Calculating finite field SAPT with field = {field_strength:.6f}")

        try:
            result = finite_field_sapt(
                geometry=GEO,
                prop="dipole",
                reference="RHF",
                field_strength=field_strength,
                results=f"richardson_field_{field_strength:.6f}.csv",
            )
            ff_results[field_strength] = result
            print(f"Success: Field {field_strength:.6f}")

        except Exception as e:
            print(f"Error with field {field_strength:.6f}: {e}")
            return None

    print("\nAll finite field calculations completed.")

    # Perform Richardson extrapolation
    components = ['ELST1', 'EXCH1', 'IND2,R', 'EXCH-IND2,R', 'DISP2', 'EXCH-DISP2']
    axes = ['X', 'Y', 'Z']

    extrapolated_results = {}

    print("\n" + "=" * 80)
    print("RICHARDSON EXTRAPOLATION RESULTS")
    print("=" * 80)

    for axis in axes:
        print(f"\n{axis}-axis Results:")
        print("-" * 90)
        print(
            f"{'Component':>15} {'h':>12} {'h/2':>12} {'h/4':>12} {'Extrapolated':>15} {'Error Est.':>12} {'propSAPT':>12}"
        )
        print("-" * 90)

        extrapolated_results[axis] = {}

        for component in components:
            try:
                # Extract values for this component and axis
                f_h = ff_results[FIELD_STRENGTHS[0]].loc[axis, component]
                f_h2 = ff_results[FIELD_STRENGTHS[1]].loc[axis, component]
                f_h4 = ff_results[FIELD_STRENGTHS[2]].loc[axis, component]

                # Perform Richardson extrapolation
                f_extrap, error_est = richardson_extrapolation(f_h, f_h2, f_h4)

                extrapolated_results[axis][component] = {
                    'extrapolated': f_extrap,
                    'error_estimate': error_est,
                    'h': f_h,
                    'h/2': f_h2,
                    'h/4': f_h4,
                }

                # Get propSAPT reference if available
                component_mapping = {
                    'ELST1': 'x1_pol,r',
                    'EXCH1': 'x1_exch,r',
                    'IND2,R': 'x2_ind,r',
                    'EXCH-IND2,R': 'x2_exch-ind_S2',
                    'DISP2': 'x2_disp',
                    'EXCH-DISP2': 'x2_exch-disp_S2',
                }

                prop_ref = np.nan
                if component in component_mapping:
                    prop_comp = component_mapping[component]
                    if (
                        axis in prop_sapt_ref.index
                        and prop_comp in prop_sapt_ref.columns
                    ):
                        prop_ref = prop_sapt_ref.loc[axis, prop_comp]

                # Print results
                print(
                    f"{component:>15} {f_h:12.8f} {f_h2:12.8f} {f_h4:12.8f} "
                    f"{f_extrap:15.8f} {error_est:12.8f} {prop_ref:12.8f}"
                )

            except (KeyError, IndexError) as e:
                print(f"{component:>15} {'Error: Missing data':>60}")
                continue

    # Calculate and display summary statistics
    print("\n" + "=" * 80)
    print("ACCURACY ASSESSMENT")
    print("=" * 80)

    for axis in axes:
        if axis not in extrapolated_results:
            continue

        print(f"\n{axis}-axis accuracy:")
        print("-" * 60)
        print(
            f"{'Component':>15} {'Richardson':>12} {'propSAPT':>12} {'Abs. Error':>12} {'% Error':>10}"
        )
        print("-" * 60)

        for component in components:
            if component not in extrapolated_results[axis]:
                continue

            extrap_val = extrapolated_results[axis][component]['extrapolated']

            # Get propSAPT reference
            component_mapping = {
                'ELST1': 'x1_pol,r',
                'EXCH1': 'x1_exch,r',
                'IND2,R': 'x2_ind,r',
                'EXCH-IND2,R': 'x2_exch-ind_S2',
                'DISP2': 'x2_disp',
                'EXCH-DISP2': 'x2_exch-disp_S2',
            }

            if component in component_mapping:
                prop_comp = component_mapping[component]
                if axis in prop_sapt_ref.index and prop_comp in prop_sapt_ref.columns:
                    prop_ref = prop_sapt_ref.loc[axis, prop_comp]
                    abs_error = abs(extrap_val - prop_ref)
                    percent_error = (
                        abs_error / abs(prop_ref) * 100 if prop_ref != 0 else np.inf
                    )

                    print(
                        f"{component:>15} {extrap_val:12.8f} {prop_ref:12.8f} "
                        f"{abs_error:12.8f} {percent_error:10.4f}"
                    )

    # Save extrapolated results
    save_extrapolated_results(extrapolated_results, prop_sapt_ref)

    return extrapolated_results


def save_extrapolated_results(extrapolated_results, prop_sapt_ref):
    """
    Save Richardson extrapolation results to CSV files.
    """

    # Prepare data for saving
    results_data = []

    for axis in ['X', 'Y', 'Z']:
        if axis not in extrapolated_results:
            continue

        for component, data in extrapolated_results[axis].items():
            row = {
                'axis': axis,
                'component': component,
                'field_h': data['h'],
                'field_h2': data['h/2'],
                'field_h4': data['h/4'],
                'richardson_extrapolated': data['extrapolated'],
                'error_estimate': data['error_estimate'],
            }

            # Add propSAPT reference if available
            component_mapping = {
                'ELST1': 'x1_pol,r',
                'EXCH1': 'x1_exch,r',
                'IND2,R': 'x2_ind,r',
                'EXCH-IND2,R': 'x2_exch-ind_S2',
                'DISP2': 'x2_disp',
                'EXCH-DISP2': 'x2_exch-disp_S2',
            }

            if component in component_mapping:
                prop_comp = component_mapping[component]
                if axis in prop_sapt_ref.index and prop_comp in prop_sapt_ref.columns:
                    row['propSAPT_reference'] = prop_sapt_ref.loc[axis, prop_comp]
                    row['absolute_error'] = abs(
                        row['richardson_extrapolated'] - row['propSAPT_reference']
                    )
                    if row['propSAPT_reference'] != 0:
                        row['percent_error'] = (
                            row['absolute_error'] / abs(row['propSAPT_reference']) * 100
                        )

            results_data.append(row)

    # Save to CSV
    df_results = pd.DataFrame(results_data)
    df_results.to_csv("richardson_extrapolation_results.csv", index=False)

    print(f"\nResults saved to 'richardson_extrapolation_results.csv'")


if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    # Run Richardson extrapolation analysis
    results = finite_field_with_richardson()

    if results:
        print("\n" + "=" * 80)
        print("SUMMARY AND RECOMMENDATIONS")
        print("=" * 80)
        print(
            "1. Richardson extrapolation provides improved accuracy over single field strength"
        )
        print("2. Compare extrapolated results with propSAPT reference values")
        print("3. Error estimates help assess reliability of extrapolated values")
        print("4. Consider using smaller field strengths if error estimates are large")
        print(
            "5. This method is particularly useful for properties sensitive to field strength"
        )
    else:
        print("Richardson extrapolation failed due to calculation errors.")
