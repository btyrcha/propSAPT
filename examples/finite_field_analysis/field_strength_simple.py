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

# specify options with tight convergence for field strength testing
OPTIONS = {
    "basis": BASIS,
    "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "DF_BASIS_SAPT": DF_BASIS + "-ri",
    "scf_type": "df",
    "save_jk": True,  # necessary option
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    # Integral thresholds for higher precision
    "ints_tolerance": 1e-14,
    "screening": "schwarz",
    "cholesky_tolerance": 1e-8,
}

# Field strength values to test (in atomic units)
FIELD_STRENGTHS = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

# specify output filename
OUTPUT_FILE_PATH = "field_strength_simple.dat"


def test_field_strength_stability():
    """
    Simple test of finite field stability across different field strengths.
    """

    print("=" * 80)
    print("FINITE FIELD STRENGTH STABILITY TEST")
    print("=" * 80)
    print("Testing field strengths:", FIELD_STRENGTHS)
    print()

    # Store results for comparison
    all_results = {}

    # Calculate reference with propSAPT
    print("Calculating propSAPT reference...")
    dimer = Dimer(GEO)
    prop_sapt_ref = calc_property(dimer, "dipole")
    print("propSAPT reference calculated.")
    print()

    # Test each field strength
    for i, field_strength in enumerate(FIELD_STRENGTHS):
        print(f"Field strength {i+1}/{len(FIELD_STRENGTHS)}: {field_strength:.6f}")
        print("-" * 50)

        try:
            # Calculate finite field SAPT
            ff_results = finite_field_sapt(
                geometry=GEO,
                prop="dipole",
                reference="RHF",
                field_strength=field_strength,
                results=f"results_field_{field_strength:.6f}.csv",
            )

            all_results[field_strength] = ff_results

            # Print results for X-component as example
            if 'X' in ff_results.index:
                x_results = ff_results.loc['X']
                print(f"X-component results:")
                for component in ['ELST1', 'EXCH1', 'IND2,R', 'DISP2']:
                    if component in x_results:
                        print(f"  {component:>12}: {x_results[component]:12.8f}")
            print()

        except Exception as e:
            print(f"Error with field strength {field_strength}: {e}")
            print()
            continue

    # Analyze convergence
    print("=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    if len(all_results) >= 2:
        analyze_convergence(all_results, prop_sapt_ref)
    else:
        print("Not enough successful calculations for convergence analysis.")


def analyze_convergence(all_results, prop_sapt_ref):
    """
    Analyze convergence of finite field results.
    """

    # Components to analyze
    components = ['ELST1', 'EXCH1', 'IND2,R', 'EXCH-IND2,R', 'DISP2', 'EXCH-DISP2']
    axes = ['X', 'Y', 'Z']

    # Create convergence table
    field_strengths = sorted(all_results.keys())

    for axis in axes:
        print(f"\n{axis}-axis convergence:")
        print("-" * 80)
        print(f"{'Field':>10} {'ELST1':>12} {'EXCH1':>12} {'IND2,R':>12} {'DISP2':>12}")
        print("-" * 80)

        # Collect values for this axis
        axis_data = {}
        for component in components[:4]:  # Show first 4 components
            axis_data[component] = []

        for field in field_strengths:
            if field in all_results and axis in all_results[field].index:
                row_data = all_results[field].loc[axis]
                print(f"{field:10.6f}", end="")

                for component in components[:4]:
                    value = row_data.get(component, np.nan)
                    axis_data[component].append(value)
                    print(f"{value:12.6f}", end="")
                print()

        # Calculate relative changes
        print(f"\nRelative changes (%) from previous field strength:")
        print("-" * 80)
        print(f"{'Field':>10} {'ELST1':>12} {'EXCH1':>12} {'IND2,R':>12} {'DISP2':>12}")
        print("-" * 80)

        for i, field in enumerate(field_strengths[1:], 1):
            print(f"{field:10.6f}", end="")

            for component in components[:4]:
                if len(axis_data[component]) > i and len(axis_data[component]) > i - 1:
                    current = axis_data[component][i]
                    previous = axis_data[component][i - 1]

                    if (
                        not np.isnan(current)
                        and not np.isnan(previous)
                        and previous != 0
                    ):
                        rel_change = abs((current - previous) / previous) * 100
                        print(f"{rel_change:12.4f}", end="")
                    else:
                        print(f"{'N/A':>12}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()

        # Compare with propSAPT reference
        if axis in prop_sapt_ref.index:
            print(f"\nComparison with propSAPT reference:")
            print("-" * 80)

            # Mapping between finite field and propSAPT component names
            component_mapping = {
                'ELST1': 'x1_pol,r',
                'EXCH1': 'x1_exch,r',
                'IND2,R': 'x2_ind,r',
                'DISP2': 'x2_disp',
            }

            for ff_comp, prop_comp in component_mapping.items():
                if prop_comp in prop_sapt_ref.columns:
                    ref_value = prop_sapt_ref.loc[axis, prop_comp]
                    print(f"{ff_comp:>12} reference: {ref_value:12.8f}")

                    # Show deviation for each field strength
                    print(
                        f"{'Field':>10} {'Value':>12} {'Deviation':>12} {'% Error':>12}"
                    )
                    for i, field in enumerate(field_strengths):
                        if (
                            field in all_results
                            and axis in all_results[field].index
                            and ff_comp in all_results[field].columns
                        ):

                            ff_value = all_results[field].loc[axis, ff_comp]
                            deviation = ff_value - ref_value
                            percent_error = (
                                abs(deviation / ref_value) * 100
                                if ref_value != 0
                                else 0
                            )

                            print(
                                f"{field:10.6f} {ff_value:12.8f} {deviation:12.8f} {percent_error:12.4f}"
                            )
                    print()


def print_recommendations(all_results):
    """
    Print recommendations based on the convergence analysis.
    """

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    field_strengths = sorted(all_results.keys())

    if len(field_strengths) >= 3:
        # Suggest optimal field strength based on stability
        print("1. Optimal field strength selection:")
        print(
            f"   - Consider field strengths between {field_strengths[1]:.6f} and {field_strengths[-2]:.6f}"
        )
        print(
            f"   - Avoid too small fields (< {field_strengths[1]:.6f}) due to numerical noise"
        )
        print(
            f"   - Avoid too large fields (> {field_strengths[-2]:.6f}) due to nonlinear effects"
        )
        print()

    print("2. Convergence criteria:")
    print("   - Look for field strengths where relative changes < 0.1%")
    print("   - Ensure all SAPT components show similar convergence patterns")
    print("   - Compare with propSAPT reference values when available")
    print()

    print("3. Quality checks:")
    print("   - Results should be smooth functions of field strength")
    print("   - Large oscillations indicate numerical instability")
    print("   - Consider Richardson extrapolation for higher accuracy")


if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    # Run the field strength stability test
    test_field_strength_stability()

    print("\nField strength stability test completed!")
    print("Check the output files for detailed results.")
