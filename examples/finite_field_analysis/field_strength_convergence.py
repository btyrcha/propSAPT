import pandas as pd
import psi4
import numpy as np
import matplotlib.pyplot as plt
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
FIELD_STRENGTHS = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]

# Components to analyze for convergence
SAPT_COMPONENTS = ['ELST1', 'EXCH1', 'IND2,R', 'EXCH-IND2,R', 'DISP2', 'EXCH-DISP2']

# specify output filename
OUTPUT_FILE_PATH = "field_strength_convergence.dat"


def analyze_field_strength_convergence():
    """
    Analyze the convergence of finite field SAPT results with respect to field strength.
    """

    # Initialize results storage
    convergence_results = []

    print("=" * 80)
    print("FINITE FIELD STRENGTH CONVERGENCE ANALYSIS")
    print("=" * 80)
    print(
        f"{'Field Strength':>15} {'Axis':>5} {'ELST1':>12} {'EXCH1':>12} {'IND2,R':>12} {'EXCH-IND2,R':>15} {'DISP2':>12} {'EXCH-DISP2':>15}"
    )
    print("-" * 115)

    # Calculate reference with propSAPT (analytical derivatives)
    dimer = Dimer(GEO)
    data_prop_sapt = calc_property(dimer, "dipole")

    # Test each field strength
    for field_strength in FIELD_STRENGTHS:
        print(f"\nCalculating with field strength: {field_strength:.6f}")

        try:
            # Calculate finite field SAPT with current field strength
            data_ff_sapt = finite_field_sapt(
                geometry=GEO,
                prop="dipole",
                reference="RHF",
                field_strength=field_strength,
                results=f"results_field_{field_strength:.6f}.csv",
            )

            # Store results for each axis
            for axis in ['X', 'Y', 'Z']:
                if axis in data_ff_sapt.index:
                    row_data = {
                        'field_strength': field_strength,
                        'axis': axis,
                    }

                    # Extract SAPT components
                    for component in SAPT_COMPONENTS:
                        if component in data_ff_sapt.columns:
                            row_data[component] = data_ff_sapt.loc[axis, component]
                        else:
                            row_data[component] = np.nan

                    convergence_results.append(row_data)

                    # Print current results
                    print(
                        f"{field_strength:15.6f} {axis:>5} "
                        f"{row_data.get('ELST1', np.nan):12.6f} "
                        f"{row_data.get('EXCH1', np.nan):12.6f} "
                        f"{row_data.get('IND2,R', np.nan):12.6f} "
                        f"{row_data.get('EXCH-IND2,R', np.nan):15.6f} "
                        f"{row_data.get('DISP2', np.nan):12.6f} "
                        f"{row_data.get('EXCH-DISP2', np.nan):15.6f}"
                    )

        except Exception as e:
            print(f"Error with field strength {field_strength}: {e}")
            continue

    # Convert to DataFrame for analysis
    df_convergence = pd.DataFrame(convergence_results)

    # Save detailed results
    df_convergence.to_csv("field_strength_convergence_detailed.csv", index=False)

    return df_convergence, data_prop_sapt


def plot_convergence_analysis(df_convergence, data_prop_sapt):
    """
    Create convergence plots for each SAPT component.
    """

    # Set up the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = {'X': 'red', 'Y': 'blue', 'Z': 'green'}

    for i, component in enumerate(SAPT_COMPONENTS):
        ax = axes[i]

        # Plot data for each axis
        for axis in ['X', 'Y', 'Z']:
            axis_data = df_convergence[df_convergence['axis'] == axis]
            if not axis_data.empty and component in axis_data.columns:
                field_strengths = axis_data['field_strength'].values
                values = axis_data[component].values

                ax.semilogx(
                    field_strengths,
                    values,
                    'o-',
                    color=colors[axis],
                    label=f'{axis}-axis',
                    markersize=6,
                    linewidth=2,
                )

                # Add reference line from propSAPT (if available)
                if axis in data_prop_sapt.index:
                    # Map SAPT component names to propSAPT names
                    prop_component_map = {
                        'ELST1': 'x1_pol,r',
                        'EXCH1': 'x1_exch,r',
                        'IND2,R': 'x2_ind,r',
                        'EXCH-IND2,R': 'x2_exch-ind_S2',
                        'DISP2': 'x2_disp',
                        'EXCH-DISP2': 'x2_exch-disp_S2',
                    }

                    if component in prop_component_map:
                        prop_component = prop_component_map[component]
                        if prop_component in data_prop_sapt.columns:
                            ref_value = data_prop_sapt.loc[axis, prop_component]
                            ax.axhline(
                                y=ref_value,
                                color=colors[axis],
                                linestyle='--',
                                alpha=0.7,
                                label=f'{axis}-axis (propSAPT ref)',
                            )

        ax.set_xlabel('Field Strength (a.u.)')
        ax.set_ylabel(f'{component}')
        ax.set_title(f'Convergence of {component}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig('field_strength_convergence.png', dpi=300, bbox_inches='tight')
    plt.savefig('field_strength_convergence.pdf', bbox_inches='tight')
    plt.show()


def calculate_convergence_metrics(df_convergence):
    """
    Calculate convergence metrics for each component.
    """

    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by axis and component
    convergence_summary = []

    for axis in ['X', 'Y', 'Z']:
        print(f"\n{axis}-axis convergence:")
        print("-" * 40)

        axis_data = df_convergence[df_convergence['axis'] == axis].sort_values(
            'field_strength'
        )

        for component in SAPT_COMPONENTS:
            if component in axis_data.columns:
                values = axis_data[component].dropna().values
                field_strengths = axis_data[axis_data[component].notna()][
                    'field_strength'
                ].values

                if len(values) > 2:
                    # Calculate relative changes between consecutive field strengths
                    rel_changes = np.abs(np.diff(values) / values[1:]) * 100

                    # Find the field strength where relative change < 0.1%
                    converged_idx = np.where(rel_changes < 0.1)[0]

                    if len(converged_idx) > 0:
                        converged_field = field_strengths[converged_idx[0] + 1]
                        converged_value = values[converged_idx[0] + 1]
                        print(
                            f"  {component:>15}: Converged at field = {converged_field:.6f}, value = {converged_value:.8f}"
                        )
                    else:
                        print(f"  {component:>15}: Not converged within tested range")

                    # Show final value and std deviation of last 3 points
                    if len(values) >= 3:
                        final_values = values[-3:]
                        final_mean = np.mean(final_values)
                        final_std = np.std(final_values)
                        print(
                            f"  {component:>15}: Final mean = {final_mean:.8f} ± {final_std:.8f}"
                        )

                        convergence_summary.append(
                            {
                                'axis': axis,
                                'component': component,
                                'final_mean': final_mean,
                                'final_std': final_std,
                                'converged_field': (
                                    converged_field
                                    if len(converged_idx) > 0
                                    else np.nan
                                ),
                                'relative_std_percent': (
                                    (final_std / abs(final_mean)) * 100
                                    if final_mean != 0
                                    else np.nan
                                ),
                            }
                        )

    # Save convergence summary
    df_summary = pd.DataFrame(convergence_summary)
    df_summary.to_csv("convergence_summary.csv", index=False)

    return df_summary


if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    # Run convergence analysis
    df_convergence, data_prop_sapt = analyze_field_strength_convergence()

    # Calculate convergence metrics
    df_summary = calculate_convergence_metrics(df_convergence)

    # Create convergence plots
    try:
        plot_convergence_analysis(df_convergence, data_prop_sapt)
        print(
            f"\nConvergence plots saved as 'field_strength_convergence.png' and 'field_strength_convergence.pdf'"
        )
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    except Exception as e:
        print(f"\nError creating plots: {e}")

    print(f"\nDetailed results saved to 'field_strength_convergence_detailed.csv'")
    print(f"Convergence summary saved to 'convergence_summary.csv'")

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("Based on the convergence analysis:")
    print(
        "1. Use the smallest field strength that gives stable results (< 0.1% variation)"
    )
    print("2. Check that results are converged across all SAPT components")
    print("3. Compare with propSAPT reference values for validation")
    print("4. Consider using Richardson extrapolation for improved accuracy")
