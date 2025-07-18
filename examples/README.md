# propSAPT Examples

This directory contains example scripts and tutorials demonstrating various features of the propSAPT package for calculating interaction-induced properties using SAPT theory.

## Basic Examples

### Core Functionality
- **`example.py`** - General usage examples
- **`sapt_example.py`** - Standard SAPT energy decomposition example  
- **`finite_field_sapt.py`** - Basic finite field SAPT calculation for dipole moments

### Property Calculations
- **`water_dimer.py`** - Water dimer interaction analysis
- **`read_subtract_save.py`** - Working with density matrices and cube files


## Visualization Examples

### Jupyter Notebooks
- **`visualise_density.ipynb`** - Density matrix visualization
- **`visualise_orbital.ipynb`** - Molecular orbital visualization

## Specialized Analysis

### [`finite_field_analysis/`](finite_field_analysis/)
Advanced finite field stability and convergence analysis:
- Field strength convergence testing
- Richardson extrapolation for improved accuracy
- Comprehensive plotting and error analysis

## Notes

- All examples use the aug-cc-pVDZ basis set by default for speed
- Modify `BASIS` and `DF_BASIS` variables for production calculations
- Check individual script headers for specific requirements and options
