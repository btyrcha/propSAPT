# Finite Field Analysis Examples

This directory contains specialized scripts for analyzing the stability and convergence of finite field SAPT calculations with respect to the electric field strength parameter.

## Scripts Overview

### 1. `field_strength_simple.py`
**Quick convergence test with console output**

A streamlined script for basic field strength stability testing:
- Tests field strengths: [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
- Console-based output with clear formatting
- Calculates relative changes between consecutive field strengths
- Compares with propSAPT analytical calculations
- No external dependencies beyond the propSAPT package

### 2. `field_strength_convergence.py`
**Comprehensive analysis with plotting capabilities**

Full-featured convergence analysis script:
- Extended field strength range: [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
- Detailed convergence metrics and statistics
- High-quality matplotlib plots (optional)
- Automatic identification of optimal field strengths
- CSV output files for further analysis

**Output files:**
- `field_strength_convergence_detailed.csv` - All numerical results
- `convergence_summary.csv` - Statistical summary
- `field_strength_convergence.png/pdf` - Convergence plots (if matplotlib available)

### 3. `richardson_extrapolation.py`
**Advanced extrapolation for highest accuracy**

Implements Richardson extrapolation to eliminate finite field errors:
- Uses geometric field progression: h, h/2, h/4 (default: 0.002, 0.001, 0.0005)
- Extrapolates to zero-field limit using numerical derivatives theory
- Provides error estimates for extrapolated values
- **Most accurate method for eliminating systematic finite field errors**
- Ideal for benchmark calculations

**Output files:**
- `richardson_extrapolation_results.csv` - Extrapolated values and error estimates

## Key Parameters

### Enhanced Convergence Thresholds
All scripts use tightened convergence criteria:
```python
OPTIONS = {
    "e_convergence": 1e-10,        # Energy convergence
    "d_convergence": 1e-10,        # Density matrix convergence  
    "ints_tolerance": 1e-14,       # Integral screening threshold
    "screening": "schwarz",        # Schwarz screening method
    "cholesky_tolerance": 1e-8,    # Density fitting accuracy
}
```
