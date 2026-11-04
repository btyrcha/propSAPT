# propSAPT

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://btyrcha.github.io/propSAPT/)

propSAPT computes first-order interaction-induced properties and density matrices of molecular
dimers in the spirit of symmetry-adapted perturbation theory (SAPT). The properties, such as the
interaction-induced dipole moment, are analytic derivatives of the SAPT energy corrections,
described in [J. Chem. Theory Comput. 21, 4562 (2025)](https://doi.org/10.1021/acs.jctc.5c00238).
It also gives the SAPT0 / SAPT(DFT) energy decomposition, with uncoupled dispersion.

Finite-field wrappers compute interaction-induced dipoles numerically, as field derivatives:
`finite_field_sapt` of the SAPT energy components, and `finite_field_property` of the
counterpoise-corrected interaction energy of any Psi4 method (e.g. MP2 or CCSD), a
supermolecular reference.

The package is built on [Psi4](https://github.com/psi4/psi4), which runs the SCF calculations
and the integrals. The SAPT expressions are evaluated with NumPy, with density fitting in the
MO basis.

## Installation

propSAPT needs Python 3.12 or later and Psi4 from conda:

```bash
conda create -n prop-sapt -c conda-forge python=3.12 psi4
conda activate prop-sapt
pip install "prop_sapt @ git+https://github.com/btyrcha/propSAPT@v0.4.0"
```

See the [installation guide](https://btyrcha.github.io/propSAPT/installation.html) for the
development install and the optional packages.

## Getting started

Start with the [Quick Start](https://btyrcha.github.io/propSAPT/#quick-start) and the
[usage guide](https://btyrcha.github.io/propSAPT/usage.html). The scripts in
[`examples/`](examples/) cover the main calculations.

## Known limitations

- The S^inf exchange-dispersion property `x2_exch-disp`, and with it the total `x_induced`, has
  a known error ([#1](https://github.com/btyrcha/propSAPT/issues/1)); `x_induced_S2` agrees
  with finite field.
- Dispersion is uncoupled only, so SAPT(DFT) energies differ from Psi4's default (coupled).
- The density totals use S^2 second-order exchange.
- Closed-shell RHF/RKS monomers only, all-electron.

Details are in the [usage guide](https://btyrcha.github.io/propSAPT/usage.html#known-limitations).

## Citing

If you use propSAPT, please cite B. Tyrcha, T. Gupta, K. Patkowski, P. S. Żuchowski,
*Analytical Derivatives of Symmetry-Adapted Perturbation Theory Corrections for
Interaction-Induced Properties*, J. Chem. Theory Comput. **21**, 4562–4578 (2025),
[doi:10.1021/acs.jctc.5c00238](https://doi.org/10.1021/acs.jctc.5c00238). GitHub's "Cite this
repository" gives the same, from [`CITATION.cff`](CITATION.cff).

## License

BSD 3-Clause, see [`LICENSE`](LICENSE). Contributions are welcome, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
