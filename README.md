# propSAPT

Python package for calculations of first-order interaction-induced properties and changes in monomers density matrices in the spirit of SAPT. The theoretical work has been published [here](https://doi.org/10.1021/acs.jctc.5c00238).

[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://btyrcha.github.io/propSAPT/)

## Documentation

Full documentation is available at: **https://btyrcha.github.io/propSAPT/**

## Installation

1. Clone the repository.
2. Go to its root directory.
3. Create a conda environment with all requirements by running:
   ```
   conda env create -f prop-sapt.yaml
   ```
4. Activate the environment.
   ```
   conda activate prop-sapt
   ```
5. Install the development package by running:
   ```
   pip install -e . --config-settings editable_mode=compat
   ```
   while being in the repo's root directory.
6. Check it out by running some scripts from `examples/`!


## Implementation

Implementation is based on [Psi4NumPy](https://github.com/psi4/psi4numpy)-like style, i.e. using [Psi4](https://github.com/psi4/psi4)s computational kernel for SCF and evaluation of intergrals followed by exporting those quantities to NumPy arrays.
The methods implemented are utilising density fitting approximation in MO basis.
