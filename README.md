# interaction-induced

Python package for calculations of first-order interaction-induced properties and changes in monomers density matrices in the spirit of SAPT.

## Installation

1. Clone the repository.
2. Go to its root directory.
3. Create a conda environment with all requirements by running:
   ```
   conda env create -f inter-ind.yaml
   ```
4. Activate the environment.
   ```
   conda activate inter-ind
   ```
5. Install the devolopment package by running:
   ```
   pip install -e .
   ```
   while being in the repos root directory.
6. Check it out by running some scripts form `examples/`!


## Implementation

Implementation is based on [Psi4NumPy](https://github.com/psi4/psi4numpy)-like style, i.e. using [Psi4](https://github.com/psi4/psi4)s computational kernel for SCF and evaluation of intergrals followed by exporting those quantities to NumPy arrays.
The methods implemented are utilising density fitting approximation in MO basis.
