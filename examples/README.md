# propSAPT examples

Run the scripts from this directory, in the `prop-sapt` conda environment. Each one sets its
geometry, memory, threads and basis sets at the top (`BASIS`, `DF_BASIS`). The small
examples use aug-cc-pVDZ. `sapt_example.py` and `finite_field_sapt.py` compare with Psi4 and
finite field, and use aug-cc-pVTZ with aug-cc-pVQZ fitting sets.

## Scripts

- **`example.py`**: interaction-induced dipole moment and density matrices of Ne–H2.
- **`sapt_example.py`**: the SAPT energy components of He₂, compared with Psi4's own SAPT(DFT)
  run with the HF functional.
- **`finite_field_sapt.py`**: the analytic interaction-induced dipole of He–H2 next to the
  finite-field SAPT derivative, term by term.
- **`water_dimer.py`**: the dipole and the density cubes of the water dimer.
- **`save_mo_matrices.py`**: the density matrices of Ne–H2 computed in the MO basis and saved, in
  the AO basis, into one bundle (`matrices_MO/densities.npz`).
- **`make_cubes_from_mo_matrices.py`**: cube files from that bundle, read back with
  `load_densities`. Run `save_mo_matrices.py` first.
- **`read_subtract_save.py`**: cube arithmetic on the cubes that `water_dimer.py` writes to
  `water-dimer-cubes/`. Run `water_dimer.py` first.
- **`run_config.yaml`**: a commented configuration for the `prop-sapt run` command-line runner.

## Notebooks

- **`visualise_density.ipynb`**: interaction-induced densities from the cubes of `water_dimer.py`.
- **`visualise_orbital.ipynb`**: molecular orbitals from cube files.

The notebooks use the visualisation packages of `prop-sapt.yaml` (`fortecubeview`,
`pythreejs`, `ipywidgets`).
