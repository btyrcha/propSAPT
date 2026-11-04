# Changelog

Notable changes to propSAPT, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0, a minor
version may change the API and the numerical results.

The version is defined in `prop_sapt/_version.py`. Numbers in parentheses are pull
requests of the private repository propSAPT was developed in before it was published.

## [Unreleased]

### Added

- The Zenodo DOI of the software in `CITATION.cff`, and a DOI badge and the DOIs in the
  README.

## [0.4.0] - 2026-11-04

The first public release. Highlights since 0.3.1:

- The second-order exchange density matrices are now symmetric, and the S^2
  exchange-dispersion density of monomer B is fixed for a monomer A with more than one
  occupied orbital (see Results changed).
- Densities, orbitals and the ESP at arbitrary points (`prop_sapt.evaluate`), fragment
  populations, charges and dipoles on quadrature grids (`prop_sapt.partition`), and
  self-describing density bundles (`save_densities`, `load_densities`).
- Uncoupled properties with `calc_property(response=False)`, and checks that warn about
  asymmetric density matrices or fragment weights that do not sum to 1
  (`prop_sapt.checks`).
- The CPSCF equations of the density components, and of the three dipole axes, are
  solved together.
- `pyproject.toml` (Python 3.12 or later), GitHub releases with the sdist and wheel, and
  `CITATION.cff`.

The known limitations are listed in the
[usage guide](https://btyrcha.github.io/propSAPT/usage.html#known-limitations).

### Results changed

The second-order exchange density matrices (`exch-ind_S2`, `exch-ind`, `exch-disp_S2`,
and the `Ind`, `Disp` and `total` sums that contain them) are now symmetric. Their
occupied-occupied and virtual-virtual blocks held a product of the exchange amplitudes
and the CPSCF (or dispersion) amplitudes in one order only. The field derivative they
are read off fixes only the symmetric part of such a block, so the matrices carried an
antisymmetric part of up to about 10% of their largest element (He-H2, aug-cc-pVDZ). The
builders now keep the symmetric part, `0.5 * (X + X.T)` of each block, at no extra cost.
Densities, cubes, fragment charges and dipoles, forces and every trace with a symmetric
operator are unchanged. Eigendecompositions (natural orbitals) and one-triangle formats
(fchk) of these matrices change. Bundles written by 0.4.0.dev0 before 2026-10-14 hold
the old matrices: symmetrize them with `0.5 * (D + D.T)` before such uses.

The S^2 exchange-dispersion density of monomer B, `exch-disp_S2_B`, was wrong whenever
monomer A has more than one occupied orbital. The same goes for the sums that contain it:
`Disp_B` and `total_B`, and `exch-disp_S2`, `Disp` and `total`. Three of its terms summed
monomer A's occupied orbitals separately in the dispersion and theta amplitudes, instead
of contracting them, so they also depended on the choice of occupied orbitals. Water
dimer, aug-cc-pVDZ: the X dipole of `exch-disp_S2` was off by 3.1e-5 a.u. (2.3%).
Monomer A, the other components and dimers whose monomer A has one occupied orbital
(He-H2) are unchanged. Versions 0.3.0 and 0.3.1 have the error: recompute densities,
bundles and cubes made with them. On the water dimer every density component now
reproduces its property term to 1e-9.

### Added

- `Dimer.psi4_options` and `Dimer.fitting_bases`: the Psi4 options changed from their
  defaults when the dimer was built, and its fitting basis sets as built, by option
  ("DF_BASIS_SCF", "DF_BASIS_SAPT"). Density bundles record them.
- `Cube` arithmetic and integrals: `rho - rho_pol`, `rho_A + rho_B`, `0.5 * rho`,
  `rho / 2` and `-rho` on cubes of the same grid (through `add_cubes` and
  `subtract_cubes`); `Cube.voxel_volume`; and `Cube.integrate(weight=None)`, the sum over
  voxels times the voxel volume, with an optional weight cube or array (stacked weights
  are integrated at once).
- `calc_property(response=...)`, as in `calc_sapt_energy`: True (the default, with the
  same output as before), False or "both". The property is the field derivative of the
  energy with the same `response`. False uses the uncoupled induction amplitudes, with
  no CPSCF solves for them or their derivatives, and gives the uncoupled
  exchange-induction terms `x2_exch-ind_S2` and `x2_exch-ind` next to `x2_ind`; the
  totals then take the uncoupled terms. The new terms agree with the finite-field
  derivative of `EXCH-IND2(S^2)` and `EXCH-IND2` to 1.2e-11 a.u. on He-H2. Also in
  `calc_induced_dipole`, `perform_property_contractions` and the runner's `property`
  task.
- `prop_sapt.checks`: invariant checks, which warn (`InvariantWarning`, also written into
  the Psi4 output file) instead of stopping a calculation. `check_symmetric` checks that
  density matrices are symmetric within the new option `check_symmetry_atol` (1e-10;
  None turns the check off). `calc_density_matrix` runs it on every component and
  `save_densities` on every matrix it saves. `check_partition_of_unity` checks that
  fragment weights sum to 1 wherever the basis functions reach (option
  `check_partition_atol`, 1e-8), and `check_charge_closure` that fragment populations add
  up to the total (option `check_closure_atol`, 1e-6 electrons). Warnings are attributed
  to the first caller outside propSAPT.
- `FragmentOperators.from_grid(partition_of_unity=True)` and
  `Dimer.fragment_operators(partition_of_unity=True)`, the default, check that the
  fragment weights sum to 1, and give the operators the analytic overlap matrix
  (`FragmentOperators(ao_overlap=...)`, `.ao_overlap`): every `population` and `charge`
  call then checks that the fragments account for Tr(D S), and `closure_error` gives the
  electrons a grid misses for a density. `partition_of_unity=False` checks nothing, for
  fragments that do not cover the whole system.
- `prop_sapt.evaluate`: `evaluate_density`, `evaluate_orbital`, `evaluate_esp` and
  `evaluate_basis_functions` evaluate AO matrices at arbitrary points (bohr, shape
  `(..., 3)`), for quadrature grids, lines or planes. Basis function values come from
  Psi4's batched evaluator, in blocks of `chunk_size` points (new option
  `point_eval_chunk_size`, default 2000), and are shared by all matrices in one call.
  Orbitals are the columns of the MO coefficients, as in Psi4's `Ca()`.
- `Dimer.evaluate_density`, `Dimer.evaluate_orbital` and `Dimer.evaluate_esp`, the same
  in the dimer basis set.
- `prop_sapt.partition`: fragment population operators. `FragmentOperators.from_grid`
  computes, in one quadrature pass, the weighted-overlap matrices
  `W_X = integral(w_X phi phi)` of any fragment weights on any grid (and the
  first-moment matrices), after which `population` (`Tr(D W_X)`), `charge`
  (`-Tr(D W_X)`) and `dipole` of any number of density matrices are traces.
  `molecular_grid` builds Psi4's molecular quadrature grid (new options
  `quadrature_radial_points` 75, `quadrature_spherical_points` 590 and
  `quadrature_bs_radius_alpha` 2.0, twice Psi4's DFT radial extent, which diffuse basis
  sets need). `becke_fragment_weights` and `hirshfeld_fragment_weights` give Becke and
  monomer-density Hirshfeld weights; weights from elsewhere (e.g. horton_part LISA/MBIS on
  its own grid) are passed as arrays.
- `Dimer.fragment_operators("becke" | "hirshfeld" | weights)`, the same for monomers A
  and B in the dimer basis set.
- `prop_sapt.fragments`: `DimerFragments` holds the atoms of a dimer geometry (elements,
  ghost flags, coordinates in bohr) and its fragments (monomers A and B, then any extra
  fragments, with atom indices, charges and multiplicities), with `check_same_system`,
  `to_dict` and `from_dict`. `parse_fragments` reads them from a geometry string or a
  `.p4m` file without any calculation, in the frame the `Dimer` uses.
  `Dimer.fragments` holds them for a dimer; `Dimer.functional`, `Dimer.grac_A` and
  `Dimer.grac_B` keep the DFT settings.
- `prop_sapt.artifacts`: self-describing density bundles. `save_densities` writes AO
  density matrices of a dimer, the monomer wavefunctions and their metadata into one
  `.npz` file, with nothing pickled. The metadata records the conventions (AO basis,
  Psi4 AO order, one spin: the total density is twice the matrix), the method, the basis
  set (name, composition, shell structure and a SHA-256 digest), the fitting bases, the
  Psi4 and propSAPT options, the versions, and the geometry with its fragments.
  `load_densities` (also `prop_sapt.load_densities`) reads a bundle into a `DensitySet`
  (`matrices`, `metadata`, `fragments`, `total`, `wavefunction_data`, `wavefunction`),
  checking its format, conventions and matrix shapes, and on request that it belongs to
  a dimer, a geometry or a basis set; anything else raises `ArtifactError`.
- `prop_sapt.DENSITY_COMPONENTS`, the component keys of `calc_density_matrix` in order.
- `calc_densities(matrix_format=..., metadata=...)`, the `matrix_format` setting of the
  runner's `densities` task, and `RunConfig.applied_basis_overrides`. The runner's
  bundle records the run configuration and the basis overrides applied, and
  `propsapt_run.json` the fragments of the geometry.
- `Dimer.clear_theta_amplitudes()` frees the theta amplitudes of the exchange density
  matrices, which `calc_density_matrix` now keeps on the dimer (see Changed).
  `calc_densities` calls it itself.
- `Dimer.cpscf_batch(monomer, perturbations)` solves the CPSCF equations of several
  perturbations of a monomer in one solver run, to the amplitudes `Dimer.cpscf` gives
  each. `Dimer.store_theta_amplitudes` keeps theta amplitudes for
  `Dimer.get_theta_amplitudes`.
- `CITATION.cff`, with the JCTC paper as the preferred citation.
- Releases are GitHub releases, created from version tags, with the sdist and wheel
  attached and the notes from this changelog.

### Changed

- `Cube` is a keyword-only dataclass. `n_x`, `n_y`, `n_z` and `n_atoms` are optional and
  read off `volumetric_data` and `atoms`; when given they must match (`ValueError`), and
  `save` checks again. A missing required field raises `TypeError` instead of
  `ValueError`. The origin, grid vectors, data and atom coordinates are stored as float
  NumPy arrays. `Cube.from_file` is a classmethod, `Cube.from_file(path)`. Cubes compare
  by identity, as before, and print as a one-line summary.
- `add_cubes`, `subtract_cubes`, `cosine_similarity` and the new operators also check
  that the grids have the same numbers of points. They accept origins and grid vectors
  that agree within the 6 decimals of a cube file (absolute 1e-6). Their results share
  no arrays with the operands.
- `read_cube_file` parses the data in one NumPy call, with any number of values per line:
  about 10x faster (1M voxels, 13 MB: 1.38 s to 0.13 s). `save_cube_file` writes with
  NumPy, about 2x faster (0.88 s to 0.47 s), into byte-identical files. Both accept
  `os.PathLike` paths.
- `calc_densities(save_matrices=True)` writes the bundle `densities.npz` instead of bare
  `delta_dm_<key>.npy` and `wfn_mon_<A/B>.npy` files. The first comment line of the
  cube files it writes says that they hold the total density (twice the one-spin matrix).
- `calc_densities` no longer accepts arbitrary keyword arguments: unknown ones raise
  `TypeError` instead of being ignored.

- `make_cube` evaluates densities and orbitals through `prop_sapt.evaluate`, a block of
  points at a time instead of one `compute_phi` call per voxel: about 5x faster for
  density cubes (water dimer, aug-cc-pVTZ). Cube values change only by rounding
  (relative 1e-15). It takes a `chunk_size` keyword.
- `calc_density_matrix` computes the theta amplitudes of the second-order exchange
  components (`exch-ind_S2`, `exch-ind`, `exch-disp_S2`) once, and keeps them, read-only,
  on the dimer for the other monomer. Each call used to compute the amplitudes of both
  monomers. `calc_densities` frees them when both monomers are done. It now solves 18
  CPSCF equations instead of 22: water dimer (aug-cc-pVTZ) 708 s to 629 s, He–H2
  (aug-cc-pVTZ) 12.8 s to 11.3 s. The matrices are unchanged (AO basis, within 4e-12
  between runs).
- `calc_density_matrix` solves the CPSCF equations of its components together, one
  solver run per monomer at a time: each iteration builds the J and K matrices of all
  unconverged equations in one call. `calc_densities` solves the same 16 equations in 4
  runs instead of 16, about twice as fast. Production settings (RHF, aug-cc-pVTZ,
  `scf_type direct`), SAPT energy and densities: water dimer (16 cores) 90 s to 53 s;
  IHB100 dimers (32 cores) with 391 basis functions 1034 s to 608 s, with 529 basis
  functions 2290 s to 1459 s. The matrices change only within the variation between runs
  (AO basis, 2e-9). The Psi4 output reports the solver runs ("CPSCF of monomer A for 6
  perturbations"), and the component timings no longer include CPSCF.
  `get_exch_ind_density`, `get_exch_ind_s2_density` and `get_exch_disp_s2_density` still
  solve their equations one at a time.
- `calc_property(mol, "dipole")` solves the CPSCF equations of the three axes together:
  12 equations in 4 solver runs instead of 12, about twice as fast (water dimer,
  cc-pVDZ: 17 s to 9 s). The values are unchanged (1e-16). `calc_property` with one
  operator matrix and `perform_property_contractions` solve as before: the equations of
  one operator belong to different monomers or need each other's solutions, so they
  cannot share a run.
- `Dimer` takes Psi4's memory (`psi4.set_memory`): its JK object gets Psi4's memory times
  `SCF_MEM_SAFETY_FACTOR` (0.75 by default), as Psi4's SCF gives its own, instead of a
  fixed 8 GB (in effect 64 GB, see Fixed). The runner no longer passes its `memory` to
  `Dimer`; it sets Psi4's memory, which the dimer then takes.
- `Dimer` turns `SAVE_JK` on for its monomer SCF calculations and restores the setting
  afterwards, so `save_jk: True` is no longer a required Psi4 option. The examples and
  the test fixture no longer set it.
- `calc_sapt_energy` saves its results to `sapt_results.csv` by default, as
  `calc_property` and the finite-field functions save theirs. `results=True` saves to
  the same file, `results=False` (or None) saves nothing. The name differs from the
  other defaults (`results.csv`, `results-ff.csv`).
- The public functions name the dimer argument `dimer`, as `calc_sapt_energy` already
  did: `calc_property`, `calc_induced_dipole`, `perform_property_contractions`,
  `calc_densities`, `calc_density_matrix` and `density_mo_to_ao` (it was `mol`). The
  component functions (`sapt_components`, `properties_components`,
  `densities_components`) name it `mol`; the `sapt_components` named it `dimer`. The
  operator of `perform_property_contractions` is `property_matrix`, no longer
  `property_matix`. Only calls by keyword change.
- The property components `calc_exch_ind2_resp_s2_property`,
  `calc_exch_ind2_resp_sinf_property`, `calc_exch_disp2_s2_property` and
  `calc_exch_disp2_sinf_property` return floats instead of one-element arrays, as the
  energy components do. `calc_property` gives the same frame as before (He-H2, within
  2e-13 of the previous code).
- `prop_sapt.properties_components.exch_disp2_sinf_terms` exports its three term
  functions, as the other term packages do, and every module imports `Dimer`
  relatively.
- `pyproject.toml` replaces `setup.py`. The package declares `requires-python >= 3.12`,
  the version it is tested with. Optional extras: `test`, `docs`, and `bse`
  (Basis Set Exchange, for `prop-sapt basis fetch bse`). Psi4 is still installed with
  conda. `docs/requirements.txt` is replaced by the `docs` extra.
- `prop-sapt.yaml` no longer installs the unused `qcdb` from the `psi4/label/qcdb`
  channel, and adds `pytest-cov` and `build`.
- Docs and README for the public release:
  - The installation page gives Python 3.12 or later, the repository URL and the install
    from a release tag.
  - The README is short: install, links to the docs and the examples, the known
    limitations and the citation.
  - The usage guide has a "Known limitations" section, to which `calc_property`'s
    docstring points: the S^inf `x2_exch-disp` property is not yet the derivative of its
    energy (issue #1).
  - The docs, the README and `calc_sapt_energy`'s docstring say that the dispersion is
    uncoupled only: the SAPT0 dispersion with RHF. With RKS it is not the coupled
    dispersion that Psi4's SAPT(DFT) uses by default.
  - The finite-field wrappers are described in the README and the usage guide:
    `finite_field_sapt` for the SAPT components, and
    `prop_sapt.finite_field.finite_field_property` for any Psi4 energy method, with
    `cp_correction=True` for the interaction-induced dipole. The latter's docstring now
    covers `cp_correction` and `results`.

### Deprecated

- `calc_densities(matrix_format="npy")` (bare `.npy` files),
  `calc_densities(compress_matrices=...)` and the runner setting
  `tasks.densities.compress_matrices`, with a `FutureWarning`: use `matrix_format`
  ("npz", the default, or "npy").
- `Dimer(memory=...)`, with a `FutureWarning`: set `psi4.set_memory` instead. Until it is
  removed, the value (GB) still sizes the JK object.

### Removed

- Frozen core is no longer planned: propSAPT is all-electron. `Dimer` still rejects
  `FREEZE_CORE`, now with a `ValueError` instead of `NotImplementedError`. The unused
  attributes `Dimer.is_frozen_core` (always False), `Dimer.nfrozen_A` and
  `Dimer.nfrozen_B` (always 0) are gone.
- The unused attribute `Dimer.is_density_fitted` (always True), together with the
  unreachable non-density-fitted branch of `sinfinity` that raised
  `NotImplementedError`: propSAPT is density-fitted throughout.

### Fixed

- `Dimer` rejected `psi4.set_options({"freeze_core": False})` as frozen core, because
  Psi4 stores it as "0" rather than "FALSE".
- `Dimer` gave its JK object its memory in bytes, but Psi4 counts that memory in doubles,
  so the object could take eight times the memory meant: 64 GB by default, and 8 times
  the runner's `memory`. The results are unchanged.
- `calc_sapt_energy(results=True)` failed, and its docstring named a default file
  (`results.csv`) that it did not write.
- The error of `calc_sapt_energy` for an invalid `response` named only True and False;
  "both" is valid too.
- Typos in the Psi4 output of the memory trace ("Funciton", "meamory").
- `add_cubes` and `subtract_cubes` returned cubes that shared the origin, grid vectors and
  atom coordinates of the first cube, so changing one in place changed the other.
- Cubes on grids with different numbers of points were combined whenever NumPy could
  broadcast their data, e.g. (1, n_y, n_z) with (n_x, n_y, n_z).
- `read_cube_file` built wrong cubes or failed with a reshape error for cube files of
  several orbitals or in angstrom, and failed on a fifth field (values per voxel) after
  the origin. Unsupported variants now raise `NotImplementedError`, and a file with too
  few or too many values a `ValueError` that names it.
- `calc_densities(save_matrices=True, compress_matrices=True)` in the AO basis added the
  monomer wavefunctions to the returned dictionary, so `save_cubes=True` in the same
  call failed.
- `calc_densities(orbital_basis="ao")` (lowercase) returned no summed components and,
  when saving, transformed the AO matrices as if they were in the MO basis.
- `examples/make_cubes_from_mo_matrices.py` transformed the saved matrices from the MO to
  the AO basis, but they are saved in the AO basis already; it now reads the bundle.
- `Dimer.get_psi4_basisset`, `Dimer.get_psi4_wavefunction` and `Dimer.make_cube` used
  the basis set the `BASIS` option named at the time of the call, not the dimer's own.
  After the option changed (another dimer, another calculation), `make_cube` and
  `save_cube` (and so the cubes of `calc_densities` and the runner), `evaluate_density`,
  `evaluate_orbital`, `evaluate_esp` and `fragment_operators` failed with a shape error,
  or used the wrong basis set if the two had the same size.
- Density bundles recorded the fitting bases and the Psi4 options set when the bundle
  was saved, not those the dimer was built with. The fitting bases were the option
  values, None for a Psi4 default; they are now the basis sets as built (their
  `blend()`, e.g. "AUG-CC-PVTZ-JKFIT + DEF2-UNIVERSAL-JKFIT" for the default JKFIT
  basis of He-H2 in aug-cc-pVTZ).

- `make_cube` accepts an orbital as a column of shape `(nbf, 1)`, as in
  `examples/visualise_orbital.ipynb`, again; it raised since the vectorized evaluation
  of 0.3.0.
- API docs: the keyword arguments of `calc_sapt_energy` and `Cube`, and the component
  list of `calc_sapt_energy`, rendered as plain text; the force formulas in
  `prop_sapt.forces` broke the reStructuredText markup. The docs now build without
  warnings (`sphinx-build -W`).
- Docs examples that did not run:
  - The Quick Start set no output file or basis set, and used an undefined operator.
  - The finite-field example passed `direction=` instead of a property.
  - The force and cube examples used `dimer.mol_A` and `dimer.mol`, which `Dimer` does
    not have.

  `prop_sapt.options` was missing from the API reference, and its option list did not
  render. The `cubes.make_cube` docstring now lists all its keyword arguments.
- `examples/sapt_example.py` and `examples/finite_field_sapt.py` failed before the first
  calculation: Psi4 has no aug-cc-pVQZ-JKFIT set for He, so `DF_BASIS_SCF` is now left to
  Psi4's default. `finite_field_sapt.py` labelled its relative errors with the wrong sign
  and printed -100% for components that are zero.

## [0.3.1] - 2026-10-05

Energies, properties and density matrices are unchanged from 0.3.0.

### Fixed

- `calc_property` with an AO matrix as `prop` raised under NumPy 2 ("The truth
  value of an array ... is ambiguous") before computing anything. It now runs.
- `"esp"` cubes, added in 0.3.0, held the density contraction instead of a
  potential. They now hold the electrostatic potential of the density matrix,
  from analytic potential integrals.
- `Dimer.save_cube` paired a list of cubes with the characters of a single
  filename, and dropped cubes when given fewer filenames than matrices. It now
  needs one filename per matrix, checked before the grid is evaluated, and
  accepts a one-element list and `os.PathLike` filenames.
- `compute_jk` returns `nbf x nbf` zero J/K for an empty orbital block instead of
  `nmo x nmo`, and `Dimer.s()` returns correctly shaped empty overlap blocks.
  Neither path is reached with RHF/RKS references.

### Added

- `esp_nuclei=True` in `make_cube` adds the potential of the nuclei to `"esp"`
  cubes, for total densities.

### Changed

- A monomer that is not a singlet is rejected with a `ValueError` before any SCF
  runs. Before, Psi4 stopped with its own error when it reached that monomer's
  SCF.

### Removed

- `get_exch_disp_density` is no longer exported from
  `prop_sapt.densities_components`, and it raises `NotImplementedError` at once
  instead of after building the S^inf amplitudes. It always raised: the S^inf
  exchange-dispersion density is not implemented.

## [0.3.0] - 2026-10-04

### Results changed

Second-order interaction-induced properties and densities differ from 0.2.0 and
from earlier commits on `main`. Their orbital-relaxation terms were rederived so
that each analytic term is the finite-field derivative of its SAPT energy.
Results computed with earlier versions should be recomputed before they are
compared with new ones.

- `x2_ind`, `x2_ind,r`: added the `[W, C - C^dagger]` relaxation terms. `x2_ind`
  gets only the one-electron part; `x2_ind,r` gets the one- and two-electron parts
  (#25, #29).
- `x2_exch-ind,r` (S^2 and S^inf): added the two-electron orbital-Hessian
  derivative to the coupled amplitudes `u_ra`/`u_sb`, and fixed double-counted
  denominators in those amplitudes (#29).
- `x2_disp` and `x2_exch-disp_S2`: fixed the relaxation terms, removing spurious
  terms from `x2_exch-disp_S2` (#29).
- `x2_exch-disp` (S^inf): property rederived (#13).
- Second-order densities (induction, dispersion, exchange-induction S^2 and
  S^inf, exchange-dispersion S^2) now include the same relaxation as the
  properties and reproduce them (#31).
- First-order exchange: the `omega_exch*` potentials are built by JK on the
  integrals the energy uses. This removes a disagreement of up to 1.9e-6 a.u.
  between analytic and finite-field first-order properties (#26). The S^2 and
  S^inf first-order exchange property terms were also fixed (#3).

### Added

- Command-line runner `prop-sapt run` (also `prop_sapt.run`) driven by a YAML or
  JSON configuration, writing a `propsapt_run.json` provenance record.
  Per-element basis overrides and `basis_dirs` for basis sets missing from Psi4,
  and `prop-sapt basis fetch` to write them from ccRepo or the Basis Set
  Exchange (#30).
- Package-wide options: `set_propsapt_options`, `get_propsapt_options` and
  `reset_propsapt_options`, which can reset selected keys (#27, #28).
- `prop_sapt.__version__`.
- Second-order properties: polarization terms (#4), exchange-induction and
  exchange-dispersion in S^2 (#10) and S^inf (#11, #13), and the split of
  `x2_ind,r` and `x2_disp` into monomer A and B parts (#15).
- Second-order densities: polarization terms (#5), exchange-induction and
  exchange-dispersion in S^2, and exchange-induction in S^inf (#12, #13).
- `calc_sapt_energy`: SAPT0 energies with S^inf exchange, also with a DFT
  reference (uncoupled dispersion only) (#8); `response="both"` returns
  uncoupled and coupled induction together (#25).
- `finite_field_sapt` with two- and four-point formulas (#10, #24), and
  finite-field properties from any Psi4 reference method (#17).
- `calc_densities` for the dimer and both monomers at once, saving cubes and
  density matrices, optionally compressed into one `.npz` archive (#16, #17, #24).
- Hellmann-Feynman forces from AO density matrices: `calc_force_at_nuc` and
  `calc_force_nuc_nuc` in `prop_sapt.forces` (#18).
- Cube utilities: several cubes from one call (#2), ESP cubes and reuse of a grid
  through `grid=` (#18), `add_cubes`, `Cube.copy`, `Cube.moved_charge`,
  `cosine_similarity` (#11, #16) and `translate_cube` (#18).
- First-order monomer contributions as columns in the property results (#7).
- Fragments beyond the second are used as ghost atoms (#11).
- Sphinx documentation published to GitHub Pages (#17), and an internal validation
  harness comparing analytic properties with finite field (#25, #26).
- `prop-sapt.yaml` conda environment. Reference values are validated with
  Psi4 1.11.

### Changed

- `make_cube` evaluates all matrices in one vectorized pass and writes cube files
  faster (#18, #24).
- `CalcTimer` measures wall-clock instead of CPU time (#18).
- `Cube` requires all of its attributes at construction (#11).
- Finite-field output filenames include the field strength (#15).
- Frozen core raises `NotImplementedError` instead of running (#11).

### Known issues

- `x2_exch-disp` (S^inf) property differs from its finite-field derivative
  (7.2e-5 a.u. for the He-H2 test system).
- The S^inf exchange-dispersion density is not implemented. Totals from
  `calc_densities` use the S^2 exchange densities at second order, although the
  S^inf exchange-induction density is available as its own component.
- Frozen core and unrestricted references are not implemented.

## [0.2.0] - 2024-10-23

Last version before this changelog was started.

## [0.1.0] - 2024-09-12

First packaged version.
