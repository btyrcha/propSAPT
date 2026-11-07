# AI Usage Log

This file is a running log of how AI coding assistance (Claude Code) has been used
on the propSAPT project. It exists to support accurate disclosure of AI involvement
where required.

Entries are kept short: one line per pull request or task, under its date, saying what
was done and what was produced or changed.
The log is maintained going forward from its creation date; work predating it is not exhaustively reconstructed.

## Summary

The short version, for disclosures; the log below has the detail.

- **Tools.** Claude Code (Anthropic), used as a coding assistant, and GitHub
  Copilot code review on some pull requests.
- **What the AI assistance did.** Implementation of new features and refactoring, tests,
  documentation, CI and packaging, and reviews of code written by the authors. It also
  diagnosed numerical discrepancies, and implemented corrections to the
  orbital-relaxation terms of the second-order properties.

## Log

### 2026-07-08

- Set up this log, with logging instructions in `CLAUDE.md` and `CONTRIBUTING.md`.

### 2026-09-03

- Reviewed the four-point finite-difference formula of the finite-field driver and added unit and He-H2 tests for it, with converged four-point reference values.

### 2026-09-04

- Regenerated the dipole regression values from the analytic path (to 1e-8) and added a per-term analytic vs finite-field dipole test, with the known gaps marked xfail.

### 2026-09-08

- Measured the new `[W, C - C^dagger]` terms of `x2_ind,r` against finite-field derivatives on four dimers (an improvement, not the complete fix) and updated the property tests.

### 2026-09-10

- Built the finite-field validation harness in `validation/`: code states by git ref, cached finite-field references, an SCF cross-check and test-value output, with tests.

### 2026-09-11

- Traced the first-order exchange finite-field gap to a JK vs `DF_BASIS_SAPT` integral mismatch; added matched-integral comparisons to the harness and tests, built the `omega_exch*` potentials from JK builds, and updated the regression values.
- Re-validated that branch on Psi4 1.11 (test suite and harness) and reported which documented claims no longer held.

### 2026-09-14

- Reviewed the package options module over several rounds, wrote its docstrings, and fixed the remaining review findings and typos in the files it touched.

### 2026-09-18

- Reviewed the selective `reset_propsapt_options` (bugs reported, fixed by the author), wrote its docstrings and narrowed its argument type.

### 2026-09-21

- Added an H2 dimer geometry to the validation harness, with cached finite-field references.

### 2026-09-22

- Added second-order partial sums to the harness, and corrected the orbital-relaxation terms of `x2_ind,r` and of the exchange-induction amplitudes, checked against finite field.

### 2026-09-24

- Located twelve spurious terms in `x2_exch-disp_S2`, removed the uncoupled-`x2_ind` reconstruction from the harness, and updated the property and density tests for the corrected property path.

### 2026-09-30

- Roadmap 3.4: the `prop-sapt run` command-line runner with YAML/JSON configuration, basis overrides and `basis_dirs`, and `prop-sapt basis fetch` from ccRepo or the Basis Set Exchange, with tests and docs.

### 2026-10-04

- Prepared the 0.3.0 release: one version source in `prop_sapt/_version.py`, and `CHANGELOG.md`.

### 2026-10-05

- Roadmap Tier 0 sweep: reviewed the `compute_jk` fix, fixed `save_cube` filenames, the ESP cubes, `calc_property` with an AO matrix, empty orbital blocks and `get_exch_disp_density`, and rejected non-singlet monomers; tests and the 0.3.1 release entry.

### 2026-10-07

- Roadmap 1.2: point-evaluation API (`prop_sapt/evaluate.py`), now used by `make_cube`, with tests and docs.
- Roadmap 1.1: fragment population operators (`prop_sapt/partition.py`) with tests and docs; fixed the Sphinx warnings so the docs build with `-W`.
- Reviewed `ROADMAP.md` against the source and restructured it, removing or rewriting several items at the user's request.

### 2026-10-08

- Roadmap 1.3 and 1.4: self-describing density bundles (`prop_sapt/artifacts.py`) and fragment metadata (`prop_sapt/fragments.py`), used by `calc_densities` and the runner, with tests and docs.

### 2026-10-14

- Roadmap 3.2: symmetric second-order exchange densities and invariant checks (`prop_sapt/checks.py`: symmetry, partition of unity, charge closure), with tests and docs.

### 2026-10-26

- Renamed `complete` to `partition_of_unity` in the fragment operators and documented the tolerance helper.
- Roadmap 2.7: uncoupled property path (`response=True|False|"both"`) with new `x2_exch-ind` terms checked against finite field, with tests and docs.

### 2026-10-29

- Roadmap 3.1: `Cube` as a dataclass with integration and arithmetic, and faster cube file reading and writing, with tests and docs.
- Roadmap 3.8: theta amplitudes cached on `Dimer` and freed after `calc_densities`, with tests and timings against `main`.

### 2026-10-30

- Analysed the computational scaling of energies, properties and densities; found contractions with a needlessly high scaling and a bug in the exch-disp S^2 density of monomer B.
- Fixed that exch-disp S^2 density bug, with tests.
- Tried cached dynamic-programming contraction paths on `feature/contraction-scaling` (not merged) and built a cluster benchmark for it.

### 2026-11-02

- Analysed the cluster benchmark of the contraction paths: no speed-up; CPSCF takes 91-94% of the time.
- Batched the CPSCF solves of the densities and the dipole, with tests, and analysed its cluster benchmark (runs 36-41% shorter).
- Roadmap 2.6: dropped frozen-core support, keeping a guard that rejects `FREEZE_CORE`.

### 2026-11-03

- Roadmap 3.5 and 3.7: API cleanups (`Dimer` memory and `SAVE_JK`, `calc_sapt_energy` saves a CSV, argument names, imports, return types), with tests.
- Roadmap 3.6: shared test fixture in `tests/conftest.py` and an MO-basis densities test; fixed `Dimer` and `make_cube` using the global `BASIS` option instead of the dimer's basis, and made density bundles record the dimer's own settings.
- Release packaging and CI: `pyproject.toml`, `CITATION.cff`, CI and release workflows, Python 3.12 minimum.
- Release docs pass: README, installation page, usage guide (known limitations, uncoupled dispersion, finite-field wrappers), examples and CONTRIBUTING.
- Added the summary to this log and changed it to one line per PR or task.
- Tested DF CPSCF with DIRECT monomer SCFs on the water dimer (aug-cc-pVTZ HF CPSCF 505 s to 7 s, IND2,r within 2.3e-9 Eh, densities within 7e-6) and prepared a cluster benchmark for production systems.
- Implemented the `cpscf_type` and `df_basis_cpscf` options (a DF or DIRECT J/K object for the CPSCF of the densities and properties, energies unchanged), with tests, docs, the changelog and the example config.

### 2026-11-04

- Roadmap 3.9: removed the unreachable non-density-fitted branch of `sinfinity` and commented-out code.
- Bumped the GitHub Actions to their Node 24 major versions.
- Simplified this log to one line per PR or task.
- Release PR for 0.4.0: the version, the changelog section with a highlights summary, and the roadmap's release items moved to Completed.
- Public release of 0.4.0: the one-commit export of the development repository and its checks, the launch checklist, and the Zenodo DOI in `CITATION.cff` and the README.

### 2026-11-05

- Ported the `cpscf_type` work from the development repository onto the public `main`.

### 2026-11-06

- Black check in CI and CONTRIBUTING and black in `prop-sapt.yaml`; black formatting of the five files on `main` that were not, and LF line endings everywhere (`helper_SAPT_DF.py` was CRLF) kept by a `.gitattributes`.
- Analysed the cluster benchmark of `cpscf_type` (HF energy plus densities 68-70% shorter, densities within about 1e-5); shortened its changelog entry and usage guide paragraph to the user-facing change; marked the default fitting basis being read from the global options at solve time with a TODO.
