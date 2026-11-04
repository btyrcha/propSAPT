# propSAPT development roadmap

Feature and issue candidates, first collected on 2026-07-07 from the downstream
`densities` workflow repository and an audit of the package source, and last
checked against the source on 2026-10-07. Science features and workflow
leverage come first. Sections: Tiers 1–3 by leverage, each ordered by
priority; finished items are in `CHANGELOG.md`. Item numbers are kept
when items finish or are dropped,
since commits and the changelog refer to them.

---

## Tier 1

All Tier 1 items are done, see `CHANGELOG.md`.

---

## Tier 2 — capability growth

### 2.2 Built-in supermolecular reference deltas

**Feature.** A companion to `calc_densities` producing benchmark deltas in the
same conventions and artifact format (the bundle of 1.3: `save_densities` takes any
AO matrices of a dimer):

```python
calc_supermolecular_delta(dimer, method="hf" | "ccsd" | "dft", ...)
# -> delta_dm_hf / delta_dm_ccsd / delta_dm_<functional>
```

**Motivation.** `delta_dm_ccsd` is the standard reference the propSAPT
densities are compared against — it already appears in the hardcoded analysis
tables downstream. Producing it currently takes four scripts
(`run_ccsd_dimer_densities.py`, `run_ccsd_monomer_densities.py`,
`calc_delta_dm_matrices.py`, `calc_delta_dm_ccsd_from_fchk.py`) and a detour
through fchk files read back with `iodata` in a second conda env, just to
subtract three matrices. The runner's `monomer_references` task already writes
the monomer wavefunctions as fchk files, but the dimer side and the
subtraction are still downstream.

**Effort.** Small-medium (the pieces exist; this is consolidation plus
conventions).

### 2.3 Finish S^inf second-order exchange densities

**Feature.** Complete the S-infinity versions of the second-order
exchange-induction and exchange-dispersion *density* corrections and make the
approximation an explicit argument:

```python
calc_densities(dimer, exch_order="S2" | "Sinf")
```

with both variants keyed unambiguously in the output.

**Motivation.** `calc_density_matrix` (`densities.py`) currently sums
`exch-ind_S2` / `exch-disp_S2` into the totals with `NOTE: change into Sinf
version when ready` comments (the S^inf exchange-induction density already
exists; exchange-dispersion does not, and `get_exch_disp_density` raises
`NotImplementedError`). This is both a physics milestone and API hygiene:
today "total" silently means "total within S^2 second-order exchange".

The same gap shows in the S^inf exchange-dispersion *property*: `x2_exch-disp`
is not the field derivative of `EXCH-DISP2` (7.2e-5 a.u. off finite field for
He–H2, issue #1; strict xfail in `tests/test_calc_property.py`). It is a known
limitation in the docs until fixed.

**Effort.** Large (theory + implementation + tests), but already on the
implicit roadmap.

### 2.4 GRAC automation for SAPT(DFT)

**Feature.** Compute and apply GRAC shifts inside `Dimer` when given monomer
ionization potentials:

```python
Dimer(geometry, reference="RKS", functional="pbe0", ip_A=..., ip_B=...)
# internally: shift_X = IP_X + eps_HOMO_X from the uncorrected monomer SCF
```

**Motivation.** Shifts can already be given (`Dimer(..., grac_A=..., grac_B=...)`,
the runner's `grac` section), but they are still computed outside: the
workflow runs separate supermolecular DFT calculations, then regex-parses
Psi4 output text for monomer HOMO energies (`compute_grac_shifts.py`
downstream). The package has the monomer wavefunctions in hand; the shift is
one subtraction away.

**Effort.** Small-medium (needs an uncorrected pre-SCF per monomer or a
two-pass init).

### 2.5 Moved charge and similarity of delta densities on matrices

**Feature.** Moved charge and cosine similarity defined on matrices or
quadrature grids (e.g. via `molecular_grid` and `evaluate_density`, or as
traces with the overlap metric), not only on `Cube` voxel data.

**Motivation.** Similarity metrics computed on cubes inherit voxel-grid error
(the cube route leaks up to 1.7e-2 electrons for Li+ ... benzene). ESP at
arbitrary points (`evaluate_esp`, analytic `"esp"` cubes) and fragment
populations and dipoles (`FragmentOperators`) are done; this item covers what
remains of the analysis layer.

**Effort.** Small-medium.

### 2.8 Finite-field parity with `calc_property`

**Feature.** Let `finite_field_sapt` accept an arbitrary AO-matrix operator,
as the analytic path already does.

**Motivation.** The FF entry point rejects `np.ndarray` operators
(`validate_property` in `finite_field/core.py`, "general external operator is
not implemented") while `calc_property` accepts them. Capability parity would
make finite-field the universal numerical check for any analytic property.

**Effort.** Small-medium.

### 2.9 More one-electron property operators

**Feature.** Built-in quadrupole (and higher multipole) interaction-induced
properties.

**Motivation.** Only the dipole is wired end-to-end (`calc_property`, via
`mints.ao_dipole()`); the pattern generalizes directly to `ao_quadrupole()`
and friends. Combines naturally with 2.8 (numerical validation).
`test_calc_property_unknown_name` uses `"quadrupole"`
as its unimplemented example and must change with this item.

**Effort.** Small per operator.

---

## Tier 3 — ergonomics, robustness, performance

### 3.4 Finite-field task for the runner

**Feature.** A `finite_field` task in `prop-sapt run`, so finite-field SAPT
dipoles come from the same configuration file as the analytic ones:

```yaml
tasks:
  property: {prop: dipole}
  finite_field: {prop: dipole, field_strength: 0.001, formula: two_point,
                 results: ff_results.csv}
```

It calls `finite_field_sapt` with the configuration's reference, functional
and GRAC shifts, writes into the output directory, and is listed in
`propsapt_run.json` like the other tasks.

**What needs changing.**

- The other tasks share one `Dimer`, built before the task loop
  (`build_dimer` in `run`). `finite_field_sapt` takes the geometry string
  instead and builds its own dimer for every field (`PERTURB_H` /
  `PERTURB_DIPOLE`; 6 dimers for the two-point formula, 12 for the
  four-point one). The task needs the geometry and the dimer settings, and the
  runner should not build the unperturbed dimer when `finite_field` is the
  only task.
- The per-field CSV files (`results-{axis}-{sign}{magnitude}.csv`, written by
  `calculate_perturbed_sapt_energies`) go to the current working directory,
  which is where the stray `results-X-+0.0001.csv` files in the repository
  root come from. They must go to the output directory, or become optional.
- Task settings: `prop` (only `"dipole"` until 2.8), `field_strength`,
  `formula`, `results`, and `response` passed on through `sapt_kwargs`.

**Motivation.** The finite-field derivative is the numerical check for every
analytic property term, but it runs only from Python
(`examples/finite_field_sapt.py` and an internal validation harness), outside the
runner's configuration, basis overrides and provenance record.

**Effort.** Small-medium.

### 3.6 Test-coverage gaps

- `tests/test_cubes.py` covers `Dimer.save_cube`, density, orbital and ESP cube
  values, and since 3.1 the `Cube` class, read/write, arithmetic, `moved_charge` and
  `cosine_similarity`. `translate_cube`, `calculate_isocontour` and `prepare_grid` have
  no tests of their own.
- No `test_utils.py` (`energy_printer`, `prepare_path`, `CalcTimer`,
  `trace_memory_peak`).
- The xfail placeholder `test_exch_disp_sinf_from_density_matrix`
  (`test_calc_density_matrix.py`) hardcodes 0.0 pending 2.3.

**Effort.** Small (plain NumPy and Python, no Psi4 runs beyond the shared fixture).
