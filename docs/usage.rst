Usage Guide
===========

This guide provides an overview of how to use prop-sapt for your calculations.

Basic Workflow
--------------

The typical workflow for using prop-sapt involves:

1. Define your dimer geometry
2. Create a Dimer object
3. Calculate SAPT energy decomposition
4. Compute interaction-induced properties and/or density matrices
5. Analyze and visualize results

Creating a Dimer
----------------

The fundamental object in prop-sapt is the ``Dimer`` class.
It is created by providing a geometry string in Psi4 format, which should include two fragments (for both monomers) separated by ``--``.
Set Psi4's output file, memory and basis sets first, as for any Psi4 calculation:

.. code-block:: python

   import psi4
   from prop_sapt import Dimer

   psi4.core.set_output_file("output.dat", False)
   psi4.set_memory("4 GB")
   psi4.set_options(
       {
           "basis": "aug-cc-pvdz",
           "df_basis_scf": "aug-cc-pvdz-jkfit",
           "df_basis_sapt": "aug-cc-pvdz-ri",
       }
   )

   # Define geometry in Psi4 format
   geometry = """
   0 1
   O  0.000000  0.000000  0.000000
   H  0.758602  0.000000  0.504284
   H  0.260455  0.000000 -0.872893
   --
   0 1
   O  3.000000  0.000000  0.000000
   H  3.758602  0.000000  0.504284
   H  3.260455  0.000000 -0.872893
   units angstrom
   """

   # Create dimer with optional parameters
   dimer = Dimer(geometry, reference="RHF")

The dimer runs the monomer SCF calculations with Psi4 in the dimer-centred basis set and
takes Psi4's memory setting (``Dimer(memory=...)`` is deprecated). The reference is
``"RHF"`` or ``"RKS"``. The dimer keeps the monomer JK objects that the CPSCF equations need
itself: the Psi4 option ``save_jk`` is not needed.

If your geometry string will include more fragments, they will be treated as ghost atoms.
You can specify midbonds and farbonds in this way.

The fragment structure is kept on the dimer as ``dimer.fragments``
(:class:`~prop_sapt.fragments.DimerFragments`): the atoms of monomers A and B and of any
extra fragments, their charges and multiplicities, the ghost atoms, and the coordinates in
bohr. :func:`~prop_sapt.fragments.parse_fragments` reads the same from a geometry string or
a :file:`.p4m` file, without running any calculation:

.. code-block:: python

   from prop_sapt.fragments import parse_fragments

   fragments = parse_fragments("geos/geo1.p4m")
   fragments.A.atoms, fragments.B.atoms     # e.g. (0, 1, 2), (3, 4, 5)
   fragments.A.charge, fragments.B.multiplicity
   fragments.extra                          # fragments after the second, e.g. midbonds

SAPT Energy Calculations
-------------------------

To perform the SAPT energy decomposition, SAPT0 for an ``RHF`` dimer and SAPT(DFT) for an
``RKS`` one:

.. code-block:: python

   from prop_sapt import calc_sapt_energy

   # Calculate all SAPT components
   energy_result = calc_sapt_energy(dimer)
   
   print(energy_result)

The result is a pandas Series of all SAPT energy components. It is also saved to
:file:`sapt_results.csv`; ``results="energies.csv"`` chooses another file and
``results=False`` saves none.

The dispersion is the *uncoupled* one: ``DISP2`` and ``EXCH-DISP2`` are built from the
monomer orbitals and orbital energies, without the response of the monomers to each other.

* With ``RHF``, that is the dispersion of SAPT0, as in Psi4's SAPT0.
* With ``RKS``, it is the uncoupled dispersion from the Kohn–Sham orbitals. Psi4's
  SAPT(DFT) uses the *coupled* dispersion by default, so its dispersion and total energies
  differ from propSAPT's. Compare with Psi4's uncoupled dispersion instead.

Coupled dispersion is not implemented (see :ref:`known-limitations`). The dispersion terms
of the properties and density matrices are the uncoupled ones too.

Property Calculations
---------------------

To calculate interaction-induced properties for a general one-electron operator:

.. code-block:: python

   from prop_sapt import calc_property
   import numpy as np

   # A one-electron operator in the dimer's AO basis, an (nbf, nbf) array
   property_matrix = np.array([...])

   # Calculate property contributions
   property_result = calc_property(dimer, property_matrix)
   
   print(property_result)

For predefined properties, you can use the convenience functions, e.g., to calculate the dipole moment:

.. code-block:: python

   from prop_sapt import calc_property

   dipole_result = calc_property(dimer, "dipole")

The property is the field derivative of the SAPT energy, and like
:func:`~prop_sapt.sapt_driver.calc_sapt_energy` it takes ``response``, which chooses the induction
amplitudes of the second-order induction terms:

.. code-block:: python

   uncoupled = calc_property(dimer, "dipole", response=False)

.. list-table::
   :header-rows: 1

   * - ``response``
     - Second-order induction terms
     - Totals (``x_induced``, ``x_induced_S2``) use
   * - ``True`` (default)
     - coupled (CPSCF): ``x2_ind,r``, ``x2_exch-ind,r_S2``, ``x2_exch-ind,r``; also the
       uncoupled ``x2_ind``
     - the coupled terms
   * - ``False``
     - uncoupled: ``x2_ind``, ``x2_exch-ind_S2``, ``x2_exch-ind``
     - the uncoupled terms
   * - ``"both"``
     - all of the above
     - the coupled terms

The monomers relax in the field in every mode, so the first-order, dispersion and
exchange-dispersion terms do not depend on ``response``. With ``response=False`` no CPSCF
equations are solved for the induction amplitudes or their field derivatives; the two
left are the responses of the monomers to the property operator. This helps when
comparing methods and when isolating problems in the CPSCF solver.

The S^inf exchange-dispersion term ``x2_exch-disp``, and with it the total ``x_induced``,
has a known error; see :ref:`known-limitations`.

Density Matrix Calculations
----------------------------

To compute interaction-induced density matrix components from propSAPT:

.. code-block:: python

   from prop_sapt import calc_densities

   density_matrices = calc_densities(dimer)

The result is a dictionary of density matrices: ``"<component>_A"``, ``"<component>_B"``
and their sum ``"<component>"`` for each component of
:data:`~prop_sapt.densities.DENSITY_COMPONENTS` (``"pol"``, ``"exch"``, ``"ind"``, ...).
The matrices are for one spin, the ``Da`` convention: the total density is twice the
matrix.

The second-order exchange components of each monomer need theta amplitudes of both
monomers. :func:`~prop_sapt.densities.calc_density_matrix` computes them once and keeps
them on the dimer for the other monomer's call, and
:func:`~prop_sapt.densities.calc_densities` frees them when both monomers are done. If
you call :func:`~prop_sapt.densities.calc_density_matrix` for each monomer yourself, free
them afterwards with :meth:`~prop_sapt.molecule.Dimer.clear_theta_amplitudes`: the
exchange-dispersion ones take as much memory as the dispersion amplitudes.

Most of the time goes into solving CPSCF equations, one for each component of each
monomer and for the theta amplitudes. :func:`~prop_sapt.densities.calc_density_matrix`
solves the equations of a monomer together, in one run of the solver
(``Dimer.cpscf_batch``). Each iteration then builds the J and K
matrices of all of them at once, which costs much less than solving them one by one.

Those J and K matrices come from the monomer SCF's J/K object, so they follow
``SCF_TYPE``: with ``DIRECT``, every iteration recomputes the four-index integrals. The
``cpscf_type`` option switches the CPSCF solves of the densities and the properties to
density fitting, while the SAPT energies keep the monomer SCF's J/K object and stay
consistent with Psi4's SAPT:

.. code-block:: python

   from prop_sapt import set_propsapt_options

   set_propsapt_options({"cpscf_type": "DF", "df_basis_cpscf": "aug-cc-pvqz-jkfit"})

``cpscf_type`` is ``None`` (the default, the monomer SCF's J/K object), ``"DIRECT"`` or
``"DF"``. ``df_basis_cpscf`` is the fitting basis of ``"DF"``; it defaults to
``DF_BASIS_SCF``, or else the JKFIT basis of the orbital basis. With ``"DF"`` the
densities and properties solve their own induction amplitudes and differ from the
``DIRECT`` ones by a small fitting error, which a larger fitting basis such as
aug-cc-pVQZ-JKFIT lowers further. The gain is largest with ``SCF_TYPE DIRECT`` and HF,
where the density matrices become several times faster. With DFT the
exchange-correlation kernel takes a large part of the CPSCF time, so the gain is smaller.
In a run configuration, set the options under ``propsapt_options``.

Every matrix is symmetric. :func:`~prop_sapt.densities.calc_density_matrix` and
:func:`~prop_sapt.artifacts.save_densities` check this and warn with
:class:`~prop_sapt.checks.InvariantWarning` if a matrix is not symmetric within the
``check_symmetry_atol`` option (1e-10; ``None`` turns the check off). Bundles written by
0.4.0.dev0 before 2026-10-14 hold second-order exchange components (``exch-ind_S2``,
``exch-ind``, ``exch-disp_S2`` and the ``Ind``, ``Disp`` and ``total`` sums) with an
antisymmetric part. It does not enter densities, cubes or traces with symmetric operators,
but symmetrize such matrices with ``0.5 * (D + D.T)`` before an eigendecomposition (e.g.
natural orbitals) or before writing one triangle of them (e.g. into an fchk file).

With :python:`save_cubes=True` the total densities are saved as :file:`.cube` files for
visualization. With :python:`save_matrices=True` the matrices are saved, always in the AO
basis, into one self-describing bundle, :file:`densities.npz`. It also holds the monomer
wavefunctions and the metadata: the geometry and its fragments, the basis set and the
fitting bases, the method, the Psi4 and propSAPT options, the versions, and the
conventions. The basis sets and the Psi4 options are those the dimer was built with.
Nothing in it is pickled. Read it back with :func:`~prop_sapt.artifacts.load_densities`, which checks the
conventions and, if asked, that the matrices belong to a geometry, a dimer or a basis set:

.. code-block:: python

   from prop_sapt import load_densities

   densities = load_densities("results/geo1", geometry="geos/geo1.p4m")
   rho = densities.matrices["total_A"]      # one spin, AO basis
   rho_total = densities.total("total_A")   # both spins, 2 * rho
   densities.fragments.A.atoms              # fragment structure
   densities.metadata["basis"]["blend"]     # e.g. "AUG-CC-PVTZ"
   wfn_A = densities.wavefunction("A")      # psi4.core.Wavefunction of monomer A

A bundle that does not match raises :class:`~prop_sapt.artifacts.ArtifactError`.
:func:`~prop_sapt.artifacts.save_densities` writes a bundle of any AO matrices of a dimer.
The older layout of bare :file:`delta_dm_<key>.npy` and :file:`wfn_mon_<A/B>.npy` files is
still written with :python:`matrix_format="npy"`, but it is deprecated, and so is the
``compress_matrices`` argument.
   

Finite-Field Properties
-----------------------

Two wrappers compute interaction-induced dipoles numerically, as field derivatives of
energies. Each places the system in a uniform electric field of ``±field_strength`` along
every axis (6 fields for the two-point formula, 12 for the four-point one) through Psi4's
``PERTURB_H``, and differentiates the energies. Both take the geometry string, not a
``Dimer``, and use the Psi4 options set at the time. Only ``"dipole"`` is available.

SAPT
~~~~

:func:`~prop_sapt.finite_field.finite_field_sapt` differentiates every component of
:func:`~prop_sapt.sapt_driver.calc_sapt_energy`, building a dimer for each field. The
result has one row per axis and one column per energy component (``ELST1``, ``EXCH1``,
``IND2``, ``DISP2``, ...). Each column is the finite-field counterpart of a
:func:`~prop_sapt.properties.calc_property` term, so the analytic properties can be
checked term by term; :file:`examples/finite_field_sapt.py` lines them up:

.. code-block:: python

   from prop_sapt import finite_field_sapt

   ff_dipole = finite_field_sapt(
       geometry,
       "dipole",
       field_strength=0.001,
       formula="two_point",     # or "four_point"
   )

``sapt_kwargs`` is passed on to :func:`~prop_sapt.sapt_driver.calc_sapt_energy`, e.g.
``sapt_kwargs={"response": False}``. The result is also saved to :file:`results-ff.csv`
(``results=`` chooses another file), and the SAPT energies of every field to
:file:`results-<axis>-<sign><field>.csv`, both in the working directory.

Any Psi4 energy method
~~~~~~~~~~~~~~~~~~~~~~

:func:`~prop_sapt.finite_field.finite_field_property` does the same for any method
``psi4.energy`` runs: ``"scf"``, a density functional, ``"mp2"``, ``"ccsd"``, and so on.
With ``cp_correction=True`` each energy is the counterpoise-corrected interaction energy
(``bsse_type="cp"``). The result is then the supermolecular interaction-induced dipole of
that method, a reference for the SAPT values:

.. code-block:: python

   from prop_sapt.finite_field import finite_field_property

   ff_ccsd = finite_field_property(
       geometry,
       "ccsd",
       "dipole",
       cp_correction=True,
       field_strength=0.001,
       formula="four_point",
   )

The derivative includes the orbital relaxation. For the HF molecule, the finite-field
DF-MP2 and CCSD dipoles agree with Psi4's analytic orbital-relaxed ones within 3e-9 a.u.
Without ``cp_correction`` the total energy of the system is differentiated. Psi4's field
acts on the electrons only, so that gives the electronic part of the dipole, which depends
on the origin; the nuclear parts cancel in an interaction energy. The result is also saved
to :file:`results-ff-<method>.csv` (``results=`` chooses another file).

Command-Line Runner
-------------------

For production runs, the ``prop-sapt`` command runs a calculation from a configuration
file, with no Python script needed:

.. code-block:: bash

   prop-sapt run config.yaml geos/geo1.p4m --memory 32 --threads 8

The configuration (YAML or JSON) sets the reference, basis sets, resources, Psi4 and
propSAPT options, the output directory and the tasks to run.
See :file:`examples/run_config.yaml` for a commented example:

.. code-block:: yaml

   reference: RHF
   basis: aug-cc-pvtz
   df_basis: aug-cc-pvtz     # DF_BASIS_SCF = aug-cc-pvtz-jkfit, DF_BASIS_SAPT = aug-cc-pvtz-ri
   memory: 2                 # GB
   threads: 2
   psi4_options:
     scf_type: direct
     e_convergence: 1.0e-10
   output:
     dir: "{geometry}"       # expands to the geometry file name without its extension
   tasks:
     monomer_references: true   # monomer density cubes and .fchk files
     sapt_energy: true          # sapt_results.csv
     densities: true            # densities.npz and delta_dm_*.cube

Values come from the built-in defaults, then the file, then the command-line flags
(``--memory``, ``--threads``, ``--reference``, ``--functional``, ``--grac-a``, ``--grac-b``,
``--basis``, ``--df-basis``, ``--output-dir``). The available tasks are
``monomer_references``, ``sapt_energy``, ``property`` and ``densities``; they always run
in that order. The fitting bases can also be given by their full names
(``df_basis_scf``, ``df_basis_sapt``). Without ``df_basis`` or ``df_basis_scf``,
``DF_BASIS_SCF`` is left to Psi4's default.

Elements that lack a basis set can be given a substitute with ``basis_overrides``, either
inline or as a JSON file next to the configuration:

.. code-block:: json

   {"Zn": {"DF_BASIS_SCF": "def2-tzvpd-ri", "DF_BASIS_SAPT": "def2-tzvpd-ri"}}

Add ``--dry-run`` to print the resolved configuration and Psi4 options without calculating
anything. Each run also writes :file:`propsapt_run.json` into the output directory,
recording the resolved configuration, the Psi4 options, the package versions and the
fragments of the geometry. The ``densities`` task saves its matrices into
:file:`densities.npz` (``matrix_format: npz``, the default), whose metadata also holds the
run configuration and the basis overrides applied; ``matrix_format: npy`` (deprecated)
writes bare :file:`.npy` files.
The command exits with status 0 on success, 1 if a calculation failed and 2 for an invalid
configuration, so it can be used directly in batch scripts, e.g. in a SLURM array job:

.. code-block:: bash

   GEO_FILES=(geos/*.p4m)
   prop-sapt run config.yaml "${GEO_FILES[$SLURM_ARRAY_TASK_ID]}" \
       --memory "$MEMORY" --threads "$SLURM_CPUS_PER_TASK"

.. _basis-sets-missing-from-psi4:

Basis sets missing from Psi4
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Basis sets that Psi4's library lacks, for some or all elements, can come from the
`Basis Set Exchange <https://www.basissetexchange.org>`_ (BSE) or from
`ccRepo <http://www.grant-hill.group.shef.ac.uk/ccrepo/>`_, which also holds
correlation-consistent fitting sets (``_JKFIT``, ``_MP2FIT``) for elements missing them in
Psi4. There are two ways to use them.

**For the whole molecule, from the BSE:** Psi4 reads names prefixed with ``bse:`` from the
``basis_set_exchange`` package (``pip install basis_set_exchange``), e.g.
``df_basis_scf: bse:def2-universal-jkfit``. Such names cannot be used in
``basis_overrides``, and ``df_basis`` appends ``-jkfit``/``-ri``, which rarely match BSE
names, so give ``df_basis_scf``/``df_basis_sapt`` in full.

**As files, from either source:** ``prop-sapt basis fetch`` writes a basis set into a
Psi4-format file once, and runs then read it through ``basis_dirs``, with no network or
extra packages needed on the compute nodes:

.. code-block:: bash

   prop-sapt basis fetch ccrepo aug-cc-pVTZ_MP2FIT --elements Zn --dir basis
   prop-sapt basis fetch bse <basis set> --elements <elements> --dir basis

Each file is named after its basis set, ``<source>-<basis set>`` by default (``--as``
chooses another name), e.g. :file:`basis/ccrepo-aug-cc-pvtz_mp2fit.gbs` holds the basis set
``ccrepo-aug-cc-pVTZ_MP2FIT``, and starts with a record of its source, version and date.
A file holding a few elements is used for those elements through ``basis_overrides``, while
the others keep their default. For example, for Zn with aug-cc-pVTZ, whose fitting sets
Psi4 lacks, the RI set can come from ccRepo and the JKFIT set from Psi4's library:

.. code-block:: yaml

   df_basis: aug-cc-pvtz
   basis_dirs: [basis]       # relative to the configuration file
   basis_overrides:
     Zn: {DF_BASIS_SCF: def2-universal-jkfit, DF_BASIS_SAPT: ccrepo-aug-cc-pVTZ_MP2FIT}

A file must not take the name of a basis set in Psi4's library: Psi4 reads the first file
of a name it finds, so a file holding a few elements would hide the library basis set for
every other element. ``prop-sapt basis fetch`` refuses such names, refuses to overwrite a
file without ``--force``, and writes nothing if the basis set is missing for a requested
element.

The sources are used through optional packages, installed separately and only where
``prop-sapt basis fetch`` runs; propSAPT itself does not depend on them:

* ``bse``: ``pip install basis_set_exchange``
  (or ``conda install -c conda-forge basis_set_exchange``);
* ``ccrepo``: ``pip install git+https://github.com/Sheffield-Theoretical-Chemistry/ccrepo_utility colorlog tqdm requests``
  (ccrepo_utility does not install its own dependencies, and it is licensed for
  non-commercial use only).

Please cite the source and the original publications of the basis sets you use.

The same runner is available from Python:

.. code-block:: python

   from prop_sapt import run

   results = run("config.yaml", "geos/geo1.p4m")
   sapt_energy = results["sapt_energy"]

``python -m prop_sapt run ...`` is equivalent to ``prop-sapt run ...``.

Advanced Options
----------------

Customizing Calculations
~~~~~~~~~~~~~~~~~~~~~~~~

You can customize various aspects of the calculation using Psi4 options before creating the Dimer object and running calculations:

.. code-block:: python

   psi4.set_options(
       {
           "basis": "aug-cc-pvtz",
           "DF_BASIS_SCF": "aug-cc-pvtz-jkfit",
           "DF_BASIS_SAPT": "aug-cc-pvtz-ri",
           "scf_type": "direct",
           "e_convergence": 1e-12,
           "d_convergence": 1e-12,
           # Integral thresholds for higher precision
           "ints_tolerance": 1e-14,
           "screening": "schwarz",
           "cholesky_tolerance": 1e-10,
       }
   )

Hellmann-Feynman Forces
-----------------------

To compute Hellmann-Feynman forces on nuclei from one-particle AO density matrices:

.. code-block:: python

   from prop_sapt.forces import calc_force_at_nuc, calc_force_nuc_nuc

   mol = dimer.get_psi4_molecule()
   basisset = dimer.get_psi4_basisset()

   # Force on nucleus 0 from the interaction-induced density of monomer A
   force = calc_force_at_nuc(mol, 2 * rho_A, atidx=0, basisset=basisset, add_nuclear=False)

   # Pass multiple density matrices — integrals are computed only once
   forces = calc_force_at_nuc(
       mol, [2 * rho_A_elst, 2 * rho_A_exch], atidx=0, basisset=basisset, add_nuclear=False
   )

   # Nuclear-nuclear repulsion only (no electron density needed)
   force_nuc = calc_force_nuc_nuc(mol, atidx=0)

All forces are in atomic units (Hartree/Bohr). The nuclear-nuclear repulsion contribution
is added by default; pass ``add_nuclear=False`` to obtain the electronic part only, as for
interaction-induced densities above. The matrices must be in the AO basis of ``basisset``;
without it, a basis set is built for ``mol`` from the current ``BASIS`` option, so pass the
dimer's for its density matrices. As elsewhere, no spin factor is applied: pass
``2 * rho`` for the matrices of :func:`~prop_sapt.densities.calc_densities`.

Evaluating at Arbitrary Points
------------------------------

The :mod:`prop_sapt.evaluate` module evaluates densities, orbitals and the electrostatic
potential (ESP) of AO matrices at any set of points: a quadrature grid, a line, a plane.
Points are in bohr, as an array of shape ``(..., 3)``, and the result has the shape of the
points without the last axis.

.. code-block:: python

   import numpy as np
   from prop_sapt.evaluate import evaluate_density, evaluate_orbital, evaluate_esp

   basisset = dimer.get_psi4_basisset()

   # density along the x axis
   line = np.zeros((200, 3))
   line[:, 0] = np.linspace(-6.0, 6.0, 200)
   rho = evaluate_density(rho_A, line, basisset)                 # shape (200,)

   # several matrices in one call share the basis function values
   rhos = evaluate_density([rho_elst, rho_exch], line, basisset)  # shape (2, 200)

   # orbitals are the columns of the MO coefficients, as in Psi4's Ca()
   C = dimer.wfnA.Ca().np
   psi = evaluate_orbital(C[:, :3], line, basisset)              # shape (3, 200)

   # ESP of the electrons; nuclei=True adds the nuclei of the basis set's molecule
   esp = evaluate_esp(rho_A, line, basisset)

The same functions are :class:`~prop_sapt.molecule.Dimer` methods that use the dimer basis set:
``dimer.evaluate_density(rho, points)``, ``dimer.evaluate_orbital(C, points)`` and
``dimer.evaluate_esp(rho, points)``.

* Matrices must be in the AO basis of ``basisset``, in Psi4 AO order.
* No spin factor is applied. The density matrices returned by
  :func:`~prop_sapt.densities.calc_densities` follow the ``Da`` convention, so pass ``2 * rho``
  for the total (RHF) density, as the density cubes do.
* Basis function values are computed for blocks of ``chunk_size`` points, which takes
  about ``chunk_size * nbf`` floats of memory. The default is the
  ``point_eval_chunk_size`` option (2000).

Integrals over space become weighted sums on a quadrature grid, for example Psi4's
molecular grid from :func:`prop_sapt.partition.molecular_grid`:

.. code-block:: python

   from prop_sapt.partition import molecular_grid

   grid = molecular_grid(basisset)
   n_elec = np.sum(grid.weights * evaluate_density(2 * rho_A, grid.points, basisset))

.. _fragment-populations:

Fragment Populations
--------------------

The :mod:`prop_sapt.partition` module turns a partition of space between fragments into
matrices, computed once per geometry. With fragment weights :math:`w_X(\mathbf{r})` on a
quadrature grid, the weighted-overlap matrix of fragment :math:`X` is

.. math::

   W^X_{\mu\nu} = \int w_X(\mathbf{r})\, \phi_\mu(\mathbf{r})\, \phi_\nu(\mathbf{r})\, d\mathbf{r},

and the electrons any density matrix :math:`D` puts on :math:`X` are the trace
:math:`\mathrm{Tr}(D W^X)`, with no further grid work. The first-moment matrices
(:math:`\mathbf{r}` times the same integrand) give fragment dipoles the same way.

.. code-block:: python

   ops = dimer.fragment_operators("hirshfeld")   # or "becke"

   q_A = ops.charge("A", 2 * rho_pol)            # charge moved onto A, -Tr(D W_A)
   n_B = ops.population("B", 2 * rho_A)          # electrons on B, Tr(D W_B)
   q_all = ops.charge("A", [2 * rho_elst, 2 * rho_exch])     # one value per matrix
   mu_A = ops.dipole("A", 2 * rho_pol, origin=com_A)         # electronic dipole of A

* ``"becke"`` assigns each point to the Becke cell of an atom of monomer A or B.
* ``"hirshfeld"`` uses :math:`w_X = \rho^0_X / (\rho^0_A + \rho^0_B)` from the
  unperturbed monomer densities.
* The grid is Psi4's molecular quadrature grid, 75 radial times 590 Lebedev points per
  atom by default (options ``quadrature_radial_points``, ``quadrature_spherical_points``).
  Its radial extent is twice Psi4's DFT default (option ``quadrature_bs_radius_alpha``),
  which diffuse basis sets need: the operators of fragments that cover the whole system
  sum to the overlap matrix within about 1e-6 for aug-cc-pVTZ.
* As in :mod:`prop_sapt.evaluate`, no spin factor is applied: pass ``2 * rho`` for the
  matrices of :func:`~prop_sapt.densities.calc_densities`. Dipoles are in atomic units, with the
  electrons contributing negatively, as in Psi4's ``ao_dipole``.

Fragment weights from any other partitioning (e.g. LISA or MBIS weights from
horton_part) work as well: give them at the points of the grid they were computed on,
with that grid's quadrature weights. Only these arrays are needed; the basis functions
are evaluated by Psi4 in the given basis set:

.. code-block:: python

   from prop_sapt.partition import FragmentOperators

   # points (n, 3) in bohr, quadrature weights (n,), fragment weights w_A, w_B (n,)
   ops = FragmentOperators.from_grid(
       points, weights, {"A": w_A, "B": w_B}, dimer.get_psi4_basisset()
   )
   q_A = ops.charge("A", 2 * rho_pol)

The accuracy is then that of the given grid. By default (``partition_of_unity=True``)
the fragments are taken to cover the whole system, and two checks guard it; like the
other checks of :mod:`prop_sapt.checks` they warn with
:class:`~prop_sapt.checks.InvariantWarning` and never stop a calculation:

* :meth:`~prop_sapt.partition.FragmentOperators.from_grid` checks that the fragment
  weights sum to 1 wherever the basis functions reach: the largest element of
  :math:`\sum_q w_q (\sum_X w_X(\mathbf{r}_q) - 1)\, \phi_\mu \phi_\nu` must be within the
  ``check_partition_atol`` option (1e-8). Points far from the molecule, where e.g. LISA
  weights stop summing to 1, do not count.
* Every ``population`` and ``charge`` call checks that the fragments account for all the
  electrons of each density, :math:`\sum_X \mathrm{Tr}(D W^X) = \mathrm{Tr}(D S)`, within
  the ``check_closure_atol`` option (1e-6 electrons). The default grid closes within 1e-8
  electrons for the total density of the water dimer in aug-cc-pVTZ and within 1e-9 for
  its interaction-induced densities.

``ops.closure_error(2 * rho)`` gives the electrons a grid misses for a density. For
fragments that do not cover the whole system, e.g. a single atom or a grid over part of
space, their weights do not sum to 1: pass ``partition_of_unity=False`` (also to
``dimer.fragment_operators``), and nothing is checked then.

Cube File Utilities
-------------------

Generating cube files
~~~~~~~~~~~~~~~~~~~~~

:meth:`Dimer.make_cube <prop_sapt.molecule.Dimer.make_cube>` evaluates matrices in the
dimer's basis set on a Cartesian grid, with the functions of :mod:`prop_sapt.evaluate`.
It supports three matrix types:

* ``"density"`` — electron density from a one-particle AO density matrix
* ``"orbital"`` — orbital amplitude from a coefficient vector
* ``"esp"`` — electrostatic potential (AO density matrix contracted with the Coulomb kernel)

.. code-block:: python

   # Single density cube
   cube = dimer.make_cube(2 * rho_A, obj_type="density")
   cube.save("delta_dm_A.cube")

   # Multiple matrices in one call (basis function values computed once)
   cubes = dimer.make_cube(
       [2 * rho_elst, 2 * rho_exch, orbital_vec],
       obj_type=["density", "density", "orbital"],
   )

   # A finer grid than the default (cube_grid_step 0.2 bohr, cube_grid_overage 4.0 bohr)
   cube = dimer.make_cube(2 * rho_B, grid_step=0.1)

:func:`prop_sapt.cubes.make_cube` does the same for any Psi4 molecule; pass the basis set
the matrices are in as ``basisset=``, e.g. ``dimer.get_psi4_basisset()``, since without it
a basis set is built from the current ``BASIS`` option. A grid from
:func:`prop_sapt.cubes.prepare_grid` can be reused across calls with ``grid=``.

Working with cubes
~~~~~~~~~~~~~~~~~~

A :class:`prop_sapt.cubes.Cube` holds values on a regular grid, in bohr. Cubes on the same
grid combine with ``+`` and ``-``, and scale with ``*`` and ``/`` by a number. The grids
are checked first, and the result takes the atoms of the left cube and empty comments:

.. code-block:: python

   import numpy as np
   from prop_sapt.cubes import Cube

   rho = Cube.from_file("cubes/delta_dm_total_A.cube")
   rho_pol = Cube.from_file("cubes/delta_dm_pol_A.cube")

   rho_rest = rho - rho_pol
   rho_rest.comment1 = "total minus polarization density of monomer A"
   rho_rest.save("delta_dm_rest_A.cube")

``Cube.integrate`` sums the values over the voxels times ``Cube.voxel_volume``. It takes an
optional weight at every voxel, either a cube on the same grid or an array of the grid's
shape. Leading axes of the array stack several weights:

.. code-block:: python

   rho.integrate()                       # electrons in the box
   rho.integrate(w_A)                    # electrons that the weight cube w_A gives to A
   rho.integrate(np.stack([w_A.volumetric_data, w_B.volumetric_data]))  # both at once
   np.sqrt(rho.integrate(rho.volumetric_data))  # L2 norm
   rho.moved_charge()                    # half the integral of |rho|

Integrals on a cube carry the error of its voxel grid and box. :ref:`Fragment populations <fragment-populations>` from the
density matrices avoid it.

``read_cube_file`` (also ``Cube.from_file``) accepts any number of values per line. Cubes of
several orbitals, with several values per voxel or in angstrom are not supported.

Visualization
~~~~~~~~~~~~~

The package includes utilities for visualizing orbitals and densities.
See the Jupyter notebooks in the ``examples/`` directory:

* ``visualise_orbital.ipynb`` - Visualize molecular orbitals
* ``visualise_density.ipynb`` - Visualize density matrices

For more examples, see the :doc:`examples` section.

.. _known-limitations:

Known limitations
-----------------

* **S^inf exchange-dispersion property.** The ``x2_exch-disp`` term of
  :func:`~prop_sapt.properties.calc_property` is not yet the field derivative of the S^inf
  exchange-dispersion energy ``EXCH-DISP2``. For the dipole of He–H2 in aug-cc-pVTZ it is
  7.2e-5 a.u. off the finite-field derivative, about 18% of the term, and the total
  ``x_induced`` carries that error. Every term of ``x_induced_S2``, the total with the S^2
  exchange terms, agrees with its finite-field derivative. Use it, or check a system with
  :func:`~prop_sapt.finite_field.finite_field_sapt`. Tracked in
  `issue #1 <https://github.com/btyrcha/propSAPT/issues/1>`_.
* **S^2 exchange in the density totals.** The second-order totals of
  :func:`~prop_sapt.densities.calc_densities` use the S^2 exchange densities: ``Ind`` is ``ind`` plus
  ``exch-ind_S2``, ``Disp`` is ``disp`` plus ``exch-disp_S2``, and ``total`` sums ``pol``,
  ``exch``, ``Ind`` and ``Disp``. The S^inf exchange-induction density ``exch-ind`` is
  returned on its own; an S^inf exchange-dispersion density is not implemented yet.
* **Uncoupled dispersion only.** The dispersion and exchange-dispersion energies, and the
  dispersion terms of the properties and density matrices, are the uncoupled ones. With
  ``RHF`` this is the SAPT0 dispersion. With ``RKS`` it is not the coupled dispersion that
  Psi4's SAPT(DFT) uses by default, so SAPT(DFT) dispersion and total energies differ from
  Psi4's. Coupled dispersion is not implemented yet.
* **Closed-shell references only.** The monomers must be closed-shell singlets with an
  ``RHF`` or ``RKS`` reference; unrestricted references are not implemented.
* **All-electron only.** Frozen core is not supported, and :class:`~prop_sapt.molecule.Dimer`
  rejects the ``FREEZE_CORE`` option.
* **Finite-field files.** :func:`~prop_sapt.finite_field.finite_field_sapt` writes the SAPT energies of
  every field into the working directory, :file:`results-<axis>-<sign><field>.csv`.
