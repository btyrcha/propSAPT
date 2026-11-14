.. prop-sapt documentation master file

Welcome to prop-sapt's documentation!
======================================

**prop-sapt** is a Python package for calculations of first-order interaction-induced
properties and density matrices in the spirit of SAPT (Symmetry-Adapted Perturbation
Theory). It is built on `Psi4 <https://psicode.org>`_, which runs the SCF calculations and
the integrals; the SAPT expressions are evaluated with NumPy, with density fitting in the
MO basis. The analytic property derivatives are described in
`J. Chem. Theory Comput. 21, 4562 (2025) <https://doi.org/10.1021/acs.jctc.5c00238>`_ and,
for the second-order corrections, in `arXiv:2611.07993 <https://arxiv.org/abs/2611.07993>`_;
see :ref:`citing`.

Features
--------

* Interaction-induced properties (e.g. the dipole moment) as analytic derivatives of the
  SAPT energy corrections
* Interaction-induced density matrices and their decomposition into SAPT components
* The SAPT0 / SAPT(DFT) energy decomposition, with uncoupled dispersion
* Finite-field wrappers for interaction-induced dipoles, from the SAPT energy components
  (to check the analytic properties) or from the counterpoise-corrected interaction energy
  of any Psi4 method, e.g. MP2 or CCSD (a supermolecular reference)
* Densities, orbitals and electrostatic potentials on grids, fragment populations, and
  :file:`.cube` file utilities
* A command-line runner, ``prop-sapt run``, for production calculations from a
  configuration file

See :doc:`installation` to install it, and :ref:`known-limitations` before relying on the
S^inf exchange-dispersion terms.

Quick Start
-----------

.. code-block:: python

   import psi4
   from prop_sapt import Dimer, calc_property, calc_sapt_energy

   psi4.core.set_output_file("output.dat", False)
   psi4.set_memory("2 GB")
   psi4.set_options(
       {
           "basis": "aug-cc-pvdz",
           "df_basis_scf": "aug-cc-pvdz-jkfit",
           "df_basis_sapt": "aug-cc-pvdz-ri",
       }
   )

   # the two monomers are separated by "--"
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
   symmetry c1
   no_com
   no_reorient
   """

   dimer = Dimer(geometry)

   # SAPT energy components (pandas Series, in hartree), also saved to sapt_results.csv
   sapt_energy = calc_sapt_energy(dimer)

   # interaction-induced dipole moment, one row per axis (atomic units), also results.csv
   dipole = calc_property(dimer, "dipole")
   print(dipole[["x1_pol,r", "x1_exch,r", "x2_ind,r", "x2_disp", "x_induced_S2"]])

The :doc:`usage` guide covers the density matrices, the runner and the other tools.

.. _citing:

Citing
------

If you use prop-sapt, please cite

* B. Tyrcha, T. Gupta, K. Patkowski, P. S. Żuchowski, *Analytical Derivatives of
  Symmetry-Adapted Perturbation Theory Corrections for Interaction-Induced Properties*,
  J. Chem. Theory Comput. **21**, 4562–4578 (2025),
  `doi:10.1021/acs.jctc.5c00238 <https://doi.org/10.1021/acs.jctc.5c00238>`_.

If you use the second-order properties (the ``x2_*`` terms and the totals), please also
cite

* B. Tyrcha, M. Stachowiak, K. Patkowski, P. S. Żuchowski, *Direct calculation of dipole
  moment surfaces through second-order symmetry-adapted perturbation theory for
  properties*, arXiv:2611.07993 (2026),
  `doi:10.48550/arXiv.2611.07993 <https://doi.org/10.48550/arXiv.2611.07993>`_.

.. code-block:: bibtex

   @article{Tyrcha2025,
     author  = {Tyrcha, Bartosz and Gupta, Tarun and Patkowski, Konrad and {\.Z}uchowski, Piotr S.},
     title   = {Analytical Derivatives of Symmetry-Adapted Perturbation Theory Corrections for Interaction-Induced Properties},
     journal = {Journal of Chemical Theory and Computation},
     volume  = {21},
     number  = {9},
     pages   = {4562--4578},
     year    = {2025},
     doi     = {10.1021/acs.jctc.5c00238},
   }

   @misc{Tyrcha2026,
     author        = {Tyrcha, Bartosz and Stachowiak, Marcin and Patkowski, Konrad and {\.Z}uchowski, Piotr S.},
     title         = {Direct calculation of dipole moment surfaces through second-order symmetry-adapted perturbation theory for properties},
     year          = {2026},
     eprint        = {2611.07993},
     archivePrefix = {arXiv},
     primaryClass  = {physics.chem-ph},
     doi           = {10.48550/arXiv.2611.07993},
   }

The software itself is archived on Zenodo:
`doi:10.5281/zenodo.23951143 <https://doi.org/10.5281/zenodo.23951143>`_ always points to
the latest version, and each version has its own DOI.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   usage
   examples


.. toctree::
   :maxdepth: 2
   :caption: API Documentation:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
