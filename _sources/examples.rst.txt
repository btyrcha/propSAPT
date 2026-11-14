Examples
========

The ``examples/`` directory of the repository holds scripts and notebooks. Run them from
that directory, in an environment made from :file:`prop-sapt.yaml` (see
:doc:`installation`). Each script sets its geometry, memory, threads and basis sets at the
top (``BASIS``, ``DF_BASIS``).

Scripts
-------

* :file:`example.py` - interaction-induced dipole moment and density matrices of Ne–H2
* :file:`sapt_example.py` - the SAPT energy components of He\ :sub:`2`, compared with
  Psi4's own SAPT(DFT) run with the HF functional
* :file:`finite_field_sapt.py` - the analytic interaction-induced dipole of He–H2 next to
  the finite-field SAPT derivative, term by term
* :file:`water_dimer.py` - the dipole and the density cubes of the water dimer
* :file:`save_mo_matrices.py` - the density matrices of Ne–H2 computed in the MO basis and
  saved, in the AO basis, into one bundle (:file:`matrices_MO/densities.npz`)
* :file:`make_cubes_from_mo_matrices.py` - cube files from that bundle, read back with
  :func:`~prop_sapt.artifacts.load_densities`; run :file:`save_mo_matrices.py` first
* :file:`read_subtract_save.py` - cube arithmetic on the cubes that :file:`water_dimer.py`
  writes to :file:`water-dimer-cubes/`; run :file:`water_dimer.py` first
* :file:`run_config.yaml` - a commented configuration for ``prop-sapt run``

For example:

.. code-block:: bash

   cd examples
   python water_dimer.py

Notebooks
---------

* :file:`visualise_density.ipynb` - interaction-induced densities from the cubes of
  :file:`water_dimer.py`
* :file:`visualise_orbital.ipynb` - molecular orbitals from cube files

.. code-block:: bash

   cd examples
   jupyter lab visualise_density.ipynb

The notebooks use the visualisation packages of :file:`prop-sapt.yaml` (``fortecubeview``,
``pythreejs``, ``ipywidgets``).

See the :doc:`usage` guide for the workflows, and the :doc:`api` for every function.
