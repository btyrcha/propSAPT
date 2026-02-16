Examples
========

This page contains examples of using prop-sapt for various calculations.

Example Scripts
---------------

The ``examples/`` directory contains several demonstration scripts:

Core Functionality
~~~~~~~~~~~~~~~~~~

These scripts showcase the core functionality of prop-sapt:

* :file:`example.py` - General usage example for property and density matrices
* :file:`sapt_example.py` - Calculating SAPT interaction energy components 
* :file:`finite_field_sapt.py` - Basic finite field SAPT calculation for dipole moments

Property Calculations
~~~~~~~~~~~~~~~~~~~~~

Calculating properties using finite field approach:

* :file:`water_dimer.py` - Water dimer interaction analysis


Jupyter Notebooks
-----------------

Interactive examples are available as Jupyter notebooks:

Orbital Visualization
~~~~~~~~~~~~~~~~~~~~~

``examples/visualise_orbital.ipynb`` - Interactive visualization of molecular orbitals
using the generated cube files.

Density Visualization
~~~~~~~~~~~~~~~~~~~~~

``examples/visualise_density.ipynb`` - Interactive visualization of interaction 
density matrices and their components.

Running Examples
----------------

To run the examples:

.. code-block:: bash

   cd examples
   python water_dimer.py

For Jupyter notebooks:

.. code-block:: bash

   cd examples
   jupyter notebook visualise_orbital.ipynb

Additional Resources
--------------------

* See the :doc:`api` for detailed function documentation
* Refer to the :doc:`usage` guide for general workflows
