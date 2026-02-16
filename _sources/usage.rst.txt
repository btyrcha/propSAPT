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
It is created by providing a geometry string in Psi4 format, which should include two fragments (for both monomers) separated by ``--``:

.. code-block:: python

   from prop_sapt import Dimer

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

If your geometry string will include more fragments, they will be treated as ghost atoms.
You can specify midbonds and farbonds in this way.

SAPT Energy Calculations
-------------------------

To perform SAPT energy decomposition at the SAPT0 level:

.. code-block:: python

   from prop_sapt import calc_sapt_energy

   # Calculate all SAPT components
   energy_result = calc_sapt_energy(dimer)
   
   print(energy_result)

The result is a pandas DataFrame containing all SAPT energy components.

Property Calculations
---------------------

To calculate interaction-induced properties for a general one-electron operator:

.. code-block:: python

   from prop_sapt import calc_property
   import numpy as np

   # Define property operator (e.g., dipole moment in z-direction) in AO basis
   property_matrix = np.array([...])

   # Calculate property contributions
   property_result = calc_property(dimer, property_matrix)
   
   print(property_result)

For predefined properties, you can use the convenience functions, e.g., to calculate the dipole moment:

.. code-block:: python

   from prop_sapt import calc_property

   dipole_result = calc_property(dimer, "dipole")

Density Matrix Calculations
----------------------------

To compute interaction-induced density matrix components from propSAPT:

.. code-block:: python

   from prop_sapt import calc_densities

   density_matrices = calc_densities(dimer)

The result is a dictionary containing density matrices for each SAPT component.
They can also be saved in :file:`.cube` format for visualization (add parameter :python:`save_cubes=True`) 
or in :file:`.npy` format for further analysis (add parameter :python:`save_matrices=True`).
   

Finite Field SAPT
------------------

For finite field SAPT calculations:

.. code-block:: python

   from prop_sapt import finite_field_sapt

   # Perform finite field calculation
   ff_result = finite_field_sapt(
       geometry,
       field_strength=0.0001,
       direction='Z'
   )

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
           "save_jk": True,  # necessary option
           "e_convergence": 1e-12,
           "d_convergence": 1e-12,
           # Integral thresholds for higher precision
           "ints_tolerance": 1e-14,
           "screening": "schwarz",
           "cholesky_tolerance": 1e-10,
       }
   )

Visualization
~~~~~~~~~~~~~

The package includes utilities for visualizing orbitals and densities.
See the Jupyter notebooks in the ``examples/`` directory:

* ``visualise_orbital.ipynb`` - Visualize molecular orbitals
* ``visualise_density.ipynb`` - Visualize density matrices

For more examples, see the :doc:`examples` section.
