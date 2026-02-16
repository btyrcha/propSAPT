.. prop-sapt documentation master file

Welcome to prop-sapt's documentation!
======================================

**prop-sapt** is a Python package for calculations of first-order interaction-induced 
properties and density matrices in the spirit of SAPT (Symmetry-Adapted Perturbation Theory).

Features
--------

* Compute interaction-induced properties directly
* Generate interaction-induced density matrices and its propSAPT decomposition
* Calculate standard SAPT energy decomposition
* Support for finite field SAPT calculations
* Visualization tools and `.cube` file utilities

Quick Start
-----------

Installation
~~~~~~~~~~~~

1. Clone the repository.

    .. code-block:: bash
    
        git clone https://github.com/yourusername/propSAPT.git

2. Go to its root directory.

    .. code-block:: bash
    
        cd propSAPT

3. Create a conda environment with all requirements by running:

   .. code-block:: bash

      conda env create -f prop-sapt.yaml

4. Activate the environment.

   .. code-block:: bash

      conda activate prop-sapt

5. Install the development package by running:

   .. code-block:: bash

      pip install -e . --config-settings editable_mode=compat
   
   while being in the repo's root directory.

6. Verify the installation:

   Check that prop-sapt is installed correctly and can be imported:

   .. code-block:: python

      import prop_sapt
      print(prop_sapt.__version__)


   Additionally, you can test the installation by running the test suite:

   .. code-block:: bash
    
      cd prop_sapt
      pytest 

7. Check it out by running some scripts from `examples/`!



Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from prop_sapt import Dimer, calc_property, calc_sapt_energy

   # Create a dimer system
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
   
   dimer = Dimer(geometry)
   
   # Calculate SAPT energy
   sapt_energy = calc_sapt_energy(dimer)
   
   # Calculate properties
   property_result = calc_property(dimer, property_matrix)


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
