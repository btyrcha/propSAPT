Installation
============

Requirements
------------

prop-sapt requires:

* Python 3.7 or later
* NumPy
* Pandas
* opt_einsum
* Psi4 (for quantum chemical calculations)

Installing from Source
----------------------

Currently, prop-sapt is not available on PyPI nor conda-forge, so you need to install it from source:

.. code-block:: bash

   git clone https://github.com/yourusername/propSAPT.git
   cd propSAPT

Conda Environment
-----------------

We highly recommend using a conda environment.
The `.yaml` file provided in the repository can be used to create an environment with all necessary dependencies.
No further configuration is required after executing the following commands:

.. code-block:: bash

   conda env create -n prop-sapt -f prop-sapt.yaml
   conda activate prop-sapt
   pip install -e . --config-settings editable_mode=compat

Dependencies
------------

Alternatively, you take care of dependencies manually.
The core dependencies will be installed automatically:

* `numpy` - Numerical computing
* `pandas` - Data manipulation
* `opt_einsum` - Optimized tensor operations

You will also need to install Psi4 separately. Visit the 
`Psi4 installation guide <https://psicode.org/installs/latest/>`_ for instructions.

Verifying Installation
----------------------

To verify that prop-sapt is installed correctly:

.. code-block:: python

   import prop_sapt
   print(prop_sapt.__version__)

Or run the test suite:

.. code-block:: bash

   pytest
