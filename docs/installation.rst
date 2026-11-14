Installation
============

Requirements
------------

* Python 3.12 or later
* `Psi4 <https://psicode.org>`_, tested with 1.11
* NumPy, pandas, opt_einsum and PyYAML, installed with the package

Psi4 cannot be installed with pip, so it comes from conda (conda-forge). propSAPT itself is
installed with pip from its GitHub repository; it is not on PyPI or conda-forge.

Installing a release
--------------------

Create a conda environment with Psi4, then install a tagged release of propSAPT into it:

.. code-block:: bash

   conda create -n prop-sapt -c conda-forge python=3.12 psi4
   conda activate prop-sapt
   pip install "prop_sapt @ git+https://github.com/btyrcha/propSAPT@v0.4.0"

The releases are listed on the
`GitHub releases page <https://github.com/btyrcha/propSAPT/releases>`_, which also has the
sdist and wheel of each.

Development install
-------------------

To work on propSAPT, or to run the examples and tests, clone the repository and create
the environment from :file:`prop-sapt.yaml`, which also holds the test, lint and notebook
tools:

.. code-block:: bash

   git clone https://github.com/btyrcha/propSAPT.git
   cd propSAPT
   conda env create -f prop-sapt.yaml
   conda activate prop-sapt
   pip install -e . --config-settings editable_mode=compat

Optional packages
-----------------

``prop-sapt basis fetch`` (see :ref:`the usage guide <basis-sets-missing-from-psi4>`) reads
basis sets through packages that propSAPT itself does not need:

* the Basis Set Exchange source, through the ``bse`` extra:
  ``pip install "prop_sapt[bse] @ git+https://github.com/btyrcha/propSAPT@v0.4.0"``,
  or ``pip install basis_set_exchange``;
* the ccRepo source, installed from git:
  ``pip install git+https://github.com/Sheffield-Theoretical-Chemistry/ccrepo_utility colorlog tqdm requests``.

Writing the density changes into TREXIO files, to view them in VIAMD (see
:ref:`the usage guide <viamd>`), needs the ``trexio`` package: ``pip install trexio``, or
the ``trexio`` extra. The conda environment of :file:`prop-sapt.yaml` includes it.

The ``docs`` extra installs Sphinx and its theme, to build this documentation
(``sphinx-build -W -b html docs docs/_build/html``).

Verifying the installation
--------------------------

.. code-block:: python

   import prop_sapt
   print(prop_sapt.__version__)

With a development install, run the test suite from the repository root:

.. code-block:: bash

   pytest

The tests run real Psi4 calculations; the full suite takes 10 to 15 minutes.
