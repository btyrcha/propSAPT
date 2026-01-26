from setuptools import find_packages, setup

requirements = [
    "numpy",
    "pandas",
    "opt_einsum",
]

setup(
    name="prop_sapt",
    version="0.2.0",
    author="Bartosz Tyrcha",
    author_email="bartektyrcha123@gmail.com",
    description="Package for calculations of first-order interaction-induced properties and density matrices in the spirit of SAPT.",
    packages=find_packages(),
    install_requires=requirements,
)
