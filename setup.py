from setuptools import find_packages, setup

requirements = [
    "numpy<2.0",
    "pandas",
    "opt_einsum",
]

setup(
    name="interaction-induced",
    version="0.2.0",
    author="Bartosz Tyrcha",
    author_email="bartektyrcha123@gmail.com",
    description="Package for calculations of first-order interaction-induced properties and density matrices in the spirit of SAPT.",
    packages=find_packages(),
    install_requires=requirements,
)
