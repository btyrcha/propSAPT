"""
Options module for the propSAPT package.

Package-wide default values for keyword arguments that are otherwise hardcoded in
propSAPT routines. An explicitly passed keyword argument always takes precedence
over the value stored here. Use `prop_sapt.set_propsapt_options` and
`prop_sapt.get_propsapt_options` to modify and inspect the current values.

Available options:
    default_functional (str): DFT functional used for the monomer calculations
        with the RKS reference, if `functional` is not given to `Dimer`.
        Defaults to "pbe0".
    r_convergence (float): Residual convergence threshold of the CPSCF solver.
        Defaults to 1.0e-10.
    cpscf_maxiter (int): Maximum number of CPSCF iterations.
        Defaults to 50.
    cpscf_type (str | None): How the CPSCF solves of the density matrices and the
        properties build their J and K matrices: "DIRECT" (exact integrals) or "DF"
        (density fitted with `df_basis_cpscf`), case-insensitive. None, the default,
        uses the J/K object of the monomer SCF, which follows Psi4's `SCF_TYPE`. The
        SAPT energies always use the monomer SCF's J/K object, so they stay consistent
        with Psi4's SAPT; with another type, the densities and properties solve their
        own induction amplitudes. Defaults to None.
    df_basis_cpscf (str | None): Fitting basis of the CPSCF solves with `cpscf_type`
        "DF". None takes the dimer's `DF_BASIS_SCF` fitting basis, that of Psi4's
        `DF_BASIS_SCF` when the dimer was built, or else the JKFIT basis of its orbital
        basis. Unused by the other types. Defaults to None.
    cube_grid_step (float): Step size (in bohr) of the grid built by `make_cube`,
        if no `grid` is given. Defaults to 0.2.
    cube_grid_overage (float): Extent (in bohr) of the grid built by `make_cube`
        beyond the outermost atoms, if no `grid` is given. Defaults to 4.0.
    cube_iso_sum_level (float): Fraction of the density (or of the squared orbital)
        enclosed by the isosurface, used for the isovalues reported in cube files.
        Defaults to 0.85.
    point_eval_chunk_size (int): Number of points for which basis function values are
        computed at once by the `prop_sapt.evaluate` functions and `make_cube`; the
        memory used is about `point_eval_chunk_size * nbf` floats. Defaults to 2000.
    quadrature_radial_points (int): Number of radial points per atom of the molecular
        quadrature grid built by `prop_sapt.partition.molecular_grid`. Defaults to 75.
    quadrature_spherical_points (int): Number of Lebedev points per radial shell of that
        grid. Defaults to 590.
    quadrature_bs_radius_alpha (float): Scale of the Bragg-Slater radii setting the radial
        extent of that grid, Psi4's `DFT_BS_RADIUS_ALPHA`. Psi4's default of 1.0 is too
        short for products of diffuse basis functions. Defaults to 2.0.
    check_symmetry_atol (float | None): Largest element of ``|D - D^T|`` allowed for a
        density matrix by `prop_sapt.checks.check_symmetric`, which `calc_density_matrix`
        and `save_densities` run; a larger one warns. None turns the check off.
        Defaults to 1.0e-10.
    check_partition_atol (float | None): Largest defect of fragment weights allowed by
        `prop_sapt.checks.check_partition_of_unity`, which
        `FragmentOperators.from_grid` runs: the largest element of
        ``sum_q w_q (sum_X w_X - 1) phi_mu phi_nu``. A larger one warns. None turns the
        check off. Defaults to 1.0e-8.
    check_closure_atol (float | None): Largest ``|sum_X Tr(D W_X) - Tr(D S)|`` in
        electrons allowed by `prop_sapt.checks.check_charge_closure`, which
        `FragmentOperators.population` and `charge` run; a larger one warns. None turns
        the check off. Defaults to 1.0e-6.
"""

from collections.abc import Collection, Iterator

# Default option values
_OPTION_DEFAULTS = {
    "default_functional": "pbe0",
    "r_convergence": 1.0e-10,  # CPSCF convergence threshold
    "cpscf_maxiter": 50,
    "cpscf_type": None,  # the monomer SCF's J/K
    "df_basis_cpscf": None,
    "cube_grid_step": 0.2,  # in bohr
    "cube_grid_overage": 4.0,  # in bohr
    "cube_iso_sum_level": 0.85,
    "point_eval_chunk_size": 2000,
    "quadrature_radial_points": 75,
    "quadrature_spherical_points": 590,
    "quadrature_bs_radius_alpha": 2.0,
    "check_symmetry_atol": 1.0e-10,
    "check_partition_atol": 1.0e-8,
    "check_closure_atol": 1.0e-6,
}

_VALID_OPTION_KEYS = frozenset(_OPTION_DEFAULTS.keys())

CPSCF_TYPES = ("DIRECT", "DF")


def check_cpscf_type(cpscf_type: str | None) -> str | None:
    """Checks a value of the `cpscf_type` option.

    Args:
        cpscf_type (str | None): None, "DIRECT" or "DF", case-insensitive.

    Returns:
        str | None: The type in upper case, or None.

    Raises:
        ValueError: Raised for any other value.
    """

    if cpscf_type is None:
        return None

    if not isinstance(cpscf_type, str) or cpscf_type.upper() not in CPSCF_TYPES:
        raise ValueError(
            f"Invalid `cpscf_type` {cpscf_type!r}! Use None (the monomer SCF's J/K), "
            f"{' or '.join(repr(t) for t in CPSCF_TYPES)}."
        )

    return cpscf_type.upper()


def check_df_basis_cpscf(df_basis: str | None) -> str | None:
    """Checks a value of the `df_basis_cpscf` option.

    Args:
        df_basis (str | None): None or the name of a basis set.

    Returns:
        str | None: `df_basis`.

    Raises:
        ValueError: Raised for an empty or non-string name.
    """

    if df_basis is not None and (not isinstance(df_basis, str) or not df_basis.strip()):
        raise ValueError(
            f"Invalid `df_basis_cpscf` {df_basis!r}! Use None or a basis set name."
        )

    return df_basis


# options whose values are checked when they are set
_VALUE_CHECKS = {
    "cpscf_type": check_cpscf_type,
    "df_basis_cpscf": check_df_basis_cpscf,
}


class PropsaptOptions:
    """
    Container for the package-wide propSAPT options.

    Provides read-only, dict-like access (`options[key]`, `key in options`,
    iteration over keys, `len`). Options should be modified only through
    `prop_sapt.set_propsapt_options`. The module-level `PROPSAPT_OPTIONS`
    instance is the one used throughout the package.
    """

    def __init__(self) -> None:

        self._options_dict = dict(_OPTION_DEFAULTS)

    def __getitem__(self, key: str):
        """Returns the current value of the option `key`."""

        if not isinstance(key, str):
            # NOTE: this could raise `KeyError` immediately
            # we only accept string keys
            return self._options_dict[key]

        return self._options_dict[key.lower()]

    # NOTE: The `__setitem__` and `__delitem__` are NOT set purposefully
    # to protect `_options_dict` from unwanted changes.

    def __contains__(self, key: str) -> bool:
        """Checks whether `key` is a known option."""

        if not isinstance(key, str):
            # NOTE: this could return `False` immediately
            # we only accept string keys
            return key in self._options_dict

        return key.lower() in self._options_dict

    def __iter__(self) -> Iterator[str]:
        """Iterates over the option keys."""

        return iter(self._options_dict)

    def __len__(self) -> int:
        """Returns the number of options."""

        return len(self._options_dict)

    def _update(self, options: dict):
        """Updates propSAPT options dict. This is the class protected method.
        The `prop_sapt.set_propsapt_options` is the preferred way to set options.

        Option keys are case-insensitive. Options not present in `options`
        keep their current values.

        Args:
            options (dict): Dictionary containing `option_key: option_value` pairs.

        Raises:
            TypeError: Raised when a non-string `option_key` is found in `options`.
            KeyError: Raised when invalid `option_key` is found in `options`.
            ValueError: Raised when an invalid `cpscf_type` or `df_basis_cpscf` is given.
        """

        # check if given option keys are valid
        for option_key in options:
            check_if_valid_option(option_key)

        # lowercase all option keys
        options_lower = {key.lower(): value for key, value in options.items()}

        # check the values that can be checked before they are used
        for key, value in options_lower.items():
            check_option_value(key, value)

        self._options_dict.update(options_lower)

    def get_options_dict(self) -> dict:
        """Returns a copy of the current options.

        Modifying the returned dictionary does not change the propSAPT options.

        Returns:
            dict: Dictionary containing `option_key: option_value` pairs.
        """

        return dict(self._options_dict)


# Initialize the module with default option values
PROPSAPT_OPTIONS = PropsaptOptions()


def check_if_valid_option(option_key: str) -> None:
    """Checks whether `option_key` is a valid propSAPT option key.

    Args:
        option_key (str): Option key to check (case-insensitive).

    Raises:
        TypeError: Raised when a non-string `option_key` is given.
        KeyError: Raised when invalid `option_key` is given.
    """

    if not isinstance(option_key, str):
        raise TypeError(
            f"Only strings are accepted as option keys! Received `{type(option_key)}`."
        )

    if option_key.lower() not in _VALID_OPTION_KEYS:
        raise KeyError(
            f"Invalid propSAPT option key! Option `{option_key}` not recognised."
        )


def check_option_value(option_key: str, value) -> None:
    """Checks the value of an option whose values are checked when they are set,
    `cpscf_type` and `df_basis_cpscf`. The values of the other options are checked where
    they are used.

    Args:
        option_key (str): A valid option key (case-insensitive).
        value: Its value.

    Raises:
        ValueError: Raised when `value` is invalid.
    """

    check = _VALUE_CHECKS.get(option_key.lower())
    if check is not None:
        check(value)


def set_propsapt_options(options: dict) -> None:
    """Sets package-wide propSAPT options.

    The new values are used as defaults by all subsequent propSAPT calls, unless
    overridden by an explicitly passed keyword argument. Option keys are
    case-insensitive. Options not present in `options` keep their current values.
    See the `prop_sapt.options` module docstring for the available options.

    Example:
        >>> prop_sapt.set_propsapt_options({"r_convergence": 1.0e-8, "cpscf_maxiter": 100})

    Args:
        options (dict): Dictionary containing `option_key: option_value` pairs.

    Raises:
        TypeError: Raised when a non-string `option_key` is found in `options`.
        KeyError: Raised when invalid `option_key` is found in `options`.
        ValueError: Raised when an invalid `cpscf_type` or `df_basis_cpscf` is given.
    """

    # NOTE: this is the ONLY place where `PropsaptOptions._update()` is ok.
    PROPSAPT_OPTIONS._update(options)


def get_propsapt_options() -> dict:
    """Returns the current package-wide propSAPT options.

    Modifying the returned dictionary does not change the propSAPT options,
    use `prop_sapt.set_propsapt_options` instead.

    Returns:
        dict: Dictionary containing `option_key: option_value` pairs.
    """

    return PROPSAPT_OPTIONS.get_options_dict()


def reset_propsapt_options(option_keys: Collection[str] | None = None) -> None:
    """Resets the package-wide propSAPT options to the default values.

    Args:
        option_keys (Collection[str] | None, optional): Option keys to be set to default
            values (e.g. a list, tuple, set or dict), or a single key as a string.
            If None, all options are being set to default values. Defaults to None.

    Raises:
        TypeError: Raised when `option_keys` is not a collection (e.g. a generator)
            or a non-string option key is given.
        KeyError: Raised when an invalid option key is given.
        ValueError: Raised when `option_keys` is empty.
    """

    # reset all options
    if option_keys is None:
        # NOTE: this is the ONLY place where `PropsaptOptions._options_dict` is ok.
        PROPSAPT_OPTIONS._options_dict = dict(_OPTION_DEFAULTS)

    # one-shot iterators (e.g. generators) are not accepted
    elif not isinstance(option_keys, Collection):
        raise TypeError(
            "`option_keys` has to be a collection (e.g. list, tuple, set)! "
            f"Received `{type(option_keys)}`."
        )

    # reset selected options
    elif option_keys:

        # a single option key
        if isinstance(option_keys, str):
            option_keys = [option_keys]

        # check if given option keys are valid
        for option_key in option_keys:
            check_if_valid_option(option_key)

        defaults_subdict = {
            key.lower(): _OPTION_DEFAULTS[key.lower()] for key in option_keys
        }
        set_propsapt_options(defaults_subdict)

    # something unexpected given
    else:
        raise ValueError(f"Invalid `option_keys` given! {option_keys}")
