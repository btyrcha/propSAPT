import os
import warnings
from collections.abc import Mapping
from pathlib import Path
import psi4
import numpy as np
import opt_einsum as oe
from .molecule import Dimer
from .artifacts import DENSITIES_FILENAME, save_densities
from .checks import check_symmetric
from .cubes import Cube
from .utils import trace_memory_peak, CalcTimer, prepare_path
from .cpscf_batching import DensityBuilder, solve_batched
from .densities_components.exch_disp import exch_disp_s2_density_builder
from .densities_components.exch_ind import (
    exch_ind_density_builder,
    exch_ind_s2_density_builder,
)

#: Keys of the density components returned by `calc_density_matrix`, in order;
#: `calc_densities` returns them as "<key>_A", "<key>_B" and, in the AO basis, "<key>".
DENSITY_COMPONENTS = (
    "pol",
    "exch",
    "ind",
    "exch-ind_S2",
    "exch-ind",
    "disp",
    "exch-disp_S2",
    "Ind",
    "Disp",
    "total",
)

#: Formats of the matrices saved by `calc_densities`: "npz" for the self-describing bundle
#: (see `prop_sapt.artifacts`), "npy" for bare files (deprecated).
MATRIX_FORMATS = ("npz", "npy")


def density_mo_to_ao(
    dimer: Dimer, monomer: str, density_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform density matrix from MO to AO.

    Args:
        dimer (Dimer): A dimer system.
        monomer (str): Select a monomer, either "A" or "B".
        density_matrix (np.ndarray): Density matrix to transform.

    Returns:
        np.ndarray: Transformed density matrix.
    """

    if monomer == "A":
        return dimer.C_A.dot(density_matrix).dot(dimer.C_A.T)
    elif monomer == "B":
        return dimer.C_B.dot(density_matrix).dot(dimer.C_B.T)
    else:
        psi4.core.clean()
        raise ValueError(f"Invalid monomer: {monomer}")


def calc_densities(
    dimer: Dimer,
    orbital_basis="AO",
    save_cubes=False,
    cubes_dir=".",
    save_matrices=False,
    matrices_dir=".",
    matrix_format: str | None = None,
    metadata: Mapping | None = None,
    compress_matrices: bool | None = None,
) -> dict[str, np.ndarray]:
    """
    Calculate interaction-induced changes in the density matrix of dimer (and both monomers).

    The matrices are for one spin (the ``Da`` convention): the total density of the
    closed-shell monomers is twice their sum. Saved matrices are always in the AO basis
    (transformed from the MO basis if needed); saved cubes hold the total density.

    Args:
        dimer (Dimer): A dimer system.
        orbital_basis (str, optional): Select orbital basis of the returned density matrix, either
            "AO" or "MO". Defaults to "AO".
        save_cubes (bool, optional): Whether to save density cubes. Defaults to False.
        cubes_dir (str, optional): Directory to save density cubes. Defaults to ".".
        save_matrices (bool, optional): Whether to save density matrices. Defaults to False.
        matrices_dir (str, optional): Directory to save density matrices. Defaults to ".".
        matrix_format (str | None, optional): "npz" saves the matrices, the monomer
            wavefunctions and their metadata into one self-describing bundle,
            ``densities.npz``, read with `prop_sapt.load_densities`. "npy" saves bare
            ``delta_dm_<key>.npy`` and ``wfn_mon_<A/B>.npy`` files; it is deprecated.
            Defaults to "npz".
        metadata (Mapping | None, optional): JSON-serialisable entries stored in the
            bundle's metadata under "extra". Defaults to None.
        compress_matrices (bool | None, optional): Deprecated, use `matrix_format`:
            True is "npz" and False is "npy".

    Returns:
        dict[str, np.ndarray]: Density matrices keyed "<component>_A", "<component>_B"
            and, in the AO basis, "<component>" for their sum, with the components of
            `DENSITY_COMPONENTS`. In the MO basis there is no sum: each monomer's
            matrices are in its own MO basis, and the two are not orthogonal.
    """

    orbital_basis = orbital_basis.upper()
    matrix_format = _resolve_matrix_format(matrix_format, compress_matrices)

    try:
        rho_A = calc_density_matrix(dimer, monomer="A", orbital_basis=orbital_basis)
        rho_B = calc_density_matrix(dimer, monomer="B", orbital_basis=orbital_basis)
    finally:
        # the theta amplitudes are only shared by the two monomers' densities,
        # and the exchange-dispersion ones are as large as the dispersion amplitudes
        dimer.clear_theta_amplitudes()

    rho_dict = {}
    for key in rho_A:
        rho_dict[f"{key}_A"] = rho_A[key]
        rho_dict[f"{key}_B"] = rho_B[key]
        if orbital_basis == "AO":
            rho_dict[f"{key}"] = rho_A[key] + rho_B[key]

    # matrices to save, in the AO basis; nothing is added to them (they may be rho_dict)
    rho_ao = rho_dict
    if orbital_basis != "AO" and (save_cubes or save_matrices):

        psi4.core.print_out(
            f"\nWarning: Density matrices are in {orbital_basis} basis,"
            "attempting MO -> AO transformation, for saving...\n"
        )

        rho_ao = {
            key: density_mo_to_ao(dimer, monomer=key.split("_")[-1], density_matrix=rho)
            for key, rho in rho_dict.items()
        }

        psi4.core.print_out("MO -> AO transformation complete.\n")

    cube_filenames = {key: f"{cubes_dir}/delta_dm_{key}.cube" for key in rho_ao}

    if save_matrices:

        psi4.core.print_out("\nSaving density matrices...\n")
        prepare_path(matrices_dir + "/dummy")  # create matrices_dir if it doesn't exist

        if matrix_format == "npz":

            bundle_metadata = {}
            if metadata:
                bundle_metadata["extra"] = dict(metadata)
            if save_cubes:
                bundle_metadata["cubes"] = {
                    "files": [
                        os.path.relpath(fname, matrices_dir)
                        for fname in cube_filenames.values()
                    ],
                    "density": "total",
                }

            bundle = save_densities(
                Path(matrices_dir) / DENSITIES_FILENAME,
                rho_ao,
                dimer,
                metadata=bundle_metadata,
            )

            psi4.core.print_out(
                f"Density matrices and monomer wavefunctions saved into {bundle}.\n"
                "To load use `prop_sapt.load_densities`.\n"
            )

        else:

            message = (
                'Saving density matrices as bare .npy files (matrix_format="npy") is '
                "deprecated and will be removed; the default .npz bundle records the "
                "geometry, basis set and conventions, read it with "
                "`prop_sapt.load_densities`."
            )
            warnings.warn(message, FutureWarning, stacklevel=2)
            psi4.core.print_out(f"\nWarning: {message}\n")

            for key, matrix in rho_ao.items():
                np.save(f"{matrices_dir}/delta_dm_{key}.npy", matrix)

            psi4.core.print_out("Density matrices saved.\n")

            # Save wavefunction information
            psi4.core.print_out("\nSaving monomer Wfn objects...\n")
            dimer.wfnA.to_file(f"{matrices_dir}/wfn_mon_A.npy")
            dimer.wfnB.to_file(f"{matrices_dir}/wfn_mon_B.npy")
            psi4.core.print_out(
                "Monomer Wfn objects saved. To load use `psi4.core.Wavefunction.from_file`\n"
            )

    if save_cubes:

        psi4.core.print_out("\nSaving density cubes...\n")

        prepare_path(cubes_dir + "/dummy")  # create cubes_dir if it doesn't exist

        cubes = dimer.make_cube(
            [2 * rho for rho in rho_ao.values()],  # factor 2 for RHF spin
            ["density"] * len(rho_ao),  # type of objects for cube generation
        )
        if isinstance(cubes, Cube):
            cubes = [cubes]

        for key, cube in zip(rho_ao, cubes):
            cube.comment1 = (
                f"propSAPT delta_dm_{key}: total density (2 x alpha-spin matrix)"
            )
            cube.save(cube_filenames[key])

        psi4.core.print_out("Density cubes saved.\n")

    return rho_dict


def _resolve_matrix_format(matrix_format: str | None, compress_matrices) -> str:
    """Returns the matrix format of `calc_densities`, "npz" or "npy", translating the
    deprecated `compress_matrices` argument."""

    if compress_matrices is not None:
        if matrix_format is not None:
            psi4.core.clean()
            raise TypeError(
                "`compress_matrices` is deprecated, use `matrix_format` alone!"
            )
        warnings.warn(
            '`compress_matrices` is deprecated, use `matrix_format="npz"` (the default) '
            'or `matrix_format="npy"`.',
            FutureWarning,
            stacklevel=3,
        )
        matrix_format = "npz" if compress_matrices else "npy"

    if matrix_format is None:
        return "npz"

    if (
        not isinstance(matrix_format, str)
        or matrix_format.lower() not in MATRIX_FORMATS
    ):
        psi4.core.clean()
        raise ValueError(
            f"`matrix_format` should be one of {MATRIX_FORMATS}, was {matrix_format!r}!"
        )

    return matrix_format.lower()


def _exch_density_builder(mol: Dimer, monomer: str) -> DensityBuilder:
    """Builder of the first-order exchange density matrix (MO) of `monomer`, see
    `prop_sapt.cpscf_batching`."""

    rho_MO_exch = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        (rho_exch_ra,) = yield [("A", mol.omega_exchB_ar + mol.omega_exchB_ra.T)]
        rho_exch_ra = 0.5 * rho_exch_ra

        # first-order exchange
        rho_MO_exch[mol.slices["r"], mol.slices["a"]] = rho_exch_ra
        rho_MO_exch[mol.slices["a"], mol.slices["r"]] = rho_exch_ra.T

    if monomer == "B":

        (rho_exch_sb,) = yield [("B", mol.omega_exchA_bs + mol.omega_exchA_sb.T)]
        rho_exch_sb = 0.5 * rho_exch_sb

        # first-order exchange
        rho_MO_exch[mol.slices["s"], mol.slices["b"]] = rho_exch_sb
        rho_MO_exch[mol.slices["b"], mol.slices["s"]] = rho_exch_sb.T

    return rho_MO_exch


def _ind_density_builder(mol: Dimer, monomer: str) -> DensityBuilder:
    """Builder of the second-order induction density matrix (MO) of `monomer`, see
    `prop_sapt.cpscf_batching`."""

    rho_MO_ind = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        # second-order induction
        rho_MO_ind_ra = (
            +2 * oe.contract("Qar,Qbs,sb->ra", mol.Qar, mol.Qbs, mol.get_cpscf_sb())
            + 2 * oe.contract("sb,Qar,Qbs->ra", mol.get_cpscf_sb(), mol.Qar, mol.Qbs)
            - oe.contract("ra,aA->rA", mol.get_cpscf_ra(), mol.omegaB_aa)
            + oe.contract("ra,Rr->Ra", mol.get_cpscf_ra(), mol.omegaB_rr)
            # 2 Re <C^{(1)}_A | W_A C^{[1]}_A C^{(1)}_A > + Re <C^{(1)}_A C^{(1)}_A | W_A C^{[1]}_A >
            + oe.contract(
                "cr,de,Qad,Qec->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + oe.contract(
                "cd,de,Qer,Qac->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + 4
            * oe.contract(
                "ac,de,Qcr,Qed->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            + 2
            * oe.contract(
                "cd,ec,Qde,Qar->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - oe.contract(
                "ac,de,Qcd,Qer->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - oe.contract(
                "cd,ec,Qdr,Qae->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qrr,
                mol.Qar,
            )
            - 4
            * oe.contract(
                "cr,de,Qed,Qac->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            - 2
            * oe.contract(
                "cd,de,Qar,Qec->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qaa,
            )
            + oe.contract(
                "cr,de,Qae,Qcd->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qaa,
                mol.Qar,
            )
            - oe.contract(
                "ac,de,Qec,Qrd->ra",
                mol.get_cpscf_ra().T,
                mol.get_cpscf_ra(),
                mol.Qar,
                mol.Qrr,
            )
        )
        (rho_MO_ind_ra,) = yield [("A", rho_MO_ind_ra.T)]

        rho_MO_ind[mol.slices["r"], mol.slices["a"]] = rho_MO_ind_ra
        rho_MO_ind[mol.slices["a"], mol.slices["r"]] = rho_MO_ind_ra.T

        rho_MO_ind[mol.slices["a"], mol.slices["a"]] = -oe.contract(
            "rA,ra->aA", mol.get_cpscf_ra(), mol.get_cpscf_ra()
        )
        rho_MO_ind[mol.slices["r"], mol.slices["r"]] = +oe.contract(
            "Ra,ra->Rr", mol.get_cpscf_ra(), mol.get_cpscf_ra()
        )

    if monomer == "B":

        # second-order induction
        rho_MO_ind_sb = (
            +2 * oe.contract("Qar,Qbs,ra->sb", mol.Qar, mol.Qbs, mol.get_cpscf_ra())
            + 2 * oe.contract("ra,Qar,Qbs->sb", mol.get_cpscf_ra(), mol.Qar, mol.Qbs)
            - oe.contract("sb,bB->sB", mol.get_cpscf_sb(), mol.omegaA_bb)
            + oe.contract("sb,Ss->Sb", mol.get_cpscf_sb(), mol.omegaA_ss)
            # 2 Re <C^{(1)}_B | W_B C^{[1]}_B C^{(1)}_B > + Re <C^{(1)}_B C^{(1)}_B | W_B C^{[1]}_B >
            + oe.contract(
                "cs,de,Qbd,Qec->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + oe.contract(
                "cd,de,Qes,Qbc->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "bc,de,Qcs,Qed->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            + 2
            * oe.contract(
                "cd,ec,Qde,Qbs->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - oe.contract(
                "bc,de,Qcd,Qes->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - oe.contract(
                "cd,ec,Qds,Qbe->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qss,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "cs,de,Qed,Qbc->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            - 2
            * oe.contract(
                "cd,de,Qbs,Qec->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qbb,
            )
            + oe.contract(
                "cs,de,Qbe,Qcd->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbb,
                mol.Qbs,
            )
            - oe.contract(
                "bc,de,Qec,Qsd->sb",
                mol.get_cpscf_sb().T,
                mol.get_cpscf_sb(),
                mol.Qbs,
                mol.Qss,
            )
        )
        (rho_MO_ind_sb,) = yield [("B", rho_MO_ind_sb.T)]

        rho_MO_ind[mol.slices["s"], mol.slices["b"]] = rho_MO_ind_sb
        rho_MO_ind[mol.slices["b"], mol.slices["s"]] = rho_MO_ind_sb.T

        rho_MO_ind[mol.slices["b"], mol.slices["b"]] = -oe.contract(
            "sB,sb->bB", mol.get_cpscf_sb(), mol.get_cpscf_sb()
        )
        rho_MO_ind[mol.slices["s"], mol.slices["s"]] = +oe.contract(
            "Sb,sb->Ss", mol.get_cpscf_sb(), mol.get_cpscf_sb()
        )

    return rho_MO_ind


def _disp_density_builder(mol: Dimer, monomer: str) -> DensityBuilder:
    """Builder of the second-order dispersion density matrix (MO) of `monomer`, see
    `prop_sapt.cpscf_batching`."""

    rho_MO_disp = np.zeros((mol.nmo, mol.nmo))

    if monomer == "A":

        # second-order dispersion
        rho_MO_disp_ra = (
            2 * oe.contract("rsab,QRr,Qbs->Ra", mol.t_rsab, mol.Qrr, mol.Qbs)
            - 2 * oe.contract("rsab,QaA,Qbs->rA", mol.t_rsab, mol.Qaa, mol.Qbs)
            # X_A -> X_A + [W_A, C_A - C_A^\dagger]_1
            + 2
            * oe.contract(
                "dscb,dseb,Qer,Qac->ra",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qaa,
            )
            + 4
            * oe.contract(
                "dscb,escb,Qar,Qde->ra",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qrr,
            )
            - 2
            * oe.contract(
                "dscb,escb,Qdr,Qae->ra",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qrr,
                mol.Qar,
            )
            - 4
            * oe.contract(
                "dscb,dseb,Qar,Qec->ra",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qar,
                mol.Qaa,
            )
        )
        (rho_MO_disp_ra,) = yield [("A", rho_MO_disp_ra.T)]

        rho_MO_disp[mol.slices["r"], mol.slices["a"]] = rho_MO_disp_ra
        rho_MO_disp[mol.slices["a"], mol.slices["r"]] = rho_MO_disp_ra.T

        rho_MO_disp[mol.slices["a"], mol.slices["a"]] = -2 * oe.contract(
            "rsAb,rsab->aA", mol.t_rsab, mol.t_rsab
        )
        rho_MO_disp[mol.slices["r"], mol.slices["r"]] = +2 * oe.contract(
            "Rsab,rsab->Rr", mol.t_rsab, mol.t_rsab
        )

    if monomer == "B":

        # second-order dispersion
        rho_MO_disp_sb = (
            +2 * oe.contract("rsab,Qar,QSs->Sb", mol.t_rsab, mol.Qar, mol.Qss)
            - 2
            * oe.contract(
                "rsab,Qar,QbB->sB", mol.t_rsab, mol.Qar, mol.Qbb
            )  # X_B -> X_B + [W_B, C_B - C_B^\dagger]_1
            + 2
            * oe.contract(
                "rdac,rdae,Qes,Qbc->sb",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qbb,
            )
            + 4
            * oe.contract(
                "rdac,reac,Qbs,Qde->sb",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qss,
            )
            - 2
            * oe.contract(
                "rdac,reac,Qds,Qbe->sb",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qss,
                mol.Qbs,
            )
            - 4
            * oe.contract(
                "rdac,rdae,Qbs,Qec->sb",
                mol.t_rsab,
                mol.t_rsab,
                mol.Qbs,
                mol.Qbb,
            )
        )
        (rho_MO_disp_sb,) = yield [("B", rho_MO_disp_sb.T)]

        rho_MO_disp[mol.slices["s"], mol.slices["b"]] = rho_MO_disp_sb
        rho_MO_disp[mol.slices["b"], mol.slices["s"]] = rho_MO_disp_sb.T

        rho_MO_disp[mol.slices["b"], mol.slices["b"]] = -2 * oe.contract(
            "rsaB,rsab->bB", mol.t_rsab, mol.t_rsab
        )
        rho_MO_disp[mol.slices["s"], mol.slices["s"]] = +2 * oe.contract(
            "rSab,rsab->Ss", mol.t_rsab, mol.t_rsab
        )

    return rho_MO_disp


#: Density components whose CPSCF equations `calc_density_matrix` solves in batches:
#: key, builder (see `prop_sapt.cpscf_batching`) and the name in the output.
BATCHED_COMPONENTS = (
    ("exch", _exch_density_builder, "First-order Exchange"),
    ("ind", _ind_density_builder, "Second-order Induction"),
    ("disp", _disp_density_builder, "Second-order Dispersion"),
    (
        "exch-ind_S2",
        exch_ind_s2_density_builder,
        "Second-order Exchange-Induction (S^2)",
    ),
    ("exch-ind", exch_ind_density_builder, "Second-order Exchange-Induction (S^inf)"),
    (
        "exch-disp_S2",
        exch_disp_s2_density_builder,
        "Second-order Exchange-Dispersion (S^2)",
    ),
)


@trace_memory_peak
def calc_density_matrix(
    dimer: Dimer, monomer: str, orbital_basis="AO"
) -> dict[str, np.ndarray]:
    """
    Calculate interaction-induced change in the density matrix of the 'monomer'.

    The theta amplitudes of the second-order exchange components are shared by the two
    monomers, so they are kept on `dimer` for the other monomer's call;
    `dimer.clear_theta_amplitudes()` frees them (`calc_densities` does this itself).

    The CPSCF equations of the components are solved together, with one
    `Dimer.cpscf_batch` call per monomer at a time: the first monomer's call solves its
    components and the theta amplitudes, then the exchange-induction components that
    need them (16 equations of the two calls in 4 batches).

    Args:
        dimer (Dimer): A dimer system.
        monomer (str): Select a monomer, either "A" or "B".
        orbital_basis (str, optional): Select orbital basis of the returned density matrix, either
            "AO" or "MO". Defaults to "AO".

    Returns:
        dict[str, np.ndarray]: Density matrix components, keyed by `DENSITY_COMPONENTS`:
            "pol" and "exch": first-order polarisation and exchange corrections;
            "ind", "exch-ind_S2" (S^2) and "exch-ind" (S^inf): second-order induction and
            exchange-induction; "disp" and "exch-disp_S2" (S^2): second-order dispersion
            and exchange-dispersion; "Ind" and "Disp": induction and dispersion totals with
            S^2 exchange; "total": sum of "pol", "exch", "Ind" and "Disp".
            Every component is a symmetric matrix.

    Warns:
        InvariantWarning: If a component is not symmetric within the
            `check_symmetry_atol` option, see `prop_sapt.checks.check_symmetric`.
    """

    if monomer not in ["A", "B"]:
        psi4.core.clean()
        raise ValueError(f"'{monomer}' is not a valid monomer for density matrix.")

    orbital_basis = orbital_basis.upper()
    if orbital_basis not in ["MO", "AO"]:
        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )

    rho_MO_pol = np.zeros((dimer.nmo, dimer.nmo))

    with CalcTimer("First-order Polarisation density calculation"):
        if monomer == "A":
            rho_pol_ra = dimer.get_cpscf_ra()

            # first-order polarisation
            rho_MO_pol[dimer.slices["r"], dimer.slices["a"]] = rho_pol_ra
            rho_MO_pol[dimer.slices["a"], dimer.slices["r"]] = rho_pol_ra.T

        if monomer == "B":
            rho_pol_sb = dimer.get_cpscf_sb()

            # first-order polarisation
            rho_MO_pol[dimer.slices["s"], dimer.slices["b"]] = rho_pol_sb
            rho_MO_pol[dimer.slices["b"], dimer.slices["s"]] = rho_pol_sb.T

    # the CPSCF equations of the other components are solved together, a batch per
    # monomer at a time, see `solve_batched`
    solved = solve_batched(
        dimer,
        {
            f"{name} density calculation": build(dimer, monomer)
            for _, build, name in BATCHED_COMPONENTS
        },
    )
    rho = {
        key: solved[f"{name} density calculation"]
        for key, _, name in BATCHED_COMPONENTS
    }

    rho_MO_exch = rho["exch"]
    rho_MO_ind = rho["ind"]
    rho_MO_disp = rho["disp"]
    rho_MO_exch_ind_s2 = rho["exch-ind_S2"]
    rho_MO_exch_ind = rho["exch-ind"] - rho_MO_ind
    rho_MO_exch_disp_s2 = rho["exch-disp_S2"]

    # sum up all contributions
    rho_MO_total = (
        rho_MO_pol
        + rho_MO_exch
        + rho_MO_ind
        + rho_MO_exch_ind_s2  # NOTE: change into Sinf version when ready
        + rho_MO_disp
        + rho_MO_exch_disp_s2  # NOTE: change into Sinf version when ready
    )

    # prepare total Induction and Dispersion densities
    rho_MO_IND = (
        rho_MO_ind + rho_MO_exch_ind_s2
    )  # NOTE: change into Sinf version when ready
    rho_MO_DISP = (
        rho_MO_disp + rho_MO_exch_disp_s2
    )  # NOTE: change into Sinf version when ready

    rho_dict = {
        "pol": rho_MO_pol,
        "exch": rho_MO_exch,
        "ind": rho_MO_ind,
        "exch-ind_S2": rho_MO_exch_ind_s2,
        "exch-ind": rho_MO_exch_ind,
        "disp": rho_MO_disp,
        "exch-disp_S2": rho_MO_exch_disp_s2,
        "Ind": rho_MO_IND,
        "Disp": rho_MO_DISP,
        "total": rho_MO_total,
    }

    # in the MO basis the matrices are exactly symmetric
    check_symmetric(rho_dict, where=f"monomer {monomer}")

    if orbital_basis == "AO":

        for key in rho_dict:
            rho_dict[key] = density_mo_to_ao(dimer, monomer, rho_dict[key])

        return rho_dict

    elif orbital_basis == "MO":

        return rho_dict

    else:

        psi4.core.clean()
        raise ValueError(
            f"Argument 'orbital_basis' should be either 'MO' or 'AO' but was '{orbital_basis}'!"
        )
