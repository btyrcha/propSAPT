"""Psi4 as the backend of propSAPT.

`Psi4Backend` runs the monomer SCFs of a dimer in the dimer-centred basis set and builds
the integrals propSAPT needs, as Psi4's own SAPT does. It keeps the Psi4 objects it made
(wavefunctions, `MintsHelper`, JK object), which a `Dimer` also gives as its `wfnA`,
`wfnB`, `mints`, ... compatibility attributes. The Psi4 options in place when it is built
decide the calculation: `BASIS`, `DF_BASIS_SAPT`, `DF_BASIS_SCF`, `SCF_TYPE`, the
convergence thresholds, and Psi4's memory (`psi4.set_memory`).
"""

import logging
import time
from collections.abc import Sequence
from contextlib import contextmanager
from functools import wraps
from typing import Iterator, Literal, overload

import numpy as np
import opt_einsum as oe
import psi4
from psi4.driver.p4util import message_box, OptionsState, OptionsStateCM

from ..fragments import DimerFragments
from .base import MONOMERS, Backend, MonomerReference, check_reference

logger = logging.getLogger(__name__)


@overload
def psi4_energy(name: str, return_wfn: Literal[False] = False, **kwargs) -> float: ...


@overload
def psi4_energy(
    name: str, return_wfn: Literal[True], **kwargs
) -> tuple[float, psi4.core.Wavefunction]: ...


@wraps(psi4.energy)
def psi4_energy(
    name: str, return_wfn: bool = False, **kwargs
) -> float | tuple[float, psi4.core.Wavefunction]:
    # SAVE_JK keeps the JK object on the wavefunction, which CPSCF (`cphf_Hx`) needs;
    # the user's setting is restored afterwards, as Psi4's own SAPT does
    optstash = OptionsState(["SAVE_JK"], ["SCF", "SAVE_JK"])
    try:
        psi4.core.set_global_option("SAVE_JK", True)
        psi4.core.set_local_option("SCF", "SAVE_JK", True)
        return psi4.energy(name, return_wfn=return_wfn, **kwargs)
    finally:
        optstash.restore()


def jk_memory_doubles(memory: float | None = None) -> int:
    """Memory of the JK object `Psi4Backend` builds, in doubles, the unit of
    `JK.set_memory`.

    Args:
        memory (float | None, optional): Memory in GB, from the deprecated
            `Dimer(memory=...)`. None, the default, takes Psi4's memory
            (`psi4.set_memory`) times `SCF_MEM_SAFETY_FACTOR`, as Psi4's SCF does for
            its own JK object.

    Returns:
        int: Memory in doubles (8 bytes each).
    """

    if memory is None:
        safety_factor = psi4.core.get_global_option("SCF_MEM_SAFETY_FACTOR")
        memory_bytes = psi4.get_memory() * safety_factor
    else:
        memory_bytes = memory * 1e9

    return int(memory_bytes / 8)


def changed_psi4_options() -> dict | None:
    """The Psi4 options changed from their defaults, None if Psi4 cannot list them."""

    try:
        from psi4.driver.p4util import prepare_options_for_set_options
    except ImportError:
        return None

    return prepare_options_for_set_options()


class Psi4Backend(Backend):
    """
    Psi4's reference calculation of a dimer: the SCFs of monomers A and B in the
    dimer-centred basis set, the integrals of the dimer and Psi4's J/K objects.

    Building it runs the calculation: the geometry becomes Psi4's active molecule, and
    the SCFs follow the Psi4 options in place. The dimer is kept in its input frame (no
    reorientation, symmetry c1).

    Attributes:
        molecule (psi4.core.Molecule): The dimer.
        wavefunctions (dict[str, psi4.core.Wavefunction]): Monomer SCF wavefunctions by
            monomer, "A" and "B".
        dimer_wfn (psi4.core.Wavefunction): Wavefunction of the dimer basis set, without
            an SCF.
        mints (psi4.core.MintsHelper): Integrals of the dimer basis set.
        jk (psi4.core.JK): J/K object of the dimer basis set (density fitted in
            `DF_BASIS_SCF` or exact, as `SCF_TYPE` says).
        jk_memory (int): Memory of the J/K objects, in doubles.
        memory (float): The same in GB.
        psi4_options (dict | None): The Psi4 options changed from their defaults when
            the dimer was built, before the SCFs changed any.
        fitting_bases (dict[str, str]): The fitting basis sets as built
            (``basis.blend()``), by option: "DF_BASIS_SAPT" and "DF_BASIS_SCF".

    Args:
        geometry (str): Dimer geometry in Psi4 format, with monomers A and B as its
            first two fragments; any further fragments are ghost atoms.
        reference (str, optional): "RHF" or "RKS". Defaults to "RHF".
        functional (str | None, optional): Functional of an RKS reference, required for
            it. Defaults to None.
        grac_A (float | None, optional): GRAC shift of monomer A (Psi4's
            `dft_grac_shift`). Defaults to no shift.
        grac_B (float | None, optional): GRAC shift of monomer B. Defaults to no shift.
        memory (float | None, optional): Memory of the J/K objects in GB, see
            `jk_memory_doubles`. Defaults to Psi4's memory.

    Raises:
        ValueError: If frozen core is set, the reference is not supported, the geometry
            has fewer than two fragments, or a monomer is not a singlet; all before any
            SCF is run.
    """

    def __init__(
        self,
        geometry: str,
        reference: str = "RHF",
        *,
        functional: str | None = None,
        grac_A: float | None = None,
        grac_B: float | None = None,
        memory: float | None = None,
    ):

        with self.timer():
            self._run(geometry, reference, functional, grac_A, grac_B, memory)

    def _run(self, geometry, reference, functional, grac_A, grac_B, memory):
        """The calculation of `__init__`, in Psi4's timer."""

        tstart_backend = time.time()

        # create Psi4 Molecule object
        dimer = psi4.geometry(geometry)
        dimer.print_out_in_angstrom()
        self.molecule = dimer

        # the Psi4 options the dimer is built with, before the monomer SCFs change any
        self.psi4_options = changed_psi4_options()

        # propSAPT is all-electron: reject frozen core instead of ignoring it
        if psi4.core.get_global_option("FREEZE_CORE") not in ("FALSE", "0"):
            raise ValueError(
                "propSAPT does not support frozen core; set FREEZE_CORE to false."
            )

        check_reference(reference)
        if reference == "RKS" and functional is None:
            raise ValueError("An RKS reference needs a functional.")

        dimer.reset_point_group("c1")
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()

        if nfrags < 2:
            raise ValueError(f"Found {nfrags:d} fragments, must be at least 2.")
        elif nfrags > 2:
            logger.info(
                f"Found {nfrags:d} fragments, using first 2 as monomers, "
                "and the rest as extra basis functions (ghosts)."
            )

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, [i for i in range(2, nfrags + 1)])
        monomerA.set_name("monomerA")
        monomerB = dimer.extract_subsets(2, [1] + [i for i in range(3, nfrags + 1)])
        monomerB.set_name("monomerB")
        monomer_molecules = {"A": monomerA, "B": monomerB}

        # closed-shell references only, fail before any SCF is run
        for label in MONOMERS:
            mult = monomer_molecules[label].multiplicity()
            if mult != 1:
                raise ValueError(
                    f"Monomer {label} has multiplicity {mult}, "
                    f"{reference} reference requires singlet monomers."
                )

        # Compute monomer properties
        grac = {"A": grac_A, "B": grac_B}
        energies = {}
        potentials = {}
        self.wavefunctions = {}
        for label in MONOMERS:
            logger.info(message_box(f"Monomer {label} {reference}").strip("\n"))
            tstart = time.time()
            if reference == "RHF":
                energies[label], self.wavefunctions[label] = psi4_energy(
                    "SCF", return_wfn=True, molecule=monomer_molecules[label]
                )
            elif reference == "RKS":
                with OptionsStateCM(["dft_grac_shift"]):
                    if grac[label]:
                        psi4.set_options({"dft_grac_shift": grac[label]})
                    else:
                        logger.warning(
                            f"GRAC shift for monomer {label} not specified; "
                            "running without GRAC."
                        )
                    energies[label], self.wavefunctions[label] = psi4_energy(
                        functional, return_wfn=True, molecule=monomer_molecules[label]
                    )
            potentials[label] = np.asarray(
                psi4.core.MintsHelper(
                    self.wavefunctions[label].basisset()
                ).ao_potential()
            )
            logger.info(
                f"{reference} for monomer {label} finished in "
                f"{time.time() - tstart:.2f} seconds."
            )

        # Setup a few variables
        self.jk_memory = jk_memory_doubles(memory)  # in doubles
        self.memory = self.jk_memory * 8 / 1e9  # the same in GB

        monomers = {}
        for label in MONOMERS:
            wfn = self.wavefunctions[label]
            monomers[label] = MonomerReference(
                C=np.asarray(wfn.Ca()),
                eps=np.asarray(wfn.epsilon_a()),
                ndocc=wfn.doccpi()[0],
                V=potentials[label],
                energy=energies[label],
                nuclear_repulsion=monomer_molecules[label].nuclear_repulsion_energy(),
            )

        ### Compute DF ERIs for dimer basis
        tstart = time.time()
        logger.info("Compute DF integrals for dimer-centered basis...")

        self.dimer_wfn = psi4.core.Wavefunction.build(
            dimer, psi4.core.get_global_option("BASIS")
        )
        self.mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())

        # DF integrals in AO
        aux_basis = psi4.core.BasisSet.build(
            self.dimer_wfn.molecule(),
            target=psi4.core.get_global_option("DF_BASIS_SAPT"),
            other=psi4.core.get_global_option("DF_BASIS_SAPT"),
            fitrole="RIFIT",
        )
        aux_basis.print_out()
        # the fitting bases as built, not the options naming them
        self.fitting_bases = {"DF_BASIS_SAPT": aux_basis.blend()}
        zero_basis = psi4.core.BasisSet.zero_ao_basis_set()
        metric = self.mints.ao_eri(aux_basis, zero_basis, aux_basis, zero_basis)
        metric.power(-0.5, 1.0e-14)
        metric = np.squeeze(metric)
        Ppq = self.mints.ao_eri(
            aux_basis,
            zero_basis,
            self.dimer_wfn.basisset(),
            self.dimer_wfn.basisset(),
        )
        Ppq = np.squeeze(Ppq)
        Qpq = np.array(oe.contract("QP,Ppq->Qpq", metric, Ppq))

        logger.info(f"... DF integrals finished in {time.time() - tstart:.2f} seconds.")

        # JK object
        logger.info("Initialize JK objects...")
        tstart = time.time()
        aux_basis = psi4.core.BasisSet.build(
            self.dimer_wfn.molecule(),
            "DF_BASIS_SCF",
            psi4.core.get_option("SCF", "DF_BASIS_SCF"),
            "JKFIT",
            psi4.core.get_global_option("BASIS"),
            puream=self.dimer_wfn.basisset().has_puream(),
        )
        self.fitting_bases["DF_BASIS_SCF"] = aux_basis.blend()
        # also the default fitting basis of the "DF" CPSCF, see `build_cpscf_jk`
        self._df_basis_scf = aux_basis

        self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
        self.jk.set_memory(self.jk_memory)
        self.jk.initialize()
        logger.info(f"...initialized JK objects in {time.time() - tstart:.2f} seconds.")

        # Overlap matrix in AO
        S = np.asarray(self.mints.ao_overlap())

        super().__init__(
            reference=reference,
            monomers=monomers,
            S=S,
            Qpq=Qpq,
            nuclear_repulsion=dimer.nuclear_repulsion_energy(),
            fragments=DimerFragments.from_molecule(dimer),
            functional=functional,
            grac_A=grac_A,
            grac_B=grac_B,
        )

        logger.info(
            f"...finished the Psi4 calculations in "
            f"{time.time() - tstart_backend:5.2f} seconds."
        )

    @property
    def basisset(self) -> psi4.core.BasisSet:
        """The dimer basis set, the one the backend was built with."""

        return self.dimer_wfn.basisset()

    def compute_jk(
        self, lefts: Sequence[np.ndarray], rights: Sequence[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """J and K matrices of the pairs (C_left, C_right), with the J/K object `jk`, see
        `Backend.compute_jk`."""

        for Cl, Cr in zip(lefts, rights):
            self.jk.C_left_add(psi4.core.Matrix.from_array(Cl))
            self.jk.C_right_add(psi4.core.Matrix.from_array(Cr))

        self.jk.compute()

        J_list = [np.array(J) for J in self.jk.J()]
        K_list = [np.array(K) for K in self.jk.K()]

        self.jk.C_clear()

        return J_list, K_list

    def orbital_hessian_product(
        self,
        monomer: str,
        vectors: Sequence[np.ndarray],
        cpscf_jk: psi4.core.JK | None = None,
    ) -> list[np.ndarray]:
        """Products of the orbital Hessian of `monomer` with (ov) `vectors`, see
        `Backend.orbital_hessian_product`. Psi4's `cphf_Hx` gives them, with the
        opposite sign, taking the J and K matrices from the monomer wavefunction's J/K
        object; `cpscf_jk` takes its place for the call."""

        wfn = self.wavefunctions[monomer]
        matrices = [
            psi4.core.Matrix.from_array(np.ascontiguousarray(x)) for x in vectors
        ]

        # `cphf_Hx` takes the J and K matrices from the wavefunction's J/K object
        swap_jk = cpscf_jk is not None
        if swap_jk:
            scf_jk = wfn.jk()
            wfn.set_jk(cpscf_jk)

        try:
            products = wfn.cphf_Hx(matrices)
        finally:
            if swap_jk:
                wfn.set_jk(scf_jk)

        return [-np.asarray(product) for product in products]

    def build_cpscf_jk(
        self, cpscf_type: str, df_basis: str | None = None
    ) -> psi4.core.JK | None:
        """
        J/K object for CPSCF solves of type `cpscf_type`, shared by both monomers, see
        `Backend.build_cpscf_jk`.

        It is set up as Psi4's SCF sets up its own (`initialize_jk`), with the memory of
        the backend's J/K object, `jk_memory`. The "DF" one is MemDFJK if it fits in
        that memory and DiskDFJK otherwise, as for Psi4's `SCF_TYPE` "DF".

        Args:
            cpscf_type (str): "DIRECT" or "DF".
            df_basis (str | None, optional): Fitting basis of "DF". Defaults to the
                dimer's `DF_BASIS_SCF` fitting basis (see `fitting_bases`): the one of
                Psi4's `DF_BASIS_SCF` when the dimer was built, or else the JKFIT basis
                of its orbital basis.

        Returns:
            psi4.core.JK | None: The J/K object, or None if the monomer SCF's J/K
                objects are of this type already ("DF" only without `df_basis`).
        """

        scf_jk_names = {wfn.jk().name() for wfn in self.wavefunctions.values()}
        if cpscf_type == "DIRECT" and scf_jk_names == {"DirectJK"}:
            return None
        if (
            cpscf_type == "DF"
            and df_basis is None
            and scf_jk_names <= {"MemDFJK", "DiskDFJK"}
        ):
            return None

        # the monomers share the dimer basis set (ghost atoms), as in Psi4's SAPT
        wfn = self.wavefunctions["A"]
        aux = None
        if cpscf_type == "DF" and df_basis is None:
            aux = self._df_basis_scf
        elif cpscf_type == "DF":
            # a named fitting basis covers every atom; Psi4 only needs a valid orbital
            # basis name next to it, the dimer's, not the options at this solve
            aux = psi4.core.BasisSet.build(
                wfn.molecule(),
                "DF_BASIS_SCF",
                df_basis,
                "JKFIT",
                wfn.basisset().name(),
                puream=wfn.basisset().has_puream(),
            )

        tstart = time.time()
        logger.info(f"Initialize the {cpscf_type} J/K object of CPSCF...")
        jk = psi4.core.JK.build(
            wfn.basisset(),
            aux=aux,
            jk_type=cpscf_type,
            do_wK=wfn.functional().is_x_lrc(),
            memory=self.jk_memory,
        )

        # `initialize_jk` also puts the J/K object on the wavefunction
        scf_jk = wfn.jk()
        try:
            wfn.initialize_jk(self.jk_memory, jk=jk)
        finally:
            wfn.set_jk(scf_jk)

        fitting = f", fitting basis {aux.name()} ({aux.nbf()} functions)" if aux else ""
        logger.info(
            f"...initialized {jk.name()}{fitting} in {time.time() - tstart:.2f} seconds."
        )

        return jk

    @staticmethod
    @contextmanager
    def timer() -> Iterator[None]:
        """Context between Psi4's `tstart` and `tstop`, which write the time it took into
        Psi4's output file. Psi4 has one such timer: a nested context restarts it."""

        psi4.core.tstart()
        # no tstop for a calculation that raised, as before propSAPT had backends
        yield
        psi4.core.tstop()
