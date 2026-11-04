"""
Modified version of helper_SAPT.py from Psi4NumPy repository.
Contains helper classes and functions for SAPT with MO density-fitting.

Modified by: Bartosz Tyrcha

Original authors and license info:
__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2015-12-01"

References:
- Equations and algorithms from [Szalewicz:2005:43], [Jeziorski:1994:1887],
[Szalewicz:2012:254], and [Hohenstein:2012:304]

License:
Copyright (c) 2014-2018, The Psi4NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the Psi4NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import time
from typing import overload, Literal, Any
from functools import wraps
import numpy as np
import opt_einsum as oe
import psi4
from psi4.driver.p4util import message_box, cg_solver, OptionsState, OptionsStateCM

from .options import PROPSAPT_OPTIONS


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
    """Memory of the JK object `helper_SAPT` builds, in doubles, the unit of
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


class helper_SAPT(object):
    def __init__(self, dimer, reference="RHF", memory=None, **kwargs):

        # propSAPT is all-electron: reject frozen core instead of ignoring it
        if psi4.core.get_global_option("FREEZE_CORE") not in ("FALSE", "0"):
            psi4.core.clean()
            raise ValueError(
                "propSAPT does not support frozen core; set FREEZE_CORE to false."
            )

        # Verify reference
        if reference not in ["RHF", "ROHF", "UHF", "RKS", "UKS"]:
            psi4.core.clean()
            raise ValueError(f"Unsupported reference type '{reference}'.")

        if reference in ["ROHF", "UHF", "UKS"]:
            # NOTE: Future upgrades
            psi4.core.clean()
            raise ValueError(f"Reference '{reference}' not implemented yet.")

        # Check if the functional is specified for DFT
        dft_functional = kwargs.get(
            "functional", PROPSAPT_OPTIONS["default_functional"]
        )

        # Initialize time
        tinit_start = time.time()
        psi4.core.print_out("\nInitializing SAPT object...\n")
        psi4.core.print_out(
            f"""         ---------------------------------------------------------
                             helper_SAPT_DF
                              {reference} Reference
         ---------------------------------------------------------
         """
        )
        psi4.core.print_out(f"\nSelected reference is {reference}\n")

        # Set a few crucial attributes
        self.reference = reference.upper()

        # DFT settings, recorded in the output artifacts
        self.functional = dft_functional if self.reference == "RKS" else None
        self.grac_A = kwargs.get("grac_A")
        self.grac_B = kwargs.get("grac_B")
        dimer.reset_point_group("c1")
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()

        if nfrags < 2:
            psi4.core.clean()
            raise ValueError(f"Found {nfrags:d} fragments, must be at least 2.")
        elif nfrags > 2:
            psi4.core.print_out(
                f"Found {nfrags:d} fragments, using first 2 as monomers, "
                "and the rest as extra basis functions (ghosts).\n"
            )

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, [i for i in range(2, nfrags + 1)])
        monomerA.set_name("monomerA")
        monomerB = dimer.extract_subsets(2, [1] + [i for i in range(3, nfrags + 1)])
        monomerB.set_name("monomerB")
        self.mult_A = monomerA.multiplicity()
        self.mult_B = monomerB.multiplicity()

        # closed-shell references only, fail before any SCF is run
        for label, mult in (("A", self.mult_A), ("B", self.mult_B)):
            if mult != 1:
                psi4.core.clean()
                raise ValueError(
                    f"Monomer {label} has multiplicity {mult}, "
                    f"{reference} reference requires singlet monomers."
                )

        # Compute monomer properties
        psi4.core.print_out(message_box(f"Monomer A {reference}") + "\n")
        tstart = time.time()
        if reference == "RHF":
            self.rhfA, self.wfnA = psi4_energy(
                "SCF", return_wfn=True, molecule=monomerA
            )
        elif reference == "RKS":
            with OptionsStateCM(["dft_grac_shift"]):
                if kwargs.get("grac_A"):
                    psi4.set_options({"dft_grac_shift": kwargs["grac_A"]})
                else:
                    psi4.core.print_out(
                        "\n*** WARNING!: GRAC shift for monomer A not specified! ***\n"
                        "Running without GRAC.\n"
                    )
                self.dftA, self.wfnA = psi4_energy(
                    dft_functional, return_wfn=True, molecule=monomerA
                )
        self.V_A = np.asarray(
            psi4.core.MintsHelper(self.wfnA.basisset()).ao_potential()
        )
        psi4.core.print_out(
            f"\n{reference} for monomer A finished in {time.time() - tstart:.2f} seconds.\n\n"
        )

        psi4.core.print_out(message_box(f"Monomer B {reference}") + "\n")
        tstart = time.time()
        if reference == "RHF":
            self.rhfB, self.wfnB = psi4_energy(
                "SCF", return_wfn=True, molecule=monomerB
            )
        elif reference == "RKS":
            with OptionsStateCM(["dft_grac_shift"]):
                if kwargs.get("grac_B"):
                    psi4.set_options({"dft_grac_shift": kwargs["grac_B"]})
                else:
                    psi4.core.print_out(
                        "\n*** WARNING!: GRAC shift for monomer B not specified! ***\n"
                        "Running without GRAC.\n"
                    )
                self.dftB, self.wfnB = psi4_energy(
                    dft_functional, return_wfn=True, molecule=monomerB
                )
        self.V_B = np.asarray(
            psi4.core.MintsHelper(self.wfnB.basisset()).ao_potential()
        )
        psi4.core.print_out(
            f"\n{reference} for monomer B finished in {time.time() - tstart:.2f} seconds.\n\n"
        )

        # Setup a few variables
        self.jk_memory = jk_memory_doubles(memory)  # in doubles
        self.memory = self.jk_memory * 8 / 1e9  # the same in GB
        self.nmo = self.wfnA.nmo()
        self.nbf = self.wfnA.basisset().nbf()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A

        if reference == "ROHF":
            self.idx_A = ["i", "a", "r"]
            self.nsocc_A = self.wfnA.soccpi()[0]
            occA = self.ndocc_A + self.nsocc_A

        elif reference == "UHF" or reference == "UKS":
            psi4.core.clean()
            raise NotImplementedError(f"Reference {reference} not implemented yet.")

        elif reference == "RHF" or reference == "RKS":
            self.idx_A = ["a", "r"]
            self.nsocc_A = 0
            occA = self.ndocc_A

        else:
            psi4.core.clean()
            raise ValueError(f"Unknown reference type: {reference}.")

        self.C_A = np.asarray(self.wfnA.Ca())
        self.Co_A = self.C_A[:, : self.ndocc_A]
        self.Ca_A = self.C_A[:, self.ndocc_A : occA]
        self.Cv_A = self.C_A[:, occA:]
        self.eps_A = np.asarray(self.wfnA.epsilon_a())

        # Monomer B
        self.nuc_rep_B = monomerB.nuclear_repulsion_energy()
        self.ndocc_B = self.wfnB.doccpi()[0]
        self.nvirt_B = self.nmo - self.ndocc_B

        if reference == "ROHF":
            self.idx_B = ["j", "b", "s"]
            self.nsocc_B = self.wfnB.soccpi()[0]
            occB = self.ndocc_B + self.nsocc_B

        elif reference == "UHF" or reference == "UKS":
            psi4.core.clean()
            raise NotImplementedError(f"Reference {reference} not implemented yet.")

        elif reference == "RHF" or reference == "RKS":
            self.idx_B = ["b", "s"]
            self.nsocc_B = 0
            occB = self.ndocc_B

        else:
            psi4.core.clean()
            raise ValueError(f"Unknown reference type: {reference}.")

        self.C_B = np.asarray(self.wfnB.Ca())
        self.Co_B = self.C_B[:, : self.ndocc_B]
        self.Ca_B = self.C_B[:, self.ndocc_B : occB]
        self.Cv_B = self.C_B[:, occB:]
        self.eps_B = np.asarray(self.wfnB.epsilon_a())

        # Dimer
        self.nuc_rep = (
            dimer.nuclear_repulsion_energy() - self.nuc_rep_A - self.nuc_rep_B
        )
        self.vt_nuc_rep = self.nuc_rep / (
            (2 * self.ndocc_A + self.nsocc_A) * (2 * self.ndocc_B + self.nsocc_B)
        )

        # Make slice, orbital, and size dictionaries
        if reference == "ROHF":
            self.slices = {
                "i": slice(0, self.ndocc_A),
                "a": slice(self.ndocc_A, occA),
                "r": slice(occA, None),
                "j": slice(0, self.ndocc_B),
                "b": slice(self.ndocc_B, occB),
                "s": slice(occB, None),
            }

            self.orbitals = {
                "i": self.Co_A,
                "a": self.Ca_A,
                "r": self.Cv_A,
                "j": self.Co_B,
                "b": self.Ca_B,
                "s": self.Cv_B,
            }

            self.sizes = {
                "i": self.ndocc_A,
                "a": self.nsocc_A,
                "r": self.nvirt_A,
                "j": self.ndocc_B,
                "b": self.nsocc_B,
                "s": self.nvirt_B,
            }

        elif reference == "RHF" or reference == "RKS":
            self.slices = {
                "a": slice(0, self.ndocc_A),
                "r": slice(occA, None),
                "b": slice(0, self.ndocc_B),
                "s": slice(occB, None),
            }

            self.orbitals = {
                "a": self.Co_A,
                "r": self.Cv_A,
                "b": self.Co_B,
                "s": self.Cv_B,
            }

            self.sizes = {
                "a": self.ndocc_A,
                "r": self.nvirt_A,
                "b": self.ndocc_B,
                "s": self.nvirt_B,
            }

        ### Number of basis functions
        psi4.core.print_out("\nNumber of molecular orbitals per type:\n")
        psi4.core.print_out("-" * 38 + "\n")
        psi4.core.print_out("".join([f"{dim:>5}" for dim in self.sizes]) + "\n")
        psi4.core.print_out("-" * 38 + "\n")
        psi4.core.print_out("".join([f"{dim:5d}" for _, dim in self.sizes.items()]))
        psi4.core.print_out("\n\n")

        ### Compute DF ERIs for dimer basis
        tstart = time.time()
        psi4.core.print_out("\nCompute DF integrals for dimer-centered basis...\n")

        self.dimer_wfn = psi4.core.Wavefunction.build(
            dimer, psi4.core.get_global_option("BASIS")
        )
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints

        # DF integrals in AO
        aux_basis = psi4.core.BasisSet.build(
            self.dimer_wfn.molecule(),
            target=psi4.core.get_global_option("DF_BASIS_SAPT"),
            other=psi4.core.get_global_option("DF_BASIS_SAPT"),
            fitrole="RIFIT",
        )
        aux_basis.print_out()
        self.nao_aux = aux_basis.nao()
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
        self.Qpq = np.array(oe.contract("QP,Ppq->Qpq", metric, Ppq))

        psi4.core.print_out(
            f"... DF integrals finished in {time.time() - tstart:.2f} seconds.\n"
        )

        # JK object
        psi4.core.print_out("\nInitialize JK objects...\n")
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

        self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
        self.jk.set_memory(self.jk_memory)
        self.jk.initialize()
        psi4.core.print_out(
            f"\n...initialized JK objects in {time.time() - tstart:.2f} seconds."
        )

        self.J_A, self.K_A = self.compute_jk(self.Co_A, self.Co_A)
        self.J_B, self.K_B = self.compute_jk(self.Co_B, self.Co_B)

        # Overlap matrix in AO
        self.S = np.asarray(self.mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = oe.contract("ui,vj,uv->ij", self.C_B, self.C_B, self.V_A)
        self.V_A_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_A)
        self.V_B_AA = oe.contract("ui,vj,uv->ij", self.C_A, self.C_A, self.V_B)
        self.V_B_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_B)

        self.S_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.S)

        hsapt_init_time = time.time() - tinit_start
        psi4.core.print_out(
            f"\n...finished initializing SAPT object in {hsapt_init_time:5.2f} seconds.\n\n"
        )

    # Gets transformed DF integrals in MO
    def df_ints(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"S: string {string} does not have 2 elements.")

        return np.array(
            oe.contract(
                "Qpq,pa,qr->Qar",
                self.Qpq,
                self.orbitals[string[0]],
                self.orbitals[string[1]],
            )
        )

    # Grab MO overlap matrices
    def s(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"S: string {string} does not have 2 elements.")

        # Compute on the fly, an empty orbital block gives a correctly shaped empty array
        return (self.orbitals[string[0]].T).dot(self.S).dot(self.orbitals[string[1]])

    # Grab epsilons, reshape if requested
    def eps(self, string, dim=1):
        if len(string) != 1:
            psi4.core.clean()
            raise ValueError(f"Epsilon: string {string} does not have 1 element.")

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == "i") or (string == "a") or (string == "r"):
            return self.eps_A[self.slices[string]].reshape(shape)
        elif (string == "j") or (string == "b") or (string == "s"):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            psi4.core.clean()
            raise ValueError(f"Unknown orbital type in eps: {string}.")

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"Potential: string {string} does not have 2 elements.")

        # Two separate cases
        if side == "A":
            # Compute on the fly
            return (
                (self.orbitals[string[0]].T).dot(self.V_A).dot(self.orbitals[string[1]])
            )

        elif side == "B":
            # Compute on the fly
            return (
                (self.orbitals[string[0]].T).dot(self.V_B).dot(self.orbitals[string[1]])
            )
        else:
            psi4.core.clean()
            raise ValueError(
                f"helper_SAPT.potential side must be either A or B, not {side}."
            )

    @overload
    def cpscf(
        self, monomer: Any, ind: Literal[False] = False, **kwargs
    ) -> np.ndarray: ...

    @overload
    def cpscf(
        self, monomer: Any, ind: Literal[True], **kwargs
    ) -> tuple[np.ndarray, float]: ...

    def cpscf(
        self, monomer, ind=False, **kwargs
    ) -> np.ndarray | tuple[np.ndarray, float]:
        """
        Coupled perturbed HF or KS calculations.

        Solves for the response amplitudes of `monomer` ("A" or "B") to the (ov) matrix
        `perturbation`, by default the electrostatic potential of the other monomer.
        `r_convergence` and `maxiter` default to the propSAPT options. Several
        perturbations of one monomer are solved together, at lower cost, by
        `cpscf_batch`.

        Returns:
            np.ndarray | tuple[np.ndarray, float]: The (vo) amplitudes; with `ind=True`
                and the default perturbation, also the second-order induction energy.
        """

        self._check_cpscf(monomer)

        pert_ov = kwargs.get("perturbation", None)
        if pert_ov is None:
            # Construct Omega potential of the other monomer
            if monomer == "A":
                omegaB = self.V_B + 2 * self.J_B
                pert_ov = self.orbitals["a"].T @ omegaB @ self.orbitals["r"]
            else:
                omegaA = self.V_A + 2 * self.J_A
                pert_ov = self.orbitals["b"].T @ omegaA @ self.orbitals["s"]

        (t,) = self.cpscf_batch(
            monomer,
            [pert_ov],
            r_convergence=kwargs.get("r_convergence", None),
            maxiter=kwargs.get("maxiter", None),
        )

        if ind and kwargs.get("perturbation", None) is None:
            # if perturbation is not given pert_ov = omega_ov
            e20_ind_r = 2 * np.einsum("vo,ov", t, pert_ov)
            return t, e20_ind_r

        return t

    def cpscf_batch(
        self,
        monomer: str,
        perturbations: list[np.ndarray],
        r_convergence: float | None = None,
        maxiter: int | None = None,
    ) -> list[np.ndarray]:
        """
        Coupled perturbed HF or KS calculations for several perturbations of one monomer.

        Solves the CPSCF equations of `monomer` for every (ov) matrix in `perturbations`
        in one conjugate-gradient run. Each iteration builds the Hessian products of all
        unconverged vectors with one J/K call, which costs much less than a call per
        vector. Each vector converges on its own, to the same amplitudes as `cpscf`.

        Args:
            monomer (str): "A" or "B".
            perturbations (list[np.ndarray]): (ov) perturbation matrices.
            r_convergence (float | None, optional): Residual convergence threshold.
                Defaults to the `r_convergence` option.
            maxiter (int | None, optional): Maximum number of iterations. Defaults to the
                `cpscf_maxiter` option.

        Returns:
            list[np.ndarray]: The (vo) amplitudes, in the order of `perturbations`.
        """

        self._check_cpscf(monomer)

        if r_convergence is None:
            r_convergence = PROPSAPT_OPTIONS["r_convergence"]
        if maxiter is None:
            maxiter = PROPSAPT_OPTIONS["cpscf_maxiter"]

        if len(perturbations) == 0:
            return []

        # NOTE: Changed the monomer naming scheme:
        # 'monomer' refers now to the one that returned amplitudes are for.
        occ, vir, wfn = ("a", "r", self.wfnA) if monomer == "A" else ("b", "s", self.wfnB)
        no = self.sizes[occ]
        nv = self.sizes[vir]

        # Get the orbital energy differences
        eps_ov = psi4.core.Matrix.from_array(self.eps(occ, dim=2) - self.eps(vir))

        # apply hessian func to every active vector in one call
        def _hess_x(x_vec, act_mask):
            active = [x for x, on in zip(x_vec, act_mask) if on]
            products = iter(wfn.cphf_Hx(active) if active else [])
            return [next(products) if on else False for on in act_mask]

        # preconditioner - applies denominator (for faster convergence)
        def _apply_precon(x_vec, act_mask):
            preconditioned = []
            for x, on in zip(x_vec, act_mask):
                if on:
                    p = x.clone()
                    p.apply_denominator(eps_ov)
                    preconditioned.append(p)
                else:
                    preconditioned.append(False)
            return preconditioned

        # solve cpscf
        x_vec, _ = cg_solver(
            [psi4.core.Matrix.from_array(pert_ov) for pert_ov in perturbations],
            _hess_x,
            _apply_precon,
            printlvl=2,
            maxiter=maxiter,
            rcond=r_convergence,
        )

        # We want to return (vo) matrices
        return [np.array(x).reshape(no, nv).T for x in x_vec]

    def _check_cpscf(self, monomer):
        """Raises ValueError unless CPSCF is implemented for this reference and `monomer`."""

        if self.reference in ["ROHF", "UHF", "UKS"]:
            psi4.core.clean()
            raise ValueError(
                f"CPSCF solver for a {self.reference} reference not implemented yet."
            )
        if self.reference not in ["RHF", "RKS"]:
            psi4.core.clean()
            raise ValueError(f"Unknown reference type: {self.reference}.")
        if monomer not in ["A", "B"]:
            psi4.core.clean()
            raise ValueError(f"'{monomer}' is not a valid monomer for CPSCF.")

    def chain_dot(self, *dot_list):
        result = dot_list[0]
        for x in range(len(dot_list) - 1):
            result = np.dot(result, dot_list[x + 1])
        return result

    def transform_ao_to_mo(self, x, string):
        """
        AO -> MO transformation of 2-indexed array 'x'
        into shape of 'string' in terms of molecular
        orbitals.
        """

        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"S: string {string} does not have 2 elements.")

        return self.orbitals[string[0]].T.dot(x).dot(self.orbitals[string[1]])

    def compute_jk(self, Cleft, Cright, tensor=None):

        if self.reference == "ROHF":
            psi4.core.clean()
            raise NotImplementedError(
                "JK generation not yet implemented for ROHF reference."
            )

        return_single = False
        if not isinstance(Cleft, (list, tuple)):
            Cleft = [Cleft]
            return_single = True
        if not isinstance(Cright, (list, tuple)):
            Cright = [Cright]
            return_single = True
        if (not isinstance(tensor, (list, tuple))) and (tensor is not None):
            tensor = [tensor]
            return_single = True

        if len(Cleft) != len(Cright):
            psi4.core.clean()
            raise ValueError("Cleft list is not the same length as Cright list")

        zero_append = []
        num_compute = 0

        for num, (Cl, Cr) in enumerate(zip(Cleft, Cright)):
            if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
                zero_append.append(num)
                continue

            if tensor is not None:
                mol = Cl.shape[1]
                mor = Cr.shape[1]

                if (tensor[num].shape[0] != mol) or (tensor[num].shape[1] != mor):
                    psi4.core.clean()
                    raise ValueError(
                        f"compute_sapt_JK: Tensor size does not match Cl ({mol}) "
                        f"/Cr ({mor}) : {tensor[num].shape}"
                    )
                if mol < mor:
                    Cl = np.dot(Cl, tensor[num])
                else:
                    Cr = np.dot(Cr, tensor[num].T)

            Cl = psi4.core.Matrix.from_array(Cl)
            Cr = psi4.core.Matrix.from_array(Cr)

            self.jk.C_left_add(Cl)
            self.jk.C_right_add(Cr)
            num_compute += 1

        self.jk.compute()

        J_list = []
        K_list = []
        for num in range(num_compute):
            J_list.append(np.array(self.jk.J()[num]))
            K_list.append(np.array(self.jk.K()[num]))

        self.jk.C_clear()

        z = np.zeros((self.nbf, self.nbf))
        for num in zero_append:
            J_list.insert(num, z)
            K_list.insert(num, z)

        if return_single:
            return J_list[0], K_list[0]
        else:
            return J_list, K_list


# End SAPT helper
