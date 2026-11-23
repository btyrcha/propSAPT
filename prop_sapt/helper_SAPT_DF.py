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

import logging
import time
from typing import overload, Literal, Any
import numpy as np
import opt_einsum as oe

from .backends.base import Backend
from .cpscf_solver import conjugate_gradient
from .options import PROPSAPT_OPTIONS, check_cpscf_type, check_df_basis_cpscf

logger = logging.getLogger(__name__)


class helper_SAPT(object):
    """
    The arrays of a dimer that the SAPT contractions use, from a reference calculation
    of the dimer by a backend (see `prop_sapt.backends`): orbitals, orbital energies,
    potentials and overlaps of the monomers, density-fitted integrals, and the J/K builds
    and CPSCF solves.

    Args:
        backend (Backend): The reference calculation, kept as `backend`.
    """

    def __init__(self, backend: Backend):

        reference = backend.reference

        # Initialize time
        tinit_start = time.time()
        logger.info("Initializing SAPT object...")
        logger.info(
            "         ---------------------------------------------------------\n"
            "                             helper_SAPT_DF\n"
            f"                              {reference} Reference\n"
            "         ---------------------------------------------------------"
        )
        logger.info(f"Selected reference is {reference}")

        # Set a few crucial attributes
        self.backend = backend
        self.reference = reference

        # DFT settings, recorded in the output artifacts
        self.functional = backend.functional
        self.grac_A = backend.grac_A
        self.grac_B = backend.grac_B

        self.fragments = backend.fragments
        self.mult_A = backend.fragments.A.multiplicity
        self.mult_B = backend.fragments.B.multiplicity

        monomer_A = backend.monomers["A"]
        monomer_B = backend.monomers["B"]

        # SCF energies of the monomers
        if reference == "RHF":
            self.rhfA, self.rhfB = monomer_A.energy, monomer_B.energy
        elif reference == "RKS":
            self.dftA, self.dftB = monomer_A.energy, monomer_B.energy

        self.V_A = monomer_A.V
        self.V_B = monomer_B.V

        # Setup a few variables
        self.nmo = monomer_A.nmo
        self.nbf = backend.nbf

        # Monomer A
        self.nuc_rep_A = monomer_A.nuclear_repulsion
        self.ndocc_A = monomer_A.ndocc
        self.nvirt_A = self.nmo - self.ndocc_A

        if reference == "ROHF":
            self.idx_A = ["i", "a", "r"]
            self.nsocc_A = monomer_A.nsocc
            occA = self.ndocc_A + self.nsocc_A

        elif reference == "UHF" or reference == "UKS":
            raise NotImplementedError(f"Reference {reference} not implemented yet.")

        elif reference == "RHF" or reference == "RKS":
            self.idx_A = ["a", "r"]
            self.nsocc_A = 0
            occA = self.ndocc_A

        else:
            raise ValueError(f"Unknown reference type: {reference}.")

        self.C_A = monomer_A.C
        self.Co_A = self.C_A[:, : self.ndocc_A]
        self.Ca_A = self.C_A[:, self.ndocc_A : occA]
        self.Cv_A = self.C_A[:, occA:]
        self.eps_A = monomer_A.eps

        # Monomer B
        self.nuc_rep_B = monomer_B.nuclear_repulsion
        self.ndocc_B = monomer_B.ndocc
        self.nvirt_B = monomer_B.nmo - self.ndocc_B

        if reference == "ROHF":
            self.idx_B = ["j", "b", "s"]
            self.nsocc_B = monomer_B.nsocc
            occB = self.ndocc_B + self.nsocc_B

        elif reference == "UHF" or reference == "UKS":
            raise NotImplementedError(f"Reference {reference} not implemented yet.")

        elif reference == "RHF" or reference == "RKS":
            self.idx_B = ["b", "s"]
            self.nsocc_B = 0
            occB = self.ndocc_B

        else:
            raise ValueError(f"Unknown reference type: {reference}.")

        self.C_B = monomer_B.C
        self.Co_B = self.C_B[:, : self.ndocc_B]
        self.Ca_B = self.C_B[:, self.ndocc_B : occB]
        self.Cv_B = self.C_B[:, occB:]
        self.eps_B = monomer_B.eps

        # Dimer
        self.nuc_rep = backend.nuclear_repulsion - self.nuc_rep_A - self.nuc_rep_B
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
        rule = "-" * 38
        labels = "".join(f"{name:>5}" for name in self.sizes)
        counts = "".join(f"{dim:5d}" for dim in self.sizes.values())
        logger.info(
            f"Number of molecular orbitals per type:\n{rule}\n{labels}\n{rule}\n{counts}"
        )

        ### DF ERIs and the overlap of the dimer basis
        self.Qpq = backend.Qpq
        self.nao_aux = self.Qpq.shape[0]
        self.S = backend.S

        self.J_A, self.K_A = self.compute_jk(self.Co_A, self.Co_A)
        self.J_B, self.K_B = self.compute_jk(self.Co_B, self.Co_B)

        # J/K object of the CPSCF solves, see `build_cpscf_jk`; None uses the monomer SCF's
        self.cpscf_jk = None

        # Save additional rank 2 tensors
        self.V_A_BB = oe.contract("ui,vj,uv->ij", self.C_B, self.C_B, self.V_A)
        self.V_A_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_A)
        self.V_B_AA = oe.contract("ui,vj,uv->ij", self.C_A, self.C_A, self.V_B)
        self.V_B_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_B)

        self.S_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.S)

        hsapt_init_time = time.time() - tinit_start
        logger.info(
            f"...finished initializing SAPT object in {hsapt_init_time:5.2f} seconds."
        )

    # Gets transformed DF integrals in MO
    def df_ints(self, string):
        if len(string) != 2:
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
            raise ValueError(f"S: string {string} does not have 2 elements.")

        # Compute on the fly, an empty orbital block gives a correctly shaped empty array
        return (self.orbitals[string[0]].T).dot(self.S).dot(self.orbitals[string[1]])

    # Grab epsilons, reshape if requested
    def eps(self, string, dim=1):
        if len(string) != 1:
            raise ValueError(f"Epsilon: string {string} does not have 1 element.")

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == "i") or (string == "a") or (string == "r"):
            return self.eps_A[self.slices[string]].reshape(shape)
        elif (string == "j") or (string == "b") or (string == "s"):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            raise ValueError(f"Unknown orbital type in eps: {string}.")

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
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
        in one conjugate-gradient run (`prop_sapt.cpscf_solver`). Each iteration builds
        the Hessian products of all unconverged vectors with one J/K call, which costs
        much less than a call per vector. Each vector converges on its own, to the same
        amplitudes as `cpscf`.

        The amplitudes t of a perturbation omega solve H t = -omega, with H the orbital
        Hessian of `orbital_hessian_product`; uncoupled, they are
        omega / (eps_occ - eps_vir). The J and K matrices come from the J/K object of
        the monomer SCF, or from `cpscf_jk` if it is set (see `build_cpscf_jk`).

        Args:
            monomer (str): "A" or "B".
            perturbations (list[np.ndarray]): (ov) perturbation matrices.
            r_convergence (float | None, optional): Convergence threshold of the
                relative residual. Defaults to the `r_convergence` option.
            maxiter (int | None, optional): Maximum number of iterations. Defaults to the
                `cpscf_maxiter` option.

        Returns:
            list[np.ndarray]: The (vo) amplitudes, in the order of `perturbations`.

        Warns:
            RuntimeWarning: Raised if some amplitudes have not converged.
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
        occ, vir = ("a", "r") if monomer == "A" else ("b", "s")

        amplitudes = conjugate_gradient(
            [-np.asarray(pert_ov) for pert_ov in perturbations],
            lambda vectors: self.orbital_hessian_product(monomer, vectors),
            self.eps(vir) - self.eps(occ, dim=2),
            r_convergence=r_convergence,
            maxiter=maxiter,
            title=f"CPSCF of monomer {monomer}",
        )

        # We want to return (vo) matrices
        return [t.T for t in amplitudes]

    def orbital_hessian_product(
        self, monomer: str, vectors: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Products of the orbital Hessian of `monomer` with (ov) `vectors`, the left-hand
        side of its CPSCF equations (see `cpscf_batch`).

        The Hessian is the symmetric positive definite one of a real singlet
        perturbation: for RHF,
        H_{ar,a'r'} = (eps_r - eps_a) delta_{aa'} delta_{rr'}
        + 4 (ar|a'r') - (ar'|a'r) - (aa'|rr'),
        and for RKS the exchange-correlation kernel too. The backend builds the
        products (`Backend.orbital_hessian_product`), with the J/K of the monomer SCF,
        or of `cpscf_jk` if it is set (see `build_cpscf_jk`).

        Args:
            monomer (str): "A" or "B".
            vectors (list[np.ndarray]): (ov) vectors.

        Returns:
            list[np.ndarray]: The (ov) products, in the order of `vectors`.
        """

        self._check_cpscf(monomer)

        if len(vectors) == 0:
            return []

        return self.backend.orbital_hessian_product(monomer, vectors, self.cpscf_jk)

    def build_cpscf_jk(
        self, cpscf_type: str | None, df_basis: str | None = None
    ) -> object:
        """
        J/K object for CPSCF solves of type `cpscf_type`, shared by both monomers, from
        the backend (`Backend.build_cpscf_jk`). `cpscf_batch` uses it in place of the
        monomer SCF's J/K while it is `cpscf_jk`.

        Args:
            cpscf_type (str | None): "DIRECT" or "DF", case-insensitive, see the
                `cpscf_type` option; None stands for the monomer SCF's J/K objects.
            df_basis (str | None, optional): Fitting basis of "DF". Defaults to the
                backend's default, for Psi4 the dimer's `DF_BASIS_SCF` fitting basis
                (see `fitting_bases`).

        Returns:
            object: The J/K object, or None if the monomer SCF's J/K objects are of
                this type already ("DF" only without `df_basis`).
        """

        cpscf_type = check_cpscf_type(cpscf_type)
        df_basis = check_df_basis_cpscf(df_basis)

        if cpscf_type is None:
            return None

        return self.backend.build_cpscf_jk(cpscf_type, df_basis)

    def _check_cpscf(self, monomer):
        """Raises ValueError unless CPSCF is implemented for this reference and `monomer`."""

        if self.reference in ["ROHF", "UHF", "UKS"]:
            raise ValueError(
                f"CPSCF solver for a {self.reference} reference not implemented yet."
            )
        if self.reference not in ["RHF", "RKS"]:
            raise ValueError(f"Unknown reference type: {self.reference}.")
        if monomer not in ["A", "B"]:
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
            raise ValueError(f"S: string {string} does not have 2 elements.")

        return self.orbitals[string[0]].T.dot(x).dot(self.orbitals[string[1]])

    def compute_jk(self, Cleft, Cright, tensor=None):
        """
        J and K matrices of generalized densities D = C_left C_right^T, or
        D = C_left M C_right^T with a matrix M of `tensor`, from the backend
        (`Backend.compute_jk`).

        M is folded into the side with fewer columns. A pair with no columns on a side
        gets zero matrices without reaching the backend.

        Args:
            Cleft (np.ndarray | list[np.ndarray]): C_left matrix, (nbf, k), or a list.
            Cright (np.ndarray | list[np.ndarray]): C_right matrix, (nbf, l), or a list
                as long as `Cleft`.
            tensor (np.ndarray | list[np.ndarray] | None, optional): M matrix, (k, l),
                or a list of them, one per pair. Defaults to None, no M.

        Returns:
            tuple: J and K, AO matrices if one pair is given and lists of them otherwise.
        """

        if self.reference == "ROHF":
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
            raise ValueError("Cleft list is not the same length as Cright list")

        zero_append = []
        lefts = []
        rights = []

        for num, (Cl, Cr) in enumerate(zip(Cleft, Cright)):
            if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
                zero_append.append(num)
                continue

            if tensor is not None:
                mol = Cl.shape[1]
                mor = Cr.shape[1]

                if (tensor[num].shape[0] != mol) or (tensor[num].shape[1] != mor):
                    raise ValueError(
                        f"compute_sapt_JK: Tensor size does not match Cl ({mol}) "
                        f"/Cr ({mor}) : {tensor[num].shape}"
                    )
                if mol < mor:
                    Cl = np.dot(Cl, tensor[num])
                else:
                    Cr = np.dot(Cr, tensor[num].T)

            lefts.append(Cl)
            rights.append(Cr)

        J_list = []
        K_list = []
        if lefts:
            J_list, K_list = self.backend.compute_jk(lefts, rights)
            J_list, K_list = list(J_list), list(K_list)

        z = np.zeros((self.nbf, self.nbf))
        for num in zero_append:
            J_list.insert(num, z)
            K_list.insert(num, z)

        if return_single:
            return J_list[0], K_list[0]
        else:
            return J_list, K_list


# End SAPT helper
