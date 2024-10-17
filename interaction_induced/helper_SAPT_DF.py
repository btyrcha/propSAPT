"""
Modified version of helper_SAPT.py form Psi4Numpy repository.
Contains helper classes and functions for SAPT with MO density-fitting in Psi4NumPy.

Original authors and license info:
__authors__ = "Daniel G. A. Smith"
__credits__ = ["Daniel G. A. Smith"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2015-12-01"

References:
- Equations and algorithms from [Szalewicz:2005:43], [Jeziorski:1994:1887],
[Szalewicz:2012:254], and [Hohenstein:2012:304]
"""

import time
import numpy as np
import opt_einsum as oe
import psi4
from psi4.driver.p4util import message_box, cg_solver


class helper_SAPT(object):
    def __init__(self, dimer, reference="RHF", memory=8, **kwargs):

        # This is denisty_fitted version of helper_SAPT
        self.is_density_fitted = True

        # Verify reference
        if reference not in ["RHF", "ROHF", "UHF", "RKS", "UKS"]:
            psi4.core.clean()
            raise ValueError(f"Unsupporsted reference type '{reference}'.")

        if reference in ["ROHF", "UHF", "UKS"]:
            # NOTE: Future upgrades
            psi4.core.clean()
            raise ValueError(f"Reference '{reference}' not implemented yet.")

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
        psi4.core.print_out(f"\nSelcted reference is {reference}\n")

        # Set a few crucial attributes
        self.reference = reference.upper()
        dimer.reset_point_group("c1")
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi4.core.clean()
            raise ValueError(f"Found {nfrags:d} fragments, must be 2.")

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name("monomerA")
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name("monomerB")
        self.mult_A = monomerA.multiplicity()
        self.mult_B = monomerB.multiplicity()

        # Compute monomer properties
        psi4.core.print_out(message_box(f"Monomer A {reference}") + "\n")
        tstart = time.time()
        if reference == "RHF":
            self.rhfA, self.wfnA = psi4.energy(
                "SCF", return_wfn=True, molecule=monomerA
            )
        elif reference == "RKS":
            if kwargs.get("grac_A"):
                psi4.set_options({"dft_grac_shift": kwargs["grac_A"]})
            else:
                psi4.core.print_out(
                    "\n*** WARNING!: GRAC shift for monomer A not specified! ***\n"
                )
            self.dftA, self.wfnA = psi4.energy(
                kwargs.get("functional", "pbe0"), return_wfn=True, molecule=monomerA
            )
        self.V_A = np.asarray(
            psi4.core.MintsHelper(self.wfnA.basisset()).ao_potential()
        )
        psi4.core.print_out(
            f"\n{reference} for monomer A finished in {time.time() - tstart:.2f} seconds.\n\n"
        )
        psi4.set_options({"dft_grac_shift": 0.0})  # reset GRAC shift back to zero

        psi4.core.print_out(message_box(f"Monomer B {reference}") + "\n")
        tstart = time.time()
        if reference == "RHF":
            self.rhfB, self.wfnB = psi4.energy(
                "SCF", return_wfn=True, molecule=monomerB
            )
        elif reference == "RKS":
            if kwargs.get("grac_B"):
                psi4.set_options({"dft_grac_shift": kwargs["grac_B"]})
            else:
                psi4.core.print_out(
                    "\n*** WARNING!: GRAC shift for monomer B not specified! ***\n"
                )
            self.dftB, self.wfnB = psi4.energy(
                kwargs.get("functional", "pbe0"), return_wfn=True, molecule=monomerB
            )
        self.V_B = np.asarray(
            psi4.core.MintsHelper(self.wfnB.basisset()).ao_potential()
        )
        psi4.core.print_out(
            f"\n{reference} for monomer B finished in {time.time() - tstart:.2f} seconds.\n\n"
        )
        psi4.set_options({"dft_grac_shift": 0.0})  # reset GRAC shift back to zero

        # Setup a few variables
        self.memory = memory
        self.nmo = self.wfnA.nmo()

        # Monomer A
        self.nuc_rep_A = monomerA.nuclear_repulsion_energy()
        self.ndocc_A = self.wfnA.doccpi()[0]
        self.nvirt_A = self.nmo - self.ndocc_A
        if reference == "ROHF":
            self.idx_A = ["i", "a", "r"]
            self.nsocc_A = self.wfnA.soccpi()[0]
            occA = self.ndocc_A + self.nsocc_A
        elif reference == "UHF" or reference == "UKS":
            # Olaboga co to bedzie!
            pass
        elif reference == "RHF" or reference == "RKS":
            self.idx_A = ["a", "r"]
            self.nsocc_A = 0
            occA = self.ndocc_A

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
            # Olaboga co to bedzie!
            pass
        elif reference == "RHF" or reference == "RKS":
            self.idx_B = ["b", "s"]
            self.nsocc_B = 0
            occB = self.ndocc_B

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
        psi4.core.print_out("".join([f"{self.sizes[dim]:5d}" for dim in self.sizes]))
        psi4.core.print_out("\n\n")

        ### Compute DF ERIs for dimer basis
        tstart = time.time()
        psi4.core.print_out("\nCompute DF integrals for dimer-centered basis...\n")

        self.dimer_wfn = psi4.core.Wavefunction.build(
            dimer, psi4.core.get_global_option("BASIS")
        )
        mints = psi4.core.MintsHelper(self.dimer_wfn.basisset())
        self.mints = mints

        # DF intergrals in AO
        # loading custom basis stored at keyword
        # NOTE: requiers testing
        aux_basis = psi4.core.BasisSet.build(
            self.dimer_wfn.molecule(),
            # key= "PSI4_KEYWORD",
            target=psi4.core.get_global_option("DF_BASIS_SAPT"),
            other=psi4.core.get_global_option("DF_BASIS_SAPT"),
            fitrole="RIFIT",
        )
        aux_basis.print_out()
        self.nao_aux = aux_basis.nao()
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

        # Overlap matrix in AO
        self.S = np.asarray(self.mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = oe.contract("ui,vj,uv->ij", self.C_B, self.C_B, self.V_A)
        self.V_A_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_A)
        self.V_B_AA = oe.contract("ui,vj,uv->ij", self.C_A, self.C_A, self.V_B)
        self.V_B_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_B)

        self.S_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.S)

        psi4.core.print_out(
            f"\n...finished initializing SAPT object in {time.time() - tinit_start:5.2f} seconds.\n\n"
        )

    # Gets transformed DF intergrals in MO
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

        for alpha in "ijab":
            if (alpha in string) and (self.sizes[alpha] == 0):
                return np.array([0]).reshape(1, 1)

        # Compute on the fly
        return (self.orbitals[string[0]].T).dot(self.S).dot(self.orbitals[string[1]])
        # return oe.contract('ui,vj,uv->ij', self.orbitals[string[0]], self.orbitals[string[1]], self.S)

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
            # return oe.contract('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_A)

        elif side == "B":
            # Compute on the fly
            return (
                (self.orbitals[string[0]].T).dot(self.V_B).dot(self.orbitals[string[1]])
            )
            # return oe.contract('ui,vj,uv->ij', self.orbitals[s1], self.orbitals[s2], self.V_B)
        else:
            psi4.core.clean()
            raise ValueError(
                f"helper_SAPT.potential side must be either A or B, not {side}."
            )

    def cpscf(self, monomer, ind=False, **kwargs):
        """
        Coupled perturbed HF or KS calculations.
        """

        if monomer not in ["A", "B"]:
            psi4.core.clean()
            raise ValueError(f"'{monomer}' is not a valid monomer for CHF.")

        if self.reference in ["ROHF", "UHF", "UKS"]:
            psi4.core.clean()
            raise ValueError(
                f"CPSCF solver for a {self.reference} reference not implemented yet."
            )

        if self.reference == "RHF" or self.reference == "RKS":

            # NOTE: Changed the monomer naming scheme:
            # 'monomer' refers now to the one that returned amplitudes are for.
            if monomer == "A":
                if kwargs.get("perturbation", None) is None:
                    # Construct Omega potential
                    vB_ar = self.V_B_AA[self.slices["a"], self.slices["r"]]
                    omega_ov = vB_ar + 2 * oe.contract(
                        "Qar,Qbb->ar", self.df_ints("ar"), self.df_ints("bb")
                    )
                    pert_ov = omega_ov
                else:
                    pert_ov = kwargs["perturbation"]

                # Get the orbital energy differences
                eps_ov = self.eps("a", dim=2) - self.eps("r")

                # Set number of electrons
                no = self.ndocc_A
                nv = self.nvirt_A

                # apply hessian func
                def _hess_x(x_vec, act_mask):
                    if act_mask[0]:
                        # monomer A
                        return self.wfnA.cphf_Hx([x_vec[0]])
                    else:
                        return [False]

            if monomer == "B":
                if kwargs.get("perturbation", None) is None:
                    # Construct Omega potential
                    vA_bs = self.V_A_BB[self.slices["b"], self.slices["s"]]
                    omega_ov = vA_bs + 2 * oe.contract(
                        "Qaa,Qbs->bs", self.df_ints("aa"), self.df_ints("bs")
                    )
                    pert_ov = omega_ov
                else:
                    pert_ov = kwargs["perturbation"]

                # Get the orbital energy differences
                eps_ov = self.eps("b", dim=2) - self.eps("s")

                # Set number of electrons
                no = self.ndocc_B
                nv = self.nvirt_B

                # apply hessian func
                def _hess_x(x_vec, act_mask):
                    if act_mask[0]:
                        # monomer B
                        return self.wfnB.cphf_Hx([x_vec[0]])
                    else:
                        return [False]

            # preconditioner - applies denominator (for faster convergence)
            def _apply_precon(x_vec, act_mask):
                if act_mask[0]:
                    p = x_vec[0].clone()
                    p.apply_denominator(psi4.core.Matrix.from_array(eps_ov))
                else:
                    p = False

                return [p]

            # solve cpscf
            t = cg_solver(
                [psi4.core.Matrix.from_array(pert_ov)],
                _hess_x,
                _apply_precon,
                printlvl=2,
                maxiter=20,
                rcond=1.0e-8,
            )

            # unwrap the lists
            t = np.array(t[0])[0]

            # We want to return a (vo) matrix
            t = t.reshape(no, nv).T

        if ind:
            e20_ind_r = 2 * np.einsum("vo,ov", t, omega_ov)
            return t, e20_ind_r
        else:
            return t

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


# End SAPT helper


class sapt_timer(object):
    """
    Simple timer object.
    """

    def __init__(self, name):
        self.name = name
        self.start = time.time()
        psi4.core.print_out(f"\nStarting {name}...")

    def stop(self):
        """
        Stops timer.
        """

        t = time.time() - self.start
        psi4.core.print_out(f"...{self.name} took a total of {t: .2f} seconds.")


def sapt_printer(line, value):
    """
    Prints out 'value' in mH and kcal/mol
    along a label given in 'line'.
    """

    spacer = " " * (20 - len(line))
    psi4.core.print_out(
        line + spacer + f"{value* 1000: 16.8f} mH  {value* 627.509: 16.8f} kcal/mol"
    )


# End SAPT helper
