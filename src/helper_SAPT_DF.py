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

import numpy as np
import opt_einsum as oe
import time
import psi4


class helper_SAPT(object):
    def __init__(self, dimer, df_basis, memory=8, algorithm="MO", reference="RHF"):
        psi4.core.print_out("\nInitializing SAPT object...\n")
        tinit_start = time.time()

        # Set a few crucial attributes
        self.alg = algorithm.upper()
        self.reference = reference.upper()
        dimer.reset_point_group("c1")
        dimer.fix_orientation(True)
        dimer.fix_com(True)
        dimer.update_geometry()
        nfrags = dimer.nfragments()
        if nfrags != 2:
            psi4.core.clean()
            raise ValueError("Found %d fragments, must be 2." % nfrags)

        # Grab monomers in DCBS
        monomerA = dimer.extract_subsets(1, 2)
        monomerA.set_name("monomerA")
        monomerB = dimer.extract_subsets(2, 1)
        monomerB.set_name("monomerB")
        self.mult_A = monomerA.multiplicity()
        self.mult_B = monomerB.multiplicity()

        # Compute monomer properties
        psi4.core.print_out(
            """
  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
  //              Monomer A HF         //
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//\n"""
        )
        tstart = time.time()
        self.rhfA, self.wfnA = psi4.energy("SCF", return_wfn=True, molecule=monomerA)
        self.V_A = np.asarray(
            psi4.core.MintsHelper(self.wfnA.basisset()).ao_potential()
        )
        psi4.core.print_out(
            "\nRHF for monomer A finished in %.2f seconds.\n" % (time.time() - tstart)
        )

        psi4.core.print_out(
            """
  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>//
  //              Monomer B HF         //
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<//\n"""
        )
        tstart = time.time()
        self.rhfB, self.wfnB = psi4.energy("SCF", return_wfn=True, molecule=monomerB)
        self.V_B = np.asarray(
            psi4.core.MintsHelper(self.wfnB.basisset()).ao_potential()
        )
        psi4.core.print_out(
            "\nRHF for monomer B finished in %.2f seconds.\n" % (time.time() - tstart)
        )

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
        else:
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
        else:
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

        else:
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
        aux_basis = psi4.core.BasisSet.build(
            self.dimer_wfn.molecule(),
            "DF_BASIS_SCF",
            psi4.core.get_global_option("DF_BASIS_SAPT"),
            "RIFIT",
            df_basis,
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
            "... DF integrals finished in %.2f seconds.\n" % (time.time() - tstart)
        )

        # Overlap matrix in AO
        self.S = np.asarray(self.mints.ao_overlap())

        # Save additional rank 2 tensors
        self.V_A_BB = oe.contract("ui,vj,uv->ij", self.C_B, self.C_B, self.V_A)
        self.V_A_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_A)
        self.V_B_AA = oe.contract("ui,vj,uv->ij", self.C_A, self.C_A, self.V_B)
        self.V_B_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.V_B)

        self.S_AB = oe.contract("ui,vj,uv->ij", self.C_A, self.C_B, self.S)

        if self.alg == "AO":
            tstart = time.time()
            aux_basis = psi4.core.BasisSet.build(
                self.dimer_wfn.molecule(),
                "DF_BASIS_SCF",
                psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                "JKFIT",
                df_basis,
                puream=self.dimer_wfn.basisset().has_puream(),
            )

            self.jk = psi4.core.JK.build(self.dimer_wfn.basisset(), aux_basis)
            self.jk.set_memory(int(memory * 1e9))
            self.jk.initialize()
            psi4.core.print_out(
                "\n...initialized JK objects in %5.2f seconds." % (time.time() - tstart)
            )

        psi4.core.print_out(
            "\n...finished initializing SAPT object in %5.2f seconds.\n"
            % (time.time() - tinit_start)
        )
        psi4.core.print_out("\n")

    # Gets transformed DF intergrals in MO
    def df_ints(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError("S: string %s does not have 2 elements." % string)

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
            raise ValueError("S: string %s does not have 2 elements." % string)

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
            raise ValueError("Epsilon: string %s does not have 1 element." % string)

        shape = (-1,) + tuple([1] * (dim - 1))

        if (string == "i") or (string == "a") or (string == "r"):
            return self.eps_A[self.slices[string]].reshape(shape)
        elif (string == "j") or (string == "b") or (string == "s"):
            return self.eps_B[self.slices[string]].reshape(shape)
        else:
            psi4.core.clean()
            raise ValueError("Unknown orbital type in eps: %s." % string)

    # Grab MO potential matrices
    def potential(self, string, side):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError("Potential: string %s does not have 2 elements." % string)

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
                "helper_SAPT.potential side must be either A or B, not %s." % side
            )

    def chf(self, monomer, ind=False, **kwargs):
        if monomer not in ["A", "B"]:
            psi4.core.clean()
            raise ValueError(f"'{monomer}' is not a valid monomer for CHF.")

        if self.reference == "ROHF":
            psi4.core.clean()
            raise ValueError("CPHF for a ROHF reference not implemented yet.")

        if monomer == "A":
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

            # Set indicies
            oo = "bb"
            ov = "bs"
            vv = "ss"

        elif monomer == "B":
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

            # Set indicies
            oo = "aa"
            ov = "ar"
            vv = "rr"

        # Construct DF integrals
        Qov = self.df_ints(ov)
        Qoo = self.df_ints(oo)
        Qvv = self.df_ints(vv)

        # Build H^(1) matrix
        H1 = (
            +4 * oe.contract("Qov,QOV->ovOV", Qov, Qov)
            - oe.contract("QoO,QvV->ovOV", Qoo, Qvv)
            - oe.contract("QOv,QoV->ovOV", Qov, Qov)
            - oe.contract(
                "ov,oO,vV->ovOV", eps_ov, np.diag(np.ones(no)), np.diag(np.ones(nv))
            )
        )

        # Solve liear matrix equation
        # H1 * t = omega,
        # where t and omega are vectors of size (ov)
        # and H1 is an (ov)x(ov) matrix.
        t = np.linalg.solve(H1.reshape(no * nv, no * nv), -pert_ov.ravel())

        # We want to return a (vo) matrix
        t = t.reshape(no, nv).T

        if ind:
            e20_ind_r = 2 * np.einsum("vo,ov", t, omega_ov)
            return t, e20_ind_r
        else:
            return t

    def compute_sapt_JK(self, Cleft, Cright, tensor=None):
        if self.alg != "AO":
            raise ValueError("Attempted a call to JK builder in an MO algorithm")

        if self.reference == "ROHF":
            raise ValueError("AO algorithm not yet implemented for ROHF reference.")

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
        num_compute = 0

        for num in range(len(Cleft)):
            Cl = Cleft[num]
            Cr = Cright[num]

            if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
                zero_append.append(num)
                continue

            if tensor is not None:
                mol = Cl.shape[1]
                mor = Cr.shape[1]

                if (tensor[num].shape[0] != mol) or (tensor[num].shape[1] != mor):
                    raise ValueError(
                        "compute_sapt_JK: Tensor size does not match Cl (%d) /Cr (%d) : %s"
                        % (mol, mor, str(tensor[num].shape))
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

        z = np.zeros((self.nmo, self.nmo))
        for num in zero_append:
            J_list.insert(num, z)
            K_list.insert(num, z)

        if return_single:
            return J_list[0], K_list[0]
        else:
            return J_list, K_list

    def chain_dot(self, *dot_list):
        result = dot_list[0]
        for x in range(len(dot_list) - 1):
            result = np.dot(result, dot_list[x + 1])
        return result

    def transform_ao_to_mo(self, x, string):
        if len(string) != 2:
            psi4.core.clean()
            raise ValueError(f"S: string {string} does not have 2 elements.")

        return self.orbitals[string[0]].T.dot(x).dot(self.orbitals[string[1]])


# End SAPT helper


class sapt_timer(object):
    def __init__(self, name):
        self.name = name
        self.start = time.time()
        psi4.core.print_out("\nStarting %s..." % name)

    def stop(self):
        t = time.time() - self.start
        psi4.core.print_out("...%s took a total of % .2f seconds." % (self.name, t))


def sapt_printer(line, value):
    spacer = " " * (20 - len(line))
    print(
        line + spacer + "% 16.8f mH  % 16.8f kcal/mol" % (value * 1000, value * 627.509)
    )


# End SAPT helper
