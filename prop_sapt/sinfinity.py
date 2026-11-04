"""
Second-quantized SAPT S^infinity
"""

from time import time
import psi4
import numpy as np
import opt_einsum as oe
from .helper_SAPT_DF import helper_SAPT


def get_A_bb(s_ba, s_ab):
    gamma_bb = oe.contract("Ba,ab->Bb", s_ba, s_ab)

    lam, U = np.linalg.eigh(gamma_bb)

    gamma_bb_prime = np.diag(lam)
    A_bb_prime = oe.contract("Bb,Bb->Bb", gamma_bb_prime, 1 / (1 - gamma_bb_prime))
    A_bb = U.dot(A_bb_prime.dot(U.T))
    A_bb = np.identity(A_bb.shape[0]) + A_bb

    return A_bb


def get_B_aa(s_ba, s_ab):
    delta_aa = oe.contract("ba,Ab->Aa", s_ba, s_ab)

    lam, U = np.linalg.eigh(delta_aa)

    delta_aa_prime = np.diag(lam)
    B_aa_prime = oe.contract("Aa,Aa->Aa", delta_aa_prime, 1 / (1 - delta_aa_prime))
    B_aa = U.dot(B_aa_prime.dot(U.T))
    B_aa = np.identity(B_aa.shape[0]) + B_aa

    return B_aa


class sinfinity(helper_SAPT):

    def __init__(self, dimer, reference="RHF", memory=None, **kwargs):
        super().__init__(dimer, reference, memory, **kwargs)

        ### Starting initialisation
        psi4.core.print_out("\nInitializing Sinfinity object...\n")
        t_start = time()

        ###  Matrices and tensors from helper_sapt
        s_ab = self.s("ab")
        s_ba = self.s("ba")
        s_sa = self.s("sa")
        s_as = self.s("as")
        s_rb = self.s("rb")
        s_sr = self.s("sr")
        s_rs = self.s("rs")
        s_br = self.s("br")

        ### Calculating A and B matrices
        self.A_bb = get_A_bb(s_ba, s_ab)
        self.B_aa = get_B_aa(s_ba, s_ab)

        self.H_ab = oe.contract("Bb,aB->ab", self.A_bb, s_ab)
        self.H_ba = oe.contract("Ba,bB->ba", s_ba, self.A_bb)
        self.I_rb = oe.contract("Bb,rB->rb", self.A_bb, s_rb)
        self.I_br = oe.contract("Br,bB->br", s_br, self.A_bb)
        self.J_sa = oe.contract("Aa,sA->sa", self.B_aa, s_sa)
        self.J_as = oe.contract("As,aA->as", s_as, self.B_aa)

        self.G_sr = s_sr + oe.contract("br,Bb,aB,sa->sr", s_br, self.A_bb, s_ab, s_sa)
        self.G_rs = s_rs + oe.contract("as,Aa,bA,rb->rs", s_as, self.B_aa, s_ba, s_rb)

        self.C_rr = np.identity(self.sizes["r"]) - oe.contract(
            "bR,Bb,rB->rR", s_br, self.A_bb, s_rb
        )
        self.D_ss = np.identity(self.sizes["s"]) - oe.contract(
            "aS,Aa,sA->sS", s_as, self.B_aa, s_sa
        )

        self.E_ra = oe.contract("ba,Bb,rB->ra", s_ba, self.A_bb, s_rb)
        self.E_ar = oe.contract("aB,Bb,br->ar", s_ab, self.A_bb, s_br)
        self.F_sb = oe.contract("ab,Aa,sA->sb", s_ab, self.B_aa, s_sa)
        self.F_bs = oe.contract("bA,Aa,as->bs", s_ba, self.B_aa, s_as)

        # dispersion denominator
        self.eps_rsab = 1 / (
            -self.eps("r", dim=4)
            - self.eps("s", dim=3)
            + self.eps("a", dim=2)
            + self.eps("b")
        )

        # MO transformed df ERIs
        self.Qaa = self.df_ints("aa")
        self.Qar = self.df_ints("ar")
        self.Qrr = self.df_ints("rr")
        self.Qbb = self.df_ints("bb")
        self.Qbs = self.df_ints("bs")
        self.Qss = self.df_ints("ss")

        # transpositions
        self.Qra = self.Qar.transpose(0, 2, 1)
        self.Qsb = self.Qbs.transpose(0, 2, 1)

        ### omegaA and omegaB
        omegaA = self.V_A + 2 * self.J_A
        omegaB = self.V_B + 2 * self.J_B

        self.omegaA_bs = self.orbitals["b"].T @ omegaA @ self.orbitals["s"]
        self.omegaB_ar = self.orbitals["a"].T @ omegaB @ self.orbitals["r"]

        self.omegaA_bb = self.orbitals["b"].T @ omegaA @ self.orbitals["b"]
        self.omegaB_aa = self.orbitals["a"].T @ omegaB @ self.orbitals["a"]

        self.omegaA_ss = self.orbitals["s"].T @ omegaA @ self.orbitals["s"]
        self.omegaB_rr = self.orbitals["r"].T @ omegaB @ self.orbitals["r"]

        ### omega_exchA and omega_exchB
        # Every two-electron piece of the exchange potentials is a J or K build
        # on a generalized density D = C_left M C_right^T, done with the same JK
        # object as the SAPT energies, so that the first-order exchange properties
        # are the derivative of the EXCH1 propSAPT reports rather than of its
        # DF_BASIS_SAPT approximation. All eight densities go through one JK pass;
        # the first four are needed for J only, the last four for K only.
        C_a, C_r = self.orbitals["a"], self.orbitals["r"]
        C_b, C_s = self.orbitals["b"], self.orbitals["s"]
        jk_left, jk_right, jk_tensor = zip(
            (C_a, C_r, self.E_ra.T),  # J: sum_ar (..|ar) E_ra
            (C_b, C_s, self.F_sb.T),  # J: sum_bs (..|bs) F_sb
            (C_a, C_r, s_ab @ s_br),  # J: sum_ar (..|ar) (s_ab s_br)_ar
            (C_b, C_s, s_ba @ s_as),  # J: sum_bs (..|bs) (s_ba s_as)_bs
            (C_a, C_s, s_as),  # K: sum_as (.a|.s) s_as
            (C_r, C_b, s_rb),  # K: sum_rb (.r|.b) s_rb
            (C_r, C_b, self.I_rb),  # K: sum_rb (.r|.b) I_rb
            (C_a, C_s, self.J_sa.T),  # K: sum_as (.a|.s) J_sa
        )
        J_list, K_list = self.compute_jk(list(jk_left), list(jk_right), list(jk_tensor))
        J_E, J_F, J_Sar, J_Sbs = J_list[:4]
        K_Sas, K_Srb, K_I, K_J = K_list[4:]
        mo = self.transform_ao_to_mo

        self.omega_exchA_bs_S2 = (
            oe.contract("ab,Ba,bs->Bs", s_ab, s_ba, self.omegaA_bs)
            - oe.contract("aS,sa,bs->bS", s_as, s_sa, self.omegaA_bs)
            - oe.contract("rs,ba,ar->bs", s_rs, s_ba, self.omegaB_ar)
            - mo(K_Sas, "rb").T @ s_rs
            + s_ba @ mo(K_Srb, "as")
            - 2 * mo(J_Sar, "bs")
        )

        self.omega_exchA_bs = (
            oe.contract("Bb,sS,bs->BS", self.A_bb, self.D_ss, self.omegaA_bs)
            - oe.contract("rs,ba,ar->bs", self.G_rs, self.H_ba, self.omegaB_ar)
            - 2 * self.A_bb @ mo(J_E, "bs") @ self.D_ss
            - self.A_bb @ mo(K_J, "rb").T @ self.G_rs
            + self.H_ba @ mo(K_I, "as") @ self.D_ss
            + 2 * self.H_ba @ mo(J_F, "ar") @ self.G_rs
        ) - self.omegaA_bs

        self.omega_exchA_sb_S2 = (
            +oe.contract("as,ba,cb->sc", s_as, s_ba, self.omegaA_bb)
            - oe.contract("as,ba,sc->cb", s_as, s_ba, self.omegaA_ss)
            - oe.contract("br,as,ar->sb", s_br, s_as, self.omegaB_ar)
            + s_as.T @ mo(K_Srb, "ab")
            - mo(K_Sas, "rs").T @ s_br.T
            - 2 * mo(J_Sar, "bs").T
        )

        self.omega_exchA_sb = (
            -2 * mo(J_E, "bs").T
            + oe.contract("bB,sb->sB", self.omegaA_bb, self.F_sb)
            - oe.contract("Ss,sb->Sb", self.omegaA_ss, self.F_sb)
            + self.J_sa @ mo(K_I, "ab")
            - mo(K_J, "rs").T @ self.I_rb
            - oe.contract("ar,sa,rb->sb", self.omegaB_ar, self.J_sa, self.I_rb)
            - oe.contract("BS,Sb,sB->sb", self.omegaA_bs, self.F_sb, self.F_sb)
            - self.J_sa @ mo(K_I, "as") @ self.F_sb
            - self.F_sb @ mo(K_J, "rb").T @ self.I_rb
            + 2 * self.J_sa @ mo(J_F, "ar") @ self.I_rb
            + 2 * self.F_sb @ mo(J_E, "bs") @ self.F_sb
            + 2 * (self.F_bs @ mo(J_E, "ss")).T
            - 2 * (mo(J_E, "bb") @ self.F_bs).T
        )

        self.omega_exchB_ar_S2 = (
            oe.contract("ba,Ab,ar->Ar", s_ba, s_ab, self.omegaB_ar)
            - oe.contract("bR,rb,ar->aR", s_br, s_rb, self.omegaB_ar)
            - oe.contract("sr,ab,bs->ar", s_sr, s_ab, self.omegaA_bs)
            + s_ab @ mo(K_Sas, "rb").T
            - mo(K_Srb, "as") @ s_sr
            - 2 * mo(J_Sbs, "ar")
        )

        self.omega_exchB_ar = (
            oe.contract("Aa,rR,ar->AR", self.B_aa, self.C_rr, self.omegaB_ar)
            - oe.contract("sr,ab,bs->ar", self.G_sr, self.H_ab, self.omegaA_bs)
            - 2 * self.B_aa @ mo(J_F, "ar") @ self.C_rr
            - self.B_aa @ mo(K_I, "as") @ self.G_sr
            + self.H_ab @ mo(K_J, "rb").T @ self.C_rr
            + 2 * self.H_ab @ mo(J_E, "bs") @ self.G_sr
        ) - self.omegaB_ar

        self.omega_exchB_ra_S2 = (
            +oe.contract("br,ab,ca->rc", s_br, s_ab, self.omegaB_aa)
            - oe.contract("br,ab,rc->ca", s_br, s_ab, self.omegaB_rr)
            - oe.contract("br,as,bs->ra", s_br, s_as, self.omegaA_bs)
            + s_br.T @ mo(K_Sas, "ab").T
            - mo(K_Srb, "rs") @ s_as.T
            - 2 * mo(J_Sbs, "ar").T
        )

        self.omega_exchB_ra = (
            -2 * mo(J_F, "ar").T
            + oe.contract("aA,ra->rA", self.omegaB_aa, self.E_ra)
            - oe.contract("Rr,ra->Ra", self.omegaB_rr, self.E_ra)
            + self.I_rb @ mo(K_J, "ab").T
            - mo(K_I, "rs") @ self.J_sa
            - oe.contract("bs,rb,sa->ra", self.omegaA_bs, self.I_rb, self.J_sa)
            - oe.contract("AR,Ra,rA->ra", self.omegaB_ar, self.E_ra, self.E_ra)
            - self.I_rb @ mo(K_J, "rb").T @ self.E_ra
            - self.E_ra @ mo(K_I, "as") @ self.J_sa
            + 2 * self.I_rb @ mo(J_E, "bs") @ self.J_sa
            + 2 * self.E_ra @ mo(J_F, "ar") @ self.E_ra
            + 2 * (self.E_ar @ mo(J_F, "rr")).T
            - 2 * (mo(J_F, "aa") @ self.E_ar).T
        )

        ### dispersion amplitudes
        self.t_rsab = np.array(
            oe.contract("Qar,Qbs,rsab->rsab", self.Qar, self.Qbs, self.eps_rsab)
        )

        ### Non-response amplitudes
        self.eps_ra = 1 / (-self.eps("r", dim=2) + self.eps("a"))
        self.tB_ra = self.omegaB_ar.T * self.eps_ra

        self.eps_sb = 1 / (-self.eps("s", dim=2) + self.eps("b"))
        self.tA_sb = self.omegaA_bs.T * self.eps_sb

        ### Print time
        psi4.core.print_out(
            f"...finished initializing Sinfinity object in {(time() - t_start):5.2f} seconds.\n\n"
        )
