import numpy as np
import psi4
from interaction_induced.cubes import make_cube
from interaction_induced.utils import prepare_path

# specify geometry in Psi4 format
GEO = """
symmetry c1
no_com
no_reorient
0 1
C    0.00700    0.09936   -0.01951
H   -0.04431    0.02900    1.07782
H    1.05653    0.08055   -0.35098
Cl  -0.78354   -1.40640   -0.68296
F   -0.63568    1.21747   -0.47436
"""

# specify memory and threads
MEMORY = "5 GB"
THREADS = 4

# specify basis sets
BASIS = "6-31g"
DF_BASIS = "cc-pvtz"

# specify options
OPTIONS = {
    # "option": "value",
    "basis": BASIS,
    # "DF_BASIS_SCF": DF_BASIS + "-jkfit",
    "scf_type": "direct",
}


# specify output and resultS filenames
OUTPUT_FILE_PATH = "output.dat"

if __name__ == "__main__":

    ### Psi4 options
    psi4.set_memory(MEMORY)
    psi4.set_num_threads(THREADS)
    psi4.core.set_output_file(OUTPUT_FILE_PATH, False)
    psi4.set_options(OPTIONS)

    np.set_printoptions(precision=4, suppress=True)

    ### Perform SCF
    molecule = psi4.geometry(GEO)
    # energy, wfn = psi4.optimize("scf", return_wfn=True, engine="geometric")
    energy, wfn = psi4.energy("scf", molecule=molecule, return_wfn=True)

    ### Grab occupied orbitals
    ndocc = wfn.doccpi()[0]
    C = wfn.Ca().to_array()
    Co = C[:, :ndocc]

    ### Grab density matrix
    D = wfn.Da().to_array()

    mints = psi4.core.MintsHelper(wfn.basisset())

    S_half = mints.ao_overlap()
    S_half.power(0.5)
    S_half = S_half.to_array()

    S_neg_half = mints.ao_overlap()
    S_neg_half.power(-0.5)
    S_neg_half = S_neg_half.to_array()

    eps, U = np.linalg.eigh(S_half.dot(D).dot(S_half))

    print("\n", eps, "\n")

    print(S_half.dot(Co))
    print(U[:, U.shape[1] - ndocc :])
    print("\n")

    diff = S_half.dot(Co) - U[:, U.shape[1] - ndocc :]
    summ = S_half.dot(Co) + U[:, U.shape[1] - ndocc :]
    print("diff: ", np.array([np.sqrt(np.sum(col**2)) for col in diff.T]))
    print("sum:  ", np.array([np.sqrt(np.sum(col**2)) for col in summ.T]))
    print("\n")

    dens0 = Co.dot(Co.T)
    dens = C.dot(np.diag(eps[::-1])).dot(C.T)
    dens1 = S_neg_half.dot(U).dot(np.diag(eps)).dot(U.T).dot(S_neg_half)
    print(f"{np.sqrt(np.sum((D - dens0).flatten() ** 2)):.4f}")
    print(f"{np.sqrt(np.sum((D - dens).flatten() ** 2)):.4f}")
    print(f"{np.sqrt(np.sum((D - dens1).flatten() ** 2)):.4f}")

    ### End calculations
    psi4.core.clean()
