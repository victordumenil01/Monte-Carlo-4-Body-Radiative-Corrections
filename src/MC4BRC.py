#################################################
# MC4BRC.py file (main)
#################################################

from MC_Sampling import * # type: ignore
from NDB import * # type: ignore
from Constants import * # type: ignore
from SpectralFunction import * # type: ignore
import time

if __name__ == "__main__":
    A = int(input("Enter mass number A: "))
    Z = int(input("Enter atomic number Z: "))
    N = A - Z

    result = parse_mass_file_from_Z_N(Z, N)
    if result:
        print("Initial nucleus :", result['initial']['nucleus'])
        if result['final']:
            print("Final nucleus:", result['final']['nucleus_final'])
        else:
            print("Final nucleus not found or not applicable")
    else:
        print("Nucleus not found")

    excited_energy = input("Enter excited energy (in keV): ")
    mi = float(result['initial']['mass_initial (u)']) * UMASSC2
    mf = float(result['final']['mass_final (u)']) * UMASSC2 + float(excited_energy)
    betatype = result['initial']['decay_type']
    if betatype == "beta+":
        Z = - Z
    print('mi :', mi)
    Delta = end_point(mi, mf, betatype)*me
    print('Delta = ', Delta, 'keV')

    MF = float(input("Enter Fermi matrix element MF: "))
    MGT = int(input("Enter Gamow-Teller matrix element MGT: "))
    nTotal = int(input("Enter number of events nTotal: "))

    start = time.perf_counter()
    sampleEvents(A, Z, Delta, mi, MF, MGT, nTotal)
    end = time.perf_counter()

    print(f"Execution time : {end - start} secondes")