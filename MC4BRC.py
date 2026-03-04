#################################################
# MC4BRC.py file (main)
#################################################

from MC_Sampling import * # type: ignore

if __name__ == "__main__":
    A = int(input("Enter mass number A: "))
    Z = int(input("Enter atomic number Z: "))
    Delta = float(input("Enter energy difference Delta (in keV): "))
    MF = float(input("Enter Fermi matrix element MF: "))
    MGT = int(input("Enter Gamow-Teller matrix element MGT: "))
    nTotal = int(input("Enter number of events nTotal: "))
    sampleEvents(A, Z, Delta, MF, MGT, nTotal)