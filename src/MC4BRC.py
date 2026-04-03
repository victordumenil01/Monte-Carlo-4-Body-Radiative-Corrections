#################################################
# MC4BRC.py file (main)
#################################################
import time
import sys

from MC_Sampling import * # type: ignore
from NDB import * # type: ignore
from Constants import * # type: ignore
from SpectralFunction import * # type: ignore

#################################################
def read_config_file(filepath):
    """
    Read the input file
    
    :param filepath: path to input file
    
    """
    config = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): 
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  

                config[key] = value
    return config


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
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
        
        final_nucleus = str(result['final']['nucleus_final'])
        excited_energy = input("Enter excited energy (in keV): ")
        mi = float(result['initial']['mass_initial (u)']) * UMASSC2
        mf = float(result['final']['mass_final (u)']) * UMASSC2 + float(excited_energy)
        betatype = result['initial']['decay_type']
        if betatype == "beta+":
            Z = - Z
        Delta = end_point(mi, mf, betatype)*me
        print('Delta = ', Delta, 'keV')

        MF = float(input("Enter Fermi matrix element MF: "))
        MGT = int(input("Enter Gamow-Teller matrix element MGT: "))
        nTotal = int(input("Enter number of events nTotal: "))
        input_data = input("Path of the input file (leave blank if not applicable): ")

    elif len(sys.argv) == 2:
        config_file = sys.argv[1]
        config = read_config_file(config_file)
        A = config.get('A')
        Z = config.get('Z')
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

        final_nucleus = str(result['final']['nucleus_final'])
        excited_energy = config.get('excited_energy', 0.0)
        mi = float(result['initial']['mass_initial (u)']) * UMASSC2
        mf = float(result['final']['mass_final (u)']) * UMASSC2 + float(excited_energy)
        betatype = result['initial']['decay_type']
        if betatype == "beta+":
            Z = - Z
        Delta = end_point(mi, mf, betatype)*me
        print('Delta = ', Delta, 'keV')

        MF = config.get('MF')
        MGT = config.get('MGT')
        nTotal = config.get('nTotal')
    elif len(sys.argv) == 8:
        A = int(sys.argv[1])
        Z = int(sys.argv[2])
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

        final_nucleus = str(result['final']['nucleus_final'])
        excited_energy = float(sys.argv[3])
        mi = float(result['initial']['mass_initial (u)']) * UMASSC2
        mf = float(result['final']['mass_final (u)']) * UMASSC2 + float(excited_energy)
        betatype = result['initial']['decay_type']
        if betatype == "beta+":
            Z = - Z
        Delta = end_point(mi, mf, betatype)*me
        print('Delta = ', Delta, 'keV')

        MF = float(sys.argv[4])
        MGT = float(sys.argv[5])
        nTotal = int(sys.argv[6])
        input_data = str(sys.argv[7])

    start = time.perf_counter()
    sampleEvents(A, Z, Delta, mi, MF, MGT, nTotal, input_data, final_nucleus)
    end = time.perf_counter()

    print(f"Execution time : {end - start} secondes")
