#################################################
# NDB.py file
#################################################
from io import StringIO
import re
import pandas as pd # type: ignore
import requests # type: ignore

def parse_mass_file_from_Z_N(target_Z, target_N):
    """
    Provides the initial and final nucleus for given Z et N
    
    :param Z: Atomic number of initial nucleus
    :param N: Number of neutron of initial nucleus
    
    """
    filepath = '../Nuclear Data Base/massround.mas20.txt'
    with open(filepath, 'r') as f:
        lines = f.readlines()[34:]  # skip the headers
    
    # We go through it with a clue so we can see the before and after
    for i, line in enumerate(lines):
        if len(line) < 110:
            continue

        try:
            N = int(line[5:10].strip())
            Z = int(line[10:15].strip())
            A = Z + N
            element = line[20:23].strip()
            decay_code = line[73:75].strip()
            mass_str = line[97:110].strip()
            mass_str = mass_str.replace(' ', '').replace('#', '').replace('*', '')

            if Z == target_Z and N == target_N:
                decay = {'B-': 'beta-', 'B+': 'beta+'}.get(decay_code, 'stable or unknown')
                mass_micro_u = float(mass_str) if mass_str and mass_str != '*' else None
                mass_u = mass_micro_u * 1e-6 if mass_micro_u else None

                result_initial = {
                    'Z': Z,
                    'N': N,
                    'A': A,
                    'nucleus': f"{A}{element}",
                    'decay_type': decay,
                    'mass_initial (u)': mass_u
                }

                # Find final nucleus according to decay type
                if decay == 'beta+':
                    idx_final = i - 1
                elif decay == 'beta-':
                    idx_final = i + 1
                else:
                    idx_final = None

                if idx_final is not None and 0 <= idx_final < len(lines):
                    line_final = lines[idx_final]
                    if len(line_final) >= 110:
                        try:
                            Nf = int(line_final[5:10].strip())
                            Zf = int(line_final[10:15].strip())
                            Af = Zf + Nf
                            elementf = line_final[20:23].strip()
                            decay_code_final = line_final[73:75].strip() or 'stable'
                            mass_str_final = line_final[97:110].strip()
                            mass_str_final = mass_str_final.replace(' ', '').replace('#', '').replace('*', '')
                            mass_micro_u_final = float(mass_str_final) if mass_str_final and mass_str_final != '*' else None
                            mass_u_final = mass_micro_u_final * 1e-6 if mass_micro_u_final else None

                            result_final = {
                                'Z_final': Zf,
                                'N_final': Nf,
                                'A_final': Af,
                                'nucleus_final': f"{Af}{elementf}",
                                'decay_type_final': decay_code_final,
                                'mass_final (u)': mass_u_final
                            }
                        except Exception:
                            result_final = None
                    else:
                        result_final = None
                else:
                    result_final = None

                # Turn the 2 nuclei over
                return {
                    'initial': result_initial,
                    'final': result_final
                }

        except Exception:
            continue

    return None


def downloadNubase():
    """
    Download the Nuclear Data Base and provides
    Z: Atomic number
    A: Mass number
    halflife: Half life
    halflifeunc: Uncertainties on half life
    spinParity: Spin parity of nucleus
    branchingB-: Branching ratio of beta- decay
    branchingB+: Branching ratio of beta+ decay
    
    """
    names = ['Z', 'A', 'halflife', 'halflifeUnc', 'spinParity', 'branchingB-', 'branchingB+']
    unitDict = {'Qy': 1e30*365.25*24*3600, 'Ry': 1e27*365.25*24*3600, 'Yy': 1e24*365.25*24*3600, 'Zy': 1e21*365.25*24*3600, 'Ey': 1e18*365.25*24*3600,'Py': 1e15*365.25*24*3600, 'Ty': 1e12*365.25*24*3600, 'Gy': 1e9*365.25*24*3600, 'My': 1e6*365.25*24*3600, 'ky': 1e3*365.25*24*3600, 'y': 365.25*24*3600, 'd': 24*3600., 'h': 3600., 'm': 60., 's': 1., 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12, 'fs': 1e-15, 'as': 1e-18, 'zs': 1e-21, 'ys': 1e-24}
    data = []
    url = 'https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt'
    r = requests.get(url).content
    f = StringIO(r.decode('utf-8'))
    skipLines = 25
    currentLine = 0
    for line in f:
        currentLine += 1
        if currentLine <= skipLines:
            continue
        else:
            a = int(line[0:3])
            zi = line[4:8]
            if zi[-1] != '0':
                continue
            z = int(zi[:-1])
            
            halflife = line[69:78]
            halflifeUnit = line[78:80].strip()
            halflifeUnc = line[81:88]

            spinParity = line[88:102].split(' ')[0].replace('#', '').replace('*', '')
            branches = line[119:209]
            branchingRatioBm = 100.
            branchingRatioBp = 100.
            
            try:
                halflife = float(halflife)*unitDict[halflifeUnit]
                halflifeUnc = float(halflifeUnc)*unitDict[halflifeUnit]
            except:
                halflife = -1.
                halflifeUnc = 0.
            try:
                m = re.search(r'B-=\d+[\.\d+]*', branches)
                if m:
                    branchingRatioBm = float(m.group(0).split('B-=')[1])
                m = re.search(r'B+=\d+[\.\d+]*', branches)
                if m:
                    branchingRatioBp = float(m.group(0).split('B+=')[1])
            except:
                pass

            data.append([z, a, halflife, halflifeUnc, spinParity, branchingRatioBm, branchingRatioBp])
    return pd.DataFrame(data, columns=names)

def find_halflife(A, Z):
    """
    Find the experimental half life in
    https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt
    
    :param A: Mass number
    :param Z: Atomic number of parent nucleus
    
    """
    df = downloadNubase()
    
    match = df[(df['Z'] == Z) & (df['A'] == A)]

    if not match.empty:
        row = match.iloc[0]
        return row['halflife'], row['halflifeUnc']
    else:
        return None, None
    
