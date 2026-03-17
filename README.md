# Monte Carlo 4-Body Radiative Corrections
[Glück 1997](https://www.sciencedirect.com/science/article/pii/S0010465596001683)

## Installation

Clone the repository :

git clone https://github.com/victordumenil01/Monte-Carlo-4-Body-Radiative-Corrections-.git

## Usage
### Option 1 (Unitary test)
```python
python3 MC4BRC.py 

Enter mass number A: 6
Enter atomic number Z: 2
Initial nucleus : 6He
Final nucleus: 6Li
Enter excited energy (in keV): 0
Delta =  4015.140655197147 keV
Enter Fermi matrix element MF: 0
Enter Gamow-Teller matrix element MGT: 6
Enter number of events nTotal: 200000
```

### Option 2
```python
python3 MC4BRC.py A Z Ex MF MGT N
python3 MC4BRC.py 6 2 0 0 6 10000
```

### Option 3 (with input file)
```python
python3 MC4BRC.py input.txt
```

## Output example
```python
[TreeLevel] Efficiency: 0.4417
PH          : 0.029667
r_rho       : 1.1097 %
nS : 194025   nH : 5975   nS/nTotal : 0.9701
[Hard] Efficiency H: 24.4904 %
[Soft] Efficiency: 0.4407
Done
Execution time : 31.168557166005485 secondes
```

## Plot
```python
python3 plot_histograms.py [r_rho]
python3 plot_histograms.py 1.1097
```

## Dependancies

- numpy
- matplotlib
- scipy
- matplotlib.pyplot
- concurrent.futures
