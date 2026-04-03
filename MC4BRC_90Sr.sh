#!/bin/bash

# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH --nodelist=ubuntu22
#SBATCH --partition=htc_cpu
#SBATCH --cpus-per-task=10
#SBATCH --job-name='MC4BRC 90Sr'
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=dumenil@lpccaen.in2p3.fr
#SBATCH --mail-type=BEGIN,FAIL,END

cd src/

python3 MC4BRC.py 90 38 0 1 1 5000000 ../data/90Y_BS_output.txt
