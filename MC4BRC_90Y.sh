#!/bin/bash

# fichier submission.SBATCH
#SBATCH --nodes=1
#SBATCH --nodelist=ubuntu22
#SBATCH --partition=htc_cpu
#SBATCH --cpus-per-task=6
#SBATCH --job-name='MC4BRC 6He'
#SBATCH --output=%x.%J.out
#SBATCH --error=%x.%J.out
#SBATCH --mail-user=dumenil@lpccaen.in2p3.fr
#SBATCH --mail-type=BEGIN,FAIL,END

cd src/

python3 MC4BRC.py 90 39 0 1 1 100000000 ../data/90Zr_BS_output.txt
