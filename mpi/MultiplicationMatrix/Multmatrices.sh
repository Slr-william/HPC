#!/bin/bash
#
#SBATCH --job-name=Multmatrices
#SBATCH --output=res_Multmatrices.out
#SBATCH --nodes=3
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun Multmatrices

