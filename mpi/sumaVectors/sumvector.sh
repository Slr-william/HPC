#!/bin/bash
#
#SBATCH --job-name=sumvector
#SBATCH --output=res_sumvector.out
#SBATCH --nodes=3
#SBATCH --ntasks=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100

mpirun sumvector
