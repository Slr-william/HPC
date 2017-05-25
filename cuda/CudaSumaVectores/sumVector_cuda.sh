#!/bin/bash
#
#SBATCH --job-name=sumVector_cuda
#SBATCH --output=res_sumVector_cuda.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=100
#SBATCH --gres=gpu:1

mpirun sumVector_cuda
