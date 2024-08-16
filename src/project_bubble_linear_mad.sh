#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=pi-xinhe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1
#SBATCH --job-name=nb-tunnel
#SBATCH --output=nb-%J.out
#SBATCH --error=nb-%J.err

module unload cuda

module load cuda/12.0

module unload python

module load python/anaconda-2021.05

python train_grid_gcPBM_bubble_linear.py --DataDir ../data/gcPBM_data/ --ModelOut ../trained_model/gcPBM/ --NumTarget 1 --BatchSize 1 --lr 0.0001 --TF mad