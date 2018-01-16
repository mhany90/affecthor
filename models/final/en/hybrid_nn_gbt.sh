#!/bin/bash
#SBATCH --time=119:15:00
#SBATCH --mem=5GB
python3 hybrid_nn_gbt.py > hybrid_nn_gbt_res.txt
