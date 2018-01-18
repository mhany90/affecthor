#!/bin/bash
#SBATCH --time=119:15:00
#SBATCH --mem=20GB
python3 hybrid_nn_gbt_es_sadness.py > hybrid_nn_gbt_es_sadness_res.txt
