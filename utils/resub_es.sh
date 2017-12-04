#!/bin/bash
#SBATCH --time=68:15:00
#SBATCH --mem=2GB
while true; do
source activate SEMCPU
echo starting
srun python3 es_scrape.py
wait
echo stopped
done
