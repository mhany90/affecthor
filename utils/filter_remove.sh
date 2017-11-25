#!/bin/bash
# this script converts a CSV file to ARFF

#SBATCH --time=00:10:00
#SBATCH --mem=30GB


java -Xmx30G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.Remove -R $3 -i $1 -o $2

