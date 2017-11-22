#!/bin/bash
# this script converts a CSV file to ARFF

# -R : field
# -i : input file
# -o : output file


java -Xmx35G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.Remove -R $3 -i $1 -o $2

