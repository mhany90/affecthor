#!/bin/bash
# this script converts a CSV file to ARFF

# -i : input file
# -o : output file
# -F : field separator, e.g. ',' or '\t'

java -Xmx35G -cp weka.jar \
     weka.Run weka.core.converters.CSVLoader $1 -B 100000 -S "first" -F $3 > $2

