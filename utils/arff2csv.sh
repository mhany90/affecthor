#!/bin/bash
# this script converts a CSV file to ARFF

# -i : input file
# -o : output file
# -F : field separator, e.g. ',' or '\t'

java -Xmx35G -cp weka.jar \
     weka.Run weka.core.converters.CSVSaver -F $3 -i $1 -o $2

