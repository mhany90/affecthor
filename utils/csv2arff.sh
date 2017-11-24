#!/bin/bash
# this script converts a CSV file to ARFF

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : field separator, e.g. ',' or '\t'

java -Xmx35G -cp weka.jar \
     weka.Run weka.core.converters.CSVLoader $1 -B 200000 -S "first" -F $3 > $2

