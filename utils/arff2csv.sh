#!/bin/bash
# this script converts an ARFF file to CSV

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : field separator, e.g. ',' or '\t'

java -Xmx35G -cp weka.jar \
     weka.Run weka.core.converters.CSVSaver -F $3 -i $1 -o $2

