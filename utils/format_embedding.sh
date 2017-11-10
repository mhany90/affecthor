#!/bin/bash
# this scripts formats embedding files such that:
#   numerical values are separated by tabs
#   words appear at the end of the corresponding line

# input file
IFILE=$1

# output file
OFILE=$2

# determine whether the current embedding file has a word in the first position
wstart=0
if (awk '{if ($1 ~ /^[\-]*[0-9]+(\.[0-9]+){0,1}[e\-[0-9]+]*$/) print "0"; else print "1"}' $IFILE | grep -q 1); then
    wstart=1
fi

# reorder and rewrite to the output file
awk -v awk_ws=$wstart '{
  if ( awk_ws ) {
    for (i=2; i<=NF; i++) printf $i "\t"; print $1
  }
  else {
    for (i=1; i<NF; i++) printf $i "\t"; print $NF
  }
}' $IFILE > $OFILE

