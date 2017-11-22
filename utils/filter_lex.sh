#!/bin/bash
# this script applies a lexical filter using the predefined English lexicons

#SBATCH --time=00:15:00
#SBATCH --mem=40GB


# -i : input file
# -o : output file


java -Xmx40G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector \
     -I 1 -A -D -F -H -J -L -N -P -Q -R -T -U -i "${1}" -o "${2}"

