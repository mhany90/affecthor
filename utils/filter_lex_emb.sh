#!/bin/bash
# this script applies a filter using both the predefined English lexicons and an existing embedding

#SBATCH --time=00:15:00
#SBATCH --mem=40GB


# -i : input file
# -o : output file
# -B : file containing the word embeddings
# -K : number of words to concatenate


java -Xmx40G -cp weka.jar \
     weka.Run weka.filters.MultiFilter \
     -F "weka.filters.unsupervised.attribute.TweetToLexiconFeatureVector -I 1 -A -D -F -H -J -L -N -P -Q -R -T -U" \
     -F "weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector \
     -embeddingHandler \"affective.core.CSVEmbeddingHandler -K ${3} -sep \\\"\\\\t\\\" -I last\" -S 0 -K ${4} -I 1 -U" \
     -i "${1}" -o "${2}"

