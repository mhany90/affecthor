#!/bin/bash
# this script applies embedding filters using an existing embedding

#SBATCH --time=00:30:00
#SBATCH --mem=50GB

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : file containing the word embeddings (compressed)
#  $4 : number of words to concatenate


java -Xmx50G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector \
     -embeddingHandler "affective.core.CSVEmbeddingHandler -K ${3} -sep \"\\t\" -I last" -S 0 -K ${4} -I 1 -U -i "${1}" -o "${2}"

