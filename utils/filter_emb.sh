#!/bin/bash
# this script applies embedding filters using an existing embedding

#SBATCH --time=00:15:00
#SBATCH --mem=40GB


# -i : input file
# -o : output file
# -B : file containing the word embeddings
# -K : number of words to concatenate


java -Xmx40G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector \
     -embeddingHandler "affective.core.CSVEmbeddingHandler -K ${3} -sep \"\\t\" -I last" -S 0 -K ${4} -I 1 -U -i "${1}" -o "${2}"

