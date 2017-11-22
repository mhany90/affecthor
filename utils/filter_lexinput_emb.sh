#!/bin/bash
# this script applies a filter using lexicons in a given directory and an existing embedding

#SBATCH --time=00:15:00
#SBATCH --mem=40GB


# -i : input file
# -o : output file
# -lexicon_evaluator: lexicon evaluator to use (can be used multiple times)
# -B : file containing the word embeddings
# -K : number of words to concatenate


lexeval=""
for f in ${3}/*; do
	lex_file="-lexicon_evaluator $f"
	lexeval="${lexeval} ${lex_file}"
done

java -Xmx40G -cp weka.jar \
     weka.Run weka.filters.MultiFilter \
     -F "weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -I 1 -U ${lexeval}" \
     -F weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector \
     -embeddingHandler "affective.core.CSVEmbeddingHandler -K ${4} -sep \"\\t\" -I last" -S 0 -K ${5} -I 1 -U \
     -i "${1}" -o "${2}"

