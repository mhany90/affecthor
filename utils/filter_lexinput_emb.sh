#!/bin/bash
# this script applies a filter using lexicons in a given directory and an existing embedding

#SBATCH --time=00:30:00
#SBATCH --mem=50GB

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : directory containing lexicon evaluators
#  $4 : file containing the word embeddings (compressed)
#  $5 : number of words to concatenate


lexeval=""
for f in ${3}/*; do
	lex_file="-lexicon_evaluator $f"
	lexeval="${lexeval} ${lex_file}"
done

java -Xmx50G -cp weka.jar \
     we5a.Run weka.filters.MultiFilter \
     -F "weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector -I 1 -U ${lexeval}" \
     -F "weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector \
     -embeddingHandler \"affective.core.CSVEmbeddingHandler -K ${4} -sep \\\"\\\\t\\\" -I last\" -S 0 -K ${5} -I 1 -U" \
     -i "${1}" -o "${2}"

