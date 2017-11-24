#!/bin/bash
# this script applies a lexical filter using lexicons in a given directory

#SBATCH --time=00:15:00
#SBATCH --mem=40GB

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : directory containing lexicon evaluators


lexeval=""
for f in ${3}/*; do
	lex_file="-lexicon_evaluator $f"
	lexeval="${lexeval} ${lex_file}"
done

java -Xmx40G -cp weka.jar \
     weka.Run weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector \
     -I 1 -U "${lexeval}" -i "${1}" -o "${2}"

