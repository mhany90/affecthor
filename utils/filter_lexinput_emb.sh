#!/bin/bash
# this script applies a filter using lexicons in a given directory and an existing embedding

#SBATCH --time=00:30:00
#SBATCH --mem=50GB

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : directory containing lexicon evaluators
#  $4 : language code (En, Ar, Es)
#  $5 : file containing the word embeddings (compressed)
#  $6 : number of words to concatenate


# join lexical filter input parameters
lexparams="-I 1 -U"
for f in ${3}/*; do
	lexparams="${lexparams} -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile $f -A 1 -B ${4}\""
done
lexfilter="weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector ${lexparams}"

# join embedding filter input parameters
embparams="-embeddingHandler \"affective.core.CSVEmbeddingHandler -K ${5} -sep \\\"\\\\t\\\" -I last\" -S 0 -K ${6} -I 1 -U"
embfilter="weka.filters.unsupervised.attribute.TweetToEmbeddingsFeatureVector ${embparams}"


# build and run weka filtering command
run_filter="java -Xmx50G -cp weka.jar weka.Run weka.filters.MultiFilter -F '${lexfilter}' -F '${embfilter}' -i ${1} -o ${2}"
eval $run_filter

