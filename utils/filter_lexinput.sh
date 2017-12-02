#!/bin/bash
# this script applies a lexical filter using lexicons in a given directory

#SBATCH --time=00:15:00
#SBATCH --mem=40GB

# Arguments:
#  $1 : input file
#  $2 : output file
#  $3 : directory containing lexicon evaluators
#  $4 : language code (En, Ar, Es)


# join lexical filter input parameters
lexparams="-I 1 -U"
for f in ${3}/*; do
	lexparams="${lexparams} -lexicon_evaluator \"affective.core.ArffLexiconEvaluator -lexiconFile $f -A 1 -B ${4}\""
done
lexfilter="weka.filters.unsupervised.attribute.TweetToInputLexiconFeatureVector ${lexparams}"


# build and run weka filtering command
run_filter="java -Xmx40G -cp weka.jar weka.Run ${lexfilter} -i ${1} -o ${2}"
eval $run_filter

