#!/bin/bash
# this script applies word embeddings and weka filters


# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data (allowed: train, dev, test)
SETS=("train" "dev" "test")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")

# WEKA root directory
WEKADIR=${TOOLDIR}/weka-3-9-1


# iterate over tasks
for t in ${TASKS[@]}; do
    # iterate over each language
    for l in ${CODES[@]}; do
        TAFFECTS=${AFFECTS[@]}
        if [ ${t:0:1} == "V" ]; then
            TAFFECTS=("valence")
        fi
        for a in ${TAFFECTS[@]}; do
            for s in ${SETS[@]}; do
                arff_file="$DATADIR/$t/$l/$s/$t-$l-$a-$s.arff"

                # TODO: Define embeddings
                # TODO: Define filters (lex, emb)

                if [ $l == "En" ]; then
                    echo $arff_file
                fi

                if [ $l == 'Ar' ]; then
                    echo $arff_file
                fi

                if [ $l == 'Es' ]; then
                    echo $arff_file
                fi
            done
        done
    done
done

