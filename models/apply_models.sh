#!/bin/bash
# this script applies selected inference models to the extracted features
# WORK IN PROGRESS!


# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data to which apply the models (allowed: train, dev, test, traindev)
SETS=("test")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")

# feature sets to use
FEAT_SETS=("lex" "emb" "combined")

# embeddings to use
EMB_EN_EXT=("ed")
EMB_AR_EXT=("tweets")
EMB_ES_EXT=("tweets")


############################################ EASY SETTINGS FOR TESTING
TASKS=("EI-reg")
CODES=("En")
FEAT_SETS=("emb")
############################################ COMMENT OUT


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
                for f in ${FEAT_SETS[@]}; do
                    feats_base="$DATADIR/$t/$l/$s/features/
                    # example: EI-reg-En-joy-train.combined.400m.csv"

                    if [ $l == "En" ]; then
                        echo "TODO" > /dev/null
                    fi

                    if [ $l == "Ar" ]; then
                        echo "TODO" > /dev/null
                    fi

                    if [ $l == "Es" ]; then
                        echo "TODO" > /dev/null
                    fi

                done
            done
        done
    done
done


