#!/bin/bash
# this script downloads the data to use in our models


# root directory where the datasets are located
URL_PREFIX="saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA"

# name of each subtask (EI: Emotion Intensity, V: Valence)
TASKS=("EI-reg" "EI-oc" "V-reg" "V-oc")

# type of classification for each subtask (reg: regression, oc: ordinal)
TYPES=("reg" "oc" "reg" "oc")

# possible languages
LANGS=("English" "Arabic" "Spanish")

# abbreviation for each language
CODES=("En" "Ar" "Es")

# desired portions of the data (allowed: train, dev, test)
SETS=("train" "dev")

# affects considered in subtasks of type EI
AFFECTS=("anger" "fear" "joy" "sadness")


for (( t=0; t<4; t++ )); do
    task=${TASKS[$t]}
    type=${TYPES[$t]}

    # rewrite the corresponding directory for each task
    TDIR="$DATADIR/$task"
    rm -rf $TDIR
    mkdir -p $TDIR

    for (( l=0; l<2; l++ )); do
        code=${CODES[$l]}
        lang=${LANGS[$l]}

        # rewrite the corresponding directory for each task, language
        CDIR="$TDIR/$code"
        rm -rf $CDIR
        mkdir -p $CDIR

        pushd $CDIR > /dev/null
        echo "[INFO] Downloading $task $lang data..."


        # determine the (conventional) URL and file name for the task, language
        LOCAL_URL="${URL_PREFIX}/$task/$lang"
        LOCAL_FILE="2018"
        if [ ${t} -lt 2 ] ; then
            LOCAL_FILE="${LOCAL_FILE}-EI-$type-$code"
        else
            LOCAL_FILE="${LOCAL_FILE}-Valence-$type-$code"
        fi
        TRAIN_URL="${LOCAL_URL}/${LOCAL_FILE}-train.zip"
        DEV_URL="${LOCAL_URL}/${LOCAL_FILE}-dev.zip"


        # handle particularities of each task separately
        # directory structure / naming in the materials provided is not consistent

        if [ $t -eq 0 ]; then
            # irregular name in (EI-reg, En) - train
            if [ $l -eq 0 ]; then
                TRAIN_URL="${URL_PREFIX}/$task/$lang/$task-$lang-Train.zip"
            fi

            # download and extract
            wget "${TRAIN_URL}"
            wget "${DEV_URL}"
            unzip "*.zip"

            # clean irrelevant files
            rm -rf "__MACOSX"
            find . -type f -name '*.zip' -delete

            # rename files using our convention
            if [ $l -eq 0 ]; then
                FILES="$CDIR/$task-$lang-Train/*"
                for f in $FILES; do
                    mv -- "$f" "${f//en_/En-}"
                done
                for f in $FILES; do
                    mv -- "$f" "${f//_/-}"
                done
                mv "$CDIR/$task-$lang-Train" "train"
            fi

            for s in ${SETS[@]}; do
                mkdir -p $s
                for a in ${AFFECTS[@]}; do
                    if [ -e "${LOCAL_FILE}-$a-$s.txt" ]; then
                        mv -f "${LOCAL_FILE}-$a-$s.txt" "$s/$task-$code-$a-$s.txt"
                    fi
                done
            done
        fi


        if [ $t -eq 1 ]; then
            : # not yet available
        fi


        if [ $t -eq 2 ]; then
            # irregular name in (V-reg, En) - train, dev
            if [ $l -eq 0 ]; then
                TRAIN_URL="${LOCAL_URL}/${LOCAL_FILE}-training.txt.zip"
                DEV_URL="${LOCAL_URL}/${LOCAL_FILE}-dev.txt.zip"
            fi

            # download and extract
            wget "${TRAIN_URL}"
            wget "${DEV_URL}"
            unzip "*.zip"

            # clean irrelevant files
            rm -rf "__MACOSX"
            rm -rf "readme.txt"
            find . -type f -name '*.zip' -delete

            # rename files using our convention
            if [ $l -eq 0 ]; then
                mv "${LOCAL_FILE}-training.txt" "${LOCAL_FILE}-train.txt"
            fi

            for s in ${SETS[@]}; do
                mkdir -p $s
                if [ -e "${LOCAL_FILE}-$s.txt" ]; then
                    mv -f "${LOCAL_FILE}-$s.txt" "$s/$task-$code-valence-$s.txt"
                fi
            done
        fi


        if [ $t -eq 3 ]; then
            : # not yet available
        fi

        popd > /dev/null

    done
done

