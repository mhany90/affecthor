#!/bin/bash
# this script downloads the data to use in our models


# iterate over tasks
for (( t=0; t<4; t++ )); do
    task=${TASKS[$t]}
    type=${TYPES[$t]}

    # rewrite the corresponding directory for each task
    TDIR="$DATADIR/$task"
    rm -rf $TDIR
    mkdir -p $TDIR

    # iterate over languages
    for (( l=0; l<3; l++ )); do
        code=${CODES[$l]}
        lang=${LANGS[$l]}

        # rewrite the corresponding directory for each (task, language)
        CDIR="$TDIR/$code"
        rm -rf $CDIR
        mkdir -p $CDIR

        pushd $CDIR > /dev/null
        echo "[INFO] Downloading $task $lang data..."

        # determine the conventional URL and file name for the (task, language)
        LOCAL_URL="${URL_PREFIX}/$task/$lang"
        LOCAL_FILE="2018"
        if [ $t -lt 2 ] ; then
            LOCAL_FILE="${LOCAL_FILE}-EI-$type-$code"
        else
            LOCAL_FILE="${LOCAL_FILE}-Valence-$type-$code"
        fi
        TRAIN_URL="${LOCAL_URL}/${LOCAL_FILE}-train.zip"
        DEV_URL="${LOCAL_URL}/${LOCAL_FILE}-dev.zip"


        # handle particularities of each task separately
        # directory structure / naming in the materials provided is not consistent


        # EI-reg task
        if [ $t -eq 0 ]; then
            # irregular URL in (EI-reg, En, train)
            if [ $l -eq 0 ]; then
                TRAIN_URL="${URL_PREFIX}/$task/$lang/$task-$lang-Train.zip"
            fi

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
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

            if [ $l -eq 2 ]; then
                for f in "$CDIR/2018-$task-$code-train/*"; do
                    mv $f $CDIR
                done
                for f in "$CDIR/2018-$task-$code-dev/*"; do
                    mv $f $CDIR
                done
                rm -rf "$CDIR/2018-$task-$code-train"
                rm -rf "$CDIR/2018-$task-$code-dev"
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


        # EI-oc task
        if [ $t -eq 1 ]; then
            # ignore unavailable languages
            if [ $l -ne 0 ]; then
                continue
            fi

            # irregular URL in (EI-oc, En, train)
            if [ $l -eq 0 ]; then
                TRAIN_URL="${URL_PREFIX}/$task/$lang/$task-$code-train.zip"
            fi

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
            unzip "*.zip"

            # clean irrelevant files
            find . -type f -name '*.zip' -delete

            # rename files using our convention
            for s in ${SETS[@]}; do
                mkdir -p $s
                for a in ${AFFECTS[@]}; do
                    if [ -e "${CDIR}/$task-$code-$a-$s.txt" ]; then
                        mv -f "${CDIR}/$task-$code-$a-$s.txt" "$s/$task-$code-$a-$s.txt"
                    fi
                    if [ -e "${LOCAL_FILE}-$a-$s.txt" ]; then
                        mv -f "${LOCAL_FILE}-$a-$s.txt" "$s/$task-$code-$a-$s.txt"
                    fi
                done
            done
        fi


        # V-reg task
        if [ $t -eq 2 ]; then
            # ignore unavailable languages
            if [ $l -eq 2 ]; then
                continue
            fi

            # irregular URL in (V-reg, En, train) and (V-reg, En, dev)
            if [ $l -eq 0 ]; then
                TRAIN_URL="${LOCAL_URL}/${LOCAL_FILE}-training.txt.zip"
                DEV_URL="${LOCAL_URL}/${LOCAL_FILE}-dev.txt.zip"
            fi

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
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

        # V-oc task
        if [ $t -eq 3 ]; then
            # ignore unavailable languages
            if [ $l -ne 0 ]; then
                continue
            fi

            # irregular URL in (V-oc, En, train) and (V-oc, En, dev)
            LOCAL_URL="${URL_PREFIX}/VAD-oc/$lang"
            TRAIN_URL="${LOCAL_URL}/${LOCAL_FILE}-train.zip"
            DEV_URL="${LOCAL_URL}/${LOCAL_FILE}-dev.zip"

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
            unzip "*.zip"

            # clean irrelevant files
            find . -type f -name '*.zip' -delete

            # rename files using our convention
            for s in ${SETS[@]}; do
                mkdir -p $s
                if [ -e "${LOCAL_FILE}-$s.txt" ]; then
                    mv -f "${LOCAL_FILE}-$s.txt" "$s/$task-$code-valence-$s.txt"
                fi
            done
        fi

        popd > /dev/null

    done
done

