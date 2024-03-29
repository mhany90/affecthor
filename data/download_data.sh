#!/bin/bash
# this script downloads the data to use in our models


# iterate over tasks
for (( t=0; t<${#TASKS[@]}; t++ )); do
    task=${TASKS[$t]}
    type=${TYPES[$t]}

    # rewrite the corresponding directory for each task
    TDIR="$DATADIR/$task"
    if [ -d "$TDIR" ] && [ $newdata -eq 0 ]; then
        continue
    fi
    rm -rf $TDIR
    mkdir -p $TDIR

    # iterate over languages
    for (( l=0; l<${#CODES[@]}; l++ )); do
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

        # test url depends on language
        if [ $code == 'En' ]; then
            TEST_URL="${URL_PREFIX}/AIT2018-TEST-DATA/semeval2018englishtestfiles/${LOCAL_FILE}-test.zip"
        elif [ $code == 'Ar' ]; then
            TEST_URL="${URL_PREFIX}/AIT2018-TEST-DATA/semeval2018arabictestfiles/${LOCAL_FILE}-test.zip"
        elif [ $code == 'Es' ]; then
            TEST_URL="${URL_PREFIX}/AIT2018-TEST-DATA/semeval2018spanishtestfiles/${LOCAL_FILE}-test.zip"
        fi

        # process EI-reg / EI-oc tasks
        if [ $t -eq 0 ] || [ $t -eq 1 ]; then
            # irregular URL in (En, train)
            if [ $l -eq 0 ]; then
                TRAIN_URL="${URL_PREFIX}/$task/$lang/$task-$code-train.zip"
            fi

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
            wget -q "${TEST_URL}"
            unzip "*.zip"

            # clean irrelevant files
            find . -type f -name '*.zip' -delete

            # rename files using our convention
            for s in ${SETS[@]}; do
                mkdir -p $s
                for a in ${AFFECTS[@]}; do
                    if [ -e "${LOCAL_FILE}-$a-$s.txt" ]; then
                        mv -f "${LOCAL_FILE}-$a-$s.txt" "$s/$task-$code-$a-$s.txt"
                    fi
                    if [ -e "$task-$code-$a-$s.txt" ]; then
                        mv -f "$task-$code-$a-$s.txt" "$s/$task-$code-$a-$s.txt"
                    fi
                done
            done

            # combine train and dev
            mkdir -p "traindev"
            for a in ${AFFECTS[@]}; do
			          cp "train/$task-$code-$a-train.txt" "traindev/$task-$code-$a-traindev.txt"
			          sed 1d "dev/$task-$code-$a-dev.txt" >> "traindev/$task-$code-$a-traindev.txt"
            done
        fi

        # process V-reg / V-oc tasks
        if [ $t -eq 2 ] || [ $t -eq 3 ]; then

            # download and extract
            wget -q "${TRAIN_URL}"
            wget -q "${DEV_URL}"
            wget -q "${TEST_URL}"
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

            # combine train and dev
            mkdir -p "traindev"
			      cp "train/$task-$code-valence-train.txt" "traindev/$task-$code-valence-traindev.txt"
			      sed 1d "dev/$task-$code-valence-dev.txt" >> "traindev/$task-$code-valence-traindev.txt"
        fi

        popd > /dev/null

    done
done

